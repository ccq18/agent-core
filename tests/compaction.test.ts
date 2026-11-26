import { rm } from 'node:fs/promises';
import { createAssistantMessageEventStream, getCurrentTools, type TranscriptContext } from '@earendil-works/pi-ai';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import {
  estimateContextTokens, estimateMessageTokens, findCutPoint, JsonlTreeSessionStore,
  type AgentCoreEvent, type AgentRuntime, type SessionEntry,
} from '../src/index.js';
import { call, legacyStorage, reply, runtime, testDirectory, tool } from './harness-fixtures.js';

let dir: string;
const cores: AgentRuntime[] = [];
beforeEach(async () => { dir = await testDirectory(); });
afterEach(async () => {
  try {
    await Promise.all(cores.splice(0).map((core) => core.dispose()));
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

const text = (text: string) => ({ type: 'text' as const, text });
const user = (content: string) => ({ role: 'user' as const, content, timestamp: 1 });
const summarizes = ({ messages }: TranscriptContext) =>
  messages.some((message) => message.role === 'system' &&
    typeof message.content === 'string' && message.content.includes('checkpoint summary'));

it('estimates text and images when no provider usage is available', () => {
  expect(estimateMessageTokens({ ...user(''), content: [
    text('12345'), { type: 'image', data: 'fake', mimeType: 'image/png' },
  ] })).toBe(1202);
  expect(estimateContextTokens([user('12345'), user('second')])).toBe(4);
});

it('keeps a trailing tool result with its assistant call and skips a transcript with nothing to summarize', async () => {
  const store = new JsonlTreeSessionStore(dir);
  const { id } = await store.create();
  const older = await store.append(id, { type: 'message', message: user('Older request') });
  const core = runtime(dir, (model, context) => context.messages.at(-1)?.role === 'toolResult'
    ? reply(model, text('Done')) : reply(model, call('lookup')), {
    sessionStorage: store, tools: [tool('lookup', { execute: async () => ({ content: [text('x'.repeat(500))] }) })],
  });
  cores.push(core);
  await (await core.openSession(id)).prompt('Look up');
  const branch = await store.getBranch(id);
  const resultIndex = branch.findIndex((entry) => entry.type === 'message' && entry.message.role === 'toolResult');
  const entries = branch.slice(0, resultIndex + 1);
  const cut = findCutPoint(entries, 20);
  expect(cut).toBeDefined();
  expect(entries[cut!] as SessionEntry).toMatchObject({ type: 'message', message: { role: 'assistant' } });
  expect(findCutPoint([older], 1)).toBeUndefined();
  const assistant = entries[cut!];
  if (assistant.type !== 'message' || assistant.message.role !== 'assistant') throw Error('Expected assistant');
  expect(estimateMessageTokens(assistant.message)).toBe(1);
  expect(estimateContextTokens([
    user('x'.repeat(600)),
    { ...assistant.message, usage: { ...assistant.message.usage, totalTokens: 77 } },
    user('12345'), assistant.message,
  ])).toBe(80);
});

it('saves a manual checkpoint, carries prior summaries forward and projects the same history after reopen', async () => {
  const storage = new JsonlTreeSessionStore(dir);
  const requests: TranscriptContext[] = [];
  const stream: Parameters<typeof runtime>[1] = (model, context) => {
    if (!summarizes(context)) return reply(model, text('Finished'));
    requests.push(context);
    expect(getCurrentTools(context.messages)).toEqual([]);
    return reply(model, text(`Checkpoint ${requests.length}`));
  };
  const config = { sessionStorage: storage, compaction: { reserveTokens: 0, keepRecentTokens: 9 } };
  const core = runtime(dir, stream, config);
  cores.push(core);
  const session = await core.createSession();
  const events: AgentCoreEvent[] = [];
  session.subscribe((event) => events.push(event));
  await session.prompt('Old task detail '.repeat(30));
  await session.prompt('Recent task detail '.repeat(10));
  await expect(session.compact('Preserve file names.')).resolves.toMatchObject({ status: 'compacted', summary: 'Checkpoint 1' });
  expect(JSON.stringify(requests[0].messages)).toContain('Preserve file names.');
  await session.prompt('Newest task detail '.repeat(10));
  await expect(session.compact()).resolves.toMatchObject({ status: 'compacted', summary: 'Checkpoint 2' });
  expect(JSON.stringify(requests[1].messages)).toContain('Checkpoint 1');
  expect(JSON.stringify(requests[1].messages)).not.toContain('Old task detail');
  const projected = (await storage.projectActiveMessages(session.id)).messages;
  expect(projected).toContainEqual(expect.objectContaining({
    role: 'user', content: '<compaction-summary>\nCheckpoint 2\n</compaction-summary>',
  }));
  expect(events.filter(({ type }) => type.startsWith('compaction_')).map(({ type }) => type))
    .toEqual(['compaction_start', 'compaction_end', 'compaction_start', 'compaction_end']);
  await core.dispose();

  const reopened = runtime(dir, (model, context) => {
    expect(JSON.stringify(context.messages)).toContain('Checkpoint 2');
    expect(JSON.stringify(context.messages)).not.toContain('Old task detail');
    return reply(model, text('Reopened'));
  }, { compaction: { enabled: false } });
  cores.push(reopened);
  await expect((await reopened.openSession(session.id)).prompt('Continue')).resolves.toMatchObject({ text: 'Reopened' });
});

it('compacts before a request when the token threshold is exceeded and retains the tools', async () => {
  const requests: TranscriptContext[] = [];
  const core = runtime(dir, (model, context) => {
    requests.push(context);
    return reply(model, text(summarizes(context) ? 'Short summary' : 'Done'));
  }, { compaction: { reserveTokens: 3930, keepRecentTokens: 4 }, tools: [tool('lookup')] });
  cores.push(core);
  const session = await core.createSession();
  const events: AgentCoreEvent[] = [];
  session.subscribe((event) => events.push(event));
  await session.prompt('Previous detail '.repeat(45));
  await expect(session.prompt('Current question')).resolves.toMatchObject({ text: 'Done' });
  expect(requests.map(summarizes)).toEqual([false, true, false]);
  expect(getCurrentTools(requests[2].messages).map(({ name }) => name)).toContain('lookup');
  expect(JSON.stringify(requests[2].messages)).toContain('Short summary');
  expect(JSON.stringify(requests[2].messages)).toContain('Current question');
  expect(events).toContainEqual(expect.objectContaining({ type: 'compaction_start', trigger: 'auto' }));
});

it('uses the configured summary model and bounds the serialized output of each old tool', async () => {
  const stream: Parameters<typeof runtime>[1] = (model, context, options) => {
    const messages = JSON.stringify(context.messages);
    if (summarizes(context)) {
      expect(model.id).toBe('summary');
      expect(options?.toolChoice).toBe('none');
      expect(messages).toContain('x'.repeat(2000));
      expect(messages).not.toContain('x'.repeat(2001));
      return reply(model, text('Summarized tool output'));
    }
    expect(model.id).toBe('test');
    if (!messages.includes('Old lookup') || context.messages.some(({ role }) => role === 'toolResult')) {
      return reply(model, text('Done'));
    }
    return reply(model, call('lookup'));
  };
  const profiles = ['test', 'summary'].map((alias) => ({
    alias, modelId: alias, providerId: 'local', baseUrl: 'http://127.0.0.1:9999/v1',
    protocol: 'openai-completions' as const, contextWindow: 4096, maxTokens: 256,
  }));
  const core = runtime(dir, stream, {
    models: { defaultAlias: 'test', profiles, streamFnOverride: stream },
    compaction: { reserveTokens: 0, keepRecentTokens: 9, modelAlias: 'summary' },
    tools: [tool('lookup', { execute: async () => ({ content: [text('x'.repeat(3000))] }) })],
  });
  cores.push(core);
  const session = await core.createSession();
  await session.prompt('Old lookup');
  await session.prompt('Keep this recent request after compression');
  await expect(session.compact()).resolves.toMatchObject({ status: 'compacted', summary: 'Summarized tool output' });
});

it.each([
  { error: 'context_length_exceeded', failRetry: false },
  { error: 'context_length_exceeded', failRetry: true },
  { error: 'Your input exceeds the context window of this model', failRetry: false },
  { error: 'The input token count (4200) exceeds the maximum number of tokens allowed (4096)', failRetry: false },
  { error: 'maximum context reached', failRetry: false },
])('retries "$error" once after a tool call without replaying it (retry fails: $failRetry)', async ({ error, failRetry }) => {
  const storage = new JsonlTreeSessionStore(dir);
  let attempts = 0;
  let toolCalls = 0;
  const core = runtime(dir, (model, context) => {
    if (summarizes(context)) return reply(model, text('Recovered context'));
    if (context.messages.filter(({ role }) => role !== 'system').at(-1)?.role === 'toolResult') {
      attempts++;
      if (attempts === 1 || failRetry) throw Error(error);
      expect(JSON.stringify(context.messages)).toContain('Recovered context');
      return reply(model, text('Recovered'));
    }
    if (JSON.stringify(context.messages).includes('Trigger retry')) return reply(model, call('lookup'));
    return reply(model, text('Done'));
  }, {
    sessionStorage: storage, compaction: { reserveTokens: 0, keepRecentTokens: 10 },
    tools: [tool('lookup', { execute: async () => {
      toolCalls++;
      return { content: [text('tool result')] };
    } })],
  });
  cores.push(core);
  const session = await core.createSession();
  const events: AgentCoreEvent[] = [];
  session.subscribe((event) => events.push(event));
  await session.prompt('Old history '.repeat(30));
  const result = await session.prompt('Trigger retry');
  expect(result).toMatchObject(failRetry
    ? { state: 'failed', error }
    : { state: 'completed', text: 'Recovered', error: undefined });
  expect(attempts).toBe(2);
  expect(toolCalls).toBe(1);
  expect(events.filter((event) => event.type === 'compaction_start' && event.trigger === 'overflow')).toHaveLength(1);
  const projected = (await storage.projectActiveMessages(session.id)).messages;
  expect(projected.filter((message) => message.role === 'assistant' && message.stopReason === 'error'))
    .toHaveLength(failRetry ? 1 : 0);
});

it.each(['stop', 'length'] as const)('compacts a silent %s overflow using the prepared context window', async (reason) => {
  let requests = 0;
  let summaries = 0;
  const core = runtime(dir, async (model, context) => {
    if (summarizes(context)) {
      summaries++;
      return reply(model, text('Checkpoint'));
    }
    if (++requests !== 2) return reply(model, text('Done'));
    const message = await reply(model).result();
    message.usage.input = reason === 'stop' ? 1001 : 1000;
    message.stopReason = reason;
    const stream = createAssistantMessageEventStream();
    stream.push({ type: 'done', reason, message });
    return stream;
  }, {
    hooks: { prepareRequest: ({ model }) => ({ model: { ...model, contextWindow: 1000 } }) },
    compaction: { reserveTokens: 0, keepRecentTokens: 10 },
  });
  cores.push(core);
  const session = await core.createSession();
  await session.prompt('Old history '.repeat(30));
  await expect(session.prompt('Try again')).resolves.toMatchObject({ state: 'completed', text: 'Done' });
  expect(summaries).toBe(1);
  expect(requests).toBe(3);
});

it.each([
  'Rate limit exceeded: maximum context reached',
  '429: too many tokens',
  'API 429: context_length_exceeded',
  'OpenAI API error (429): exceeds the context window',
  'rate_limit_exceeded: too many tokens',
])('returns a rate limit error without compacting or retrying: %s', async (error) => {
  let requests = 0;
  let summaries = 0;
  const core = runtime(dir, (model, context) => {
    if (summarizes(context)) {
      summaries++;
      return reply(model, text('Checkpoint'));
    }
    if (++requests > 1) throw Error(error);
    return reply(model, text('Done'));
  }, { compaction: { reserveTokens: 0, keepRecentTokens: 10 } });
  cores.push(core);
  const session = await core.createSession();
  await session.prompt('Old history '.repeat(30));
  await expect(session.prompt('Next')).resolves.toMatchObject({ state: 'failed', error });
  expect(summaries).toBe(0);
  expect(requests).toBe(2);
});

it('shrinks an overflowing summary through 85%, 70%, 50% and 35% while keeping the checkpoint and latest exchange', async () => {
  const storage = new JsonlTreeSessionStore(dir);
  const history: string[] = [];
  const core = runtime(dir, (model, context) => {
    const last = context.messages.at(-1);
    if (summarizes(context) && last?.role === 'user' && typeof last.content === 'string') {
      history.push(last.content);
      if (history.length < 5) throw new Error('context_length_exceeded');
      return reply(model, text('Reduced checkpoint'));
    }
    const latest = last?.role === 'user' ? last.content : undefined;
    const input = Array.isArray(latest) ? latest.find((block) => block.type === 'text')?.text : latest;
    if (input?.startsWith('old lookup')) {
      return reply(model, call('lookup', { query: 'argument'.repeat(300) }));
    }
    return reply(model, text('Done'));
  }, {
    sessionStorage: storage,
    compaction: { keepRecentTokens: 40, reserveTokens: 0 },
    tools: [tool('lookup', { execute: async () => ({ content: [text('r'.repeat(3000))] }) })],
  });
  cores.push(core);
  const header = await storage.create();
  await storage.append(header.id, {
    type: 'compaction', summary: 'Previous summary to keep', firstKeptEntryId: null, tokensBefore: 100,
  });
  const session = await core.openSession(header.id);
  await session.prompt('old lookup ' + 'Old detail '.repeat(350));
  await session.prompt('Latest summarized exchange');
  await session.prompt('Recent retained request '.repeat(12));
  await expect(session.compact()).resolves.toMatchObject({ status: 'compacted', summary: 'Reduced checkpoint' });

  expect(history).toHaveLength(5);
  expect(history[0]).toContain('r'.repeat(2000));
  expect(history[1]).toContain('r'.repeat(800));
  expect(history[1]).not.toContain('r'.repeat(801));
  expect(history[2]).toContain('[toolResult omitted]');
  expect(history[2]).not.toContain('argument'.repeat(300));
  expect(history[4]).not.toContain('Old detail');
  for (const [index, ratio] of [0.85, 0.70, 0.50, 0.35].entries()) {
    expect(history[index + 1].length).toBeLessThanOrEqual(Math.floor(history[0].length * ratio));
    expect(history[index + 1].length).toBeLessThan(history[index].length);
    expect(history[index + 1]).toContain('Previous summary to keep');
    expect(history[index + 1]).toContain('Latest summarized exchange');
  }
});

it.each(['reuse', 'branch', 'pending', 'abort', 'close'] as const)('handles background precomputation at the auto cut (%s)', async (outcome) => {
  const storage = new JsonlTreeSessionStore(dir);
  const started = Promise.withResolvers<void>();
  const summaries: TranscriptContext[] = [];
  const requests: TranscriptContext[] = [];
  let backgroundSignal: AbortSignal | undefined;
  const core = runtime(dir, async (model, context, options) => {
    if (summarizes(context)) {
      summaries.push(context);
      if (summaries.length === 1) {
        backgroundSignal = options?.signal;
        started.resolve();
        if (outcome === 'pending') return createAssistantMessageEventStream();
      }
      return reply(model, text(summaries.length === 1 ? 'Background checkpoint' : 'Fresh checkpoint'));
    }
    requests.push(context);
    const response = reply(model, text('Done'));
    (await response.result()).usage.totalTokens = requests.length === 1 ? 700 : requests.length === 2 ? 1400 : 0;
    return response;
  }, { sessionStorage: storage, compaction: {
    reserveTokens: 2496, keepRecentTokens: 15, precompute: true, precomputeLeadTokens: 400,
  } });
  cores.push(core);
  let session = await core.createSession();
  const events: AgentCoreEvent[] = [];
  session.subscribe((event) => events.push(event));
  await session.prompt('Old history '.repeat(20));
  await session.prompt('Second request '.repeat(25));
  await started.promise;
  await new Promise<void>((resolve) => setImmediate(resolve));
  const before = await storage.getBranch(session.id);
  expect(summaries).toHaveLength(1);
  expect(before.some(({ type }) => type === 'compaction')).toBe(false);
  expect(events.some(({ type }) => type.startsWith('compaction_'))).toBe(false);
  if (outcome === 'branch') await session.branch(before.at(-2)!.id);
  if (outcome === 'abort') session.abort();
  if (outcome === 'close') {
    await session.close();
    session = await core.openSession(session.id);
  }

  await expect(session.prompt('New request '.repeat(outcome === 'branch' ? 310 : 100)))
    .resolves.toMatchObject({ state: 'completed' });
  expect(summaries).toHaveLength(outcome === 'reuse' ? 1 : 2);
  expect(JSON.stringify(requests.at(-1)?.messages)).toContain(outcome === 'reuse' ? 'Background checkpoint' : 'Fresh checkpoint');
  expect(JSON.stringify(requests.at(-1)?.messages)).toContain('New request');
  if (outcome === 'reuse') expect(JSON.stringify(requests.at(-1)?.messages)).toContain('Second request');
  if (outcome === 'pending' || outcome === 'abort' || outcome === 'close') {
    expect(backgroundSignal?.aborted).toBe(true);
  }
});

it('keeps summary precomputation disabled by default', async () => {
  const summaries: TranscriptContext[] = [];
  const core = runtime(dir, (model, context) => {
    if (summarizes(context)) summaries.push(context);
    return reply(model, text('Done'));
  }, { compaction: { reserveTokens: 1000, keepRecentTokens: 5 } });
  cores.push(core);
  const session = await core.createSession();
  await session.prompt('First turn '.repeat(30));
  await session.prompt('Second turn '.repeat(30));
  await new Promise<void>((resolve) => setImmediate(resolve));
  expect(summaries).toHaveLength(0);
});

it.each([
  { message: 'context_length_exceeded', attempts: 5 },
  { message: 'rate limit: too many tokens', attempts: 1 },
])('fails without writing a checkpoint on "$message"', async ({ message, attempts }) => {
  const storage = new JsonlTreeSessionStore(dir);
  let calls = 0;
  const core = runtime(dir, (model, context) => {
    if (summarizes(context)) {
      calls++;
      throw new Error(message);
    }
    return reply(model, text('Done'));
  }, { sessionStorage: storage, compaction: { keepRecentTokens: 5, reserveTokens: 0 } });
  cores.push(core);
  const session = await core.createSession();
  await session.prompt('Old material '.repeat(30));
  await session.prompt('Latest material to keep in context');
  await expect(session.compact()).resolves.toEqual({ status: 'failed', error: message });
  expect(calls).toBe(attempts);
  expect((await storage.getBranch(session.id)).some(({ type }) => type === 'compaction')).toBe(false);
});

it('uses the actual summary model budget on the first request and drains the returned event stream', async () => {
  const lengths: number[] = [];
  let drained = 0;
  const streamOverride: Parameters<typeof runtime>[1] = (model, context) => {
    if (!summarizes(context)) return reply(model, text('Done'));
    const last = context.messages.at(-1);
    expect(model.id).toBe('small');
    if (last?.role !== 'user' || typeof last.content !== 'string') throw Error('Missing summary input');
    lengths.push(last.content.length);
    expect(last.content).toContain('Newest complete exchange');
    expect(last.content).toContain('Follow this instruction');
    const stream = reply(model, text('Summary'));
    const iterate = stream[Symbol.asyncIterator].bind(stream);
    stream[Symbol.asyncIterator] = async function* () {
      for await (const event of { [Symbol.asyncIterator]: iterate }) {
        drained++;
        yield event;
      }
    };
    return stream;
  };
  const core = runtime(dir, streamOverride, {
    models: {
      defaultAlias: 'main',
      profiles: ['main', 'small'].map((alias) => ({
        alias, modelId: alias, providerId: 'local', protocol: 'openai-completions',
        baseUrl: 'http://127.0.0.1:9999/v1', contextWindow: alias === 'small' ? 1024 : 32768, maxTokens: 256,
      })),
      streamFnOverride: streamOverride,
    },
    compaction: { modelAlias: 'small', keepRecentTokens: 40 },
  });
  cores.push(core);
  const session = await core.createSession();
  await session.prompt('Very old detail '.repeat(800));
  await session.prompt('Newest complete exchange');
  await session.prompt('Retained request '.repeat(30));
  await expect(session.compact('Follow this instruction')).resolves.toMatchObject({ status: 'compacted' });
  expect(lengths).toHaveLength(1);
  expect(lengths[0]).toBeLessThan((1024 - 256) * 4);
  expect(drained).toBe(1);
});

it.each([
  new Error('API 429: context_length_exceeded'),
  new Error('HTTP 429: context_length_exceeded'),
  new Error('status code 429: context_length_exceeded'),
  new Error('error=429: context_length_exceeded'),
  new Error('code: "429" context_length_exceeded'),
  Object.assign(new Error('context_length_exceeded'), { status: 429 }),
  new Error('Rate_limit: too many tokens'),
])('does not shrink or retry when the summary provider returns %s', async (error) => {
  let requests = 0;
  const core = runtime(dir, (model, context) => {
    if (!summarizes(context)) return reply(model, text('Done'));
    requests++;
    throw error;
  }, { compaction: { keepRecentTokens: 4, reserveTokens: 0 } });
  cores.push(core);
  const session = await core.createSession();
  await session.prompt('Old conversation');
  await session.prompt('Recent conversation');
  await expect(session.compact()).resolves.toMatchObject({ status: 'failed' });
  expect(requests).toBe(1);
});

it.each(['window', 'legacy'] as const)('rejects a stale compaction at commit through %s storage', async (kind) => {
  const store = new JsonlTreeSessionStore(dir);
  const getBranch = vi.spyOn(store, 'getBranch');
  const getWindow = vi.spyOn(store, 'getCompactionWindow');
  const { id } = await store.create();
  const core = runtime(dir, async (model, context) => {
    if (!summarizes(context)) return reply(model, text('Done'));
    await store.append(id, { type: 'metadata', metadata: { title: 'Updated during summary' } });
    return reply(model, text('Stale checkpoint'));
  }, { sessionStorage: kind === 'window' ? store : legacyStorage(store), compaction: { keepRecentTokens: 4, reserveTokens: 0 } });
  cores.push(core);
  const session = await core.openSession(id);
  await session.prompt('Old conversation');
  await session.prompt('Recent conversation');
  await expect(session.compact()).resolves.toEqual({ status: 'failed', error: 'Session branch changed during compaction' });
  expect(getWindow).toHaveBeenCalledTimes(kind === 'window' ? 1 : 0);
  expect(getBranch).toHaveBeenCalledTimes(kind === 'window' ? 0 : 2);
  expect((await store.getBranch(id)).some(({ type }) => type === 'compaction')).toBe(false);
});

it('does not append through a legacy adapter if cancellation arrives during the final branch read', async () => {
  const store = new JsonlTreeSessionStore(dir);
  const storage = legacyStorage(store);
  const core = runtime(dir, (model, context) =>
    reply(model, text(summarizes(context) ? 'Checkpoint' : 'Done')),
  { sessionStorage: storage, compaction: { keepRecentTokens: 4, reserveTokens: 0 } });
  cores.push(core);
  const session = await core.createSession();
  await session.prompt('Old conversation');
  await session.prompt('Recent conversation');
  const read = storage.getBranch;
  vi.spyOn(storage, 'getBranch').mockImplementationOnce(read).mockImplementationOnce(async (id) => {
    const branch = await read(id);
    session.abort();
    return branch;
  });
  await expect(session.compact()).resolves.toEqual({ status: 'skipped', reason: 'aborted' });
  expect((await store.getBranch(session.id)).some(({ type }) => type === 'compaction')).toBe(false);
});

it('forwards cancellation to the summary producer and drains its final abort event', async () => {
  const storage = new JsonlTreeSessionStore(dir);
  const started = Promise.withResolvers<AbortSignal>();
  const drained = Promise.withResolvers<void>();
  const core = runtime(dir, async (model, context, options) => {
    if (!summarizes(context)) return reply(model, text('Done'));
    const stream = createAssistantMessageEventStream();
    const partial = await reply(model, text('')).result();
    stream.push({ type: 'start', partial });
    const iterate = stream[Symbol.asyncIterator].bind(stream);
    stream[Symbol.asyncIterator] = async function* () {
      for await (const event of { [Symbol.asyncIterator]: iterate }) {
        yield event;
      }
      drained.resolve();
    };
    const signal = options!.signal!;
    signal.addEventListener('abort', () => {
      stream.push({ type: 'error', reason: 'aborted', error: { ...partial, stopReason: 'aborted' } });
    }, { once: true });
    started.resolve(signal);
    return stream;
  }, { sessionStorage: storage, compaction: { keepRecentTokens: 4, reserveTokens: 0 } });
  cores.push(core);
  const session = await core.createSession();
  await session.prompt('Old conversation');
  await session.prompt('Recent conversation');
  const pending = session.compact();
  const checked = expect(pending).resolves.toEqual({ status: 'skipped', reason: 'aborted' });
  const signal = await started.promise;
  await new Promise<void>((resolve) => setImmediate(resolve));
  session.abort();
  await checked;
  await drained.promise;
  expect(signal.aborted).toBe(true);
  expect((await storage.getBranch(session.id)).some(({ type }) => type === 'compaction')).toBe(false);
});
