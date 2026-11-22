import { rm } from 'node:fs/promises';
import { getCurrentTools, type TranscriptContext } from '@earendil-works/pi-ai';
import { afterEach, beforeEach, expect, it } from 'vitest';
import {
  estimateContextTokens, estimateMessageTokens, findCutPoint, JsonlTreeSessionStore,
  type AgentCoreEvent, type AgentRuntime, type SessionEntry,
} from '../src/index.js';
import { call, reply, runtime, testDirectory, tool } from './harness-fixtures.js';

let dir: string;
const cores: AgentRuntime[] = [];
beforeEach(async () => { dir = await testDirectory(); });
afterEach(async () => {
  await Promise.all(cores.splice(0).map((core) => core.dispose()));
  await rm(dir, { recursive: true, force: true });
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

it.each([false, true])('retries overflow once after a tool call without replaying it (retry fails: %s)', async (failRetry) => {
  const storage = new JsonlTreeSessionStore(dir);
  let attempts = 0;
  let toolCalls = 0;
  const core = runtime(dir, (model, context) => {
    if (summarizes(context)) return reply(model, text('Recovered context'));
    if (context.messages.filter(({ role }) => role !== 'system').at(-1)?.role === 'toolResult') {
      attempts++;
      if (attempts === 1 || failRetry) throw Error('context_length_exceeded');
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
    ? { state: 'failed', error: 'context_length_exceeded' }
    : { state: 'completed', text: 'Recovered', error: undefined });
  expect(attempts).toBe(2);
  expect(toolCalls).toBe(1);
  expect(events.filter((event) => event.type === 'compaction_start' && event.trigger === 'overflow')).toHaveLength(1);
  const projected = (await storage.projectActiveMessages(session.id)).messages;
  expect(projected.filter((message) => message.role === 'assistant' && message.stopReason === 'error'))
    .toHaveLength(failRetry ? 1 : 0);
});
