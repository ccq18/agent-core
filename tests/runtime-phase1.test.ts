import { rm } from 'node:fs/promises';
import type { StreamFn } from '@earendil-works/pi-agent-core';
import {
  createAssistantMessageEventStream,
  type Api, type AssistantMessage, type Model, type TranscriptContext,
} from '@earendil-works/pi-ai';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  createAgentRuntime, JsonlTreeSessionStore,
  type AgentCoreEvent, type AgentRuntime, type AgentRuntimeConfig, type CoreToolDefinition,
} from '../src/index.js';
import { testDirectory } from './harness-fixtures.js';

const profile = {
  alias: 'test', providerId: 'test', modelId: 'test-model', protocol: 'openai-completions' as const,
  baseUrl: 'http://127.0.0.1:9999/v1', contextWindow: 10000, maxTokens: 1000, reasoning: true,
};
const usage = { input: 7, output: 3, cacheRead: 2, cacheWrite: 0, totalTokens: 12,
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } };

function assistant(
  model: Model<Api>, content: AssistantMessage['content'], stopReason: AssistantMessage['stopReason'] = 'stop',
): AssistantMessage {
  return { role: 'assistant', model: model.id, provider: model.provider, api: model.api,
    content, stopReason, usage, timestamp: Date.now() };
}

function reply(model: Model<Api>, text: string) {
  const stream = createAssistantMessageEventStream();
  stream.push({ type: 'done', reason: 'stop', message: assistant(model, [{ type: 'text', text }]) });
  return stream;
}

function userTexts(context: Pick<TranscriptContext, 'messages'>) {
  return context.messages.flatMap((message) => message.role !== 'user' ? [] :
    typeof message.content === 'string' ? [message.content] :
      message.content.flatMap((block) => block.type === 'text' ? [block.text] : []));
}

let dataDir: string;
let runtimes: AgentRuntime[];
function runtime(streamFnOverride: StreamFn, options: Partial<AgentRuntimeConfig> = {}) {
  const result = createAgentRuntime({
    dataDir, models: { defaultAlias: 'test', profiles: [profile], streamFnOverride }, ...options,
  });
  runtimes.push(result);
  return result;
}

beforeEach(async () => {
  dataDir = await testDirectory();
  runtimes = [];
});

afterEach(async () => {
  try {
    await Promise.all(runtimes.map((item) => item.dispose()));
  } finally {
    await rm(dataDir, { recursive: true, force: true });
  }
});

describe('Phase 1 runtime', () => {
  it('maps Pi thinking and text blocks, associates the run and keeps seq increasing across prompts', async () => {
    const core = runtime((model) => {
      const stream = createAssistantMessageEventStream();
      const partial = assistant(model, [
        { type: 'thinking', thinking: 'consider it' }, { type: 'text', text: 'hello' },
      ]);
      stream.push({ type: 'start', partial });
      stream.push({ type: 'thinking_start', contentIndex: 0, partial });
      stream.push({ type: 'thinking_delta', contentIndex: 0, delta: 'consider it', partial });
      stream.push({ type: 'thinking_end', contentIndex: 0, content: 'consider it', partial });
      stream.push({ type: 'text_start', contentIndex: 1, partial });
      stream.push({ type: 'text_delta', contentIndex: 1, delta: 'hel', partial });
      stream.push({ type: 'text_delta', contentIndex: 1, delta: 'lo', partial });
      stream.push({ type: 'text_end', contentIndex: 1, content: 'hello', partial });
      stream.push({ type: 'done', reason: 'stop', message: partial });
      return stream;
    });
    const session = await core.createSession({ mode: 'plan' });
    const events: AgentCoreEvent[] = [];
    const unsubscribe = session.subscribe((event) => events.push(event));
    const summary = await session.prompt('Question');
    expect(summary).toMatchObject({ sessionId: session.id, state: 'completed', text: 'hello', usage });
    expect(events).toMatchObject([
      { type: 'turn_start', mode: 'plan', modelId: 'test-model' },
      { type: 'thinking_start', blockIndex: 0 },
      { type: 'thinking_delta', blockIndex: 0, delta: 'consider it' },
      { type: 'thinking_end', blockIndex: 0, content: 'consider it' },
      { type: 'text_start', blockIndex: 1 },
      { type: 'text_delta', blockIndex: 1, delta: 'hel' },
      { type: 'text_delta', blockIndex: 1, delta: 'lo' },
      { type: 'text_end', blockIndex: 1, content: 'hello' },
      { type: 'turn_end', state: 'completed', usage },
    ]);
    expect(events.every((event) => event.runId === summary.runId && event.sessionId === session.id)).toBe(true);
    const next = await session.prompt('Another question');
    expect(next.runId).not.toBe(summary.runId);
    expect(events.map((event) => event.seq)).toEqual(events.map((_, i) => i + 1));
    unsubscribe();
    const count = events.length;
    await session.prompt('Unsubscribed');
    expect(events).toHaveLength(count);
  });

  it('executes tools, streams progress and sends thrown errors back to the model', async () => {
    const executed = vi.fn<CoreToolDefinition['execute']>(async (params, ctx) => {
      ctx.onUpdate({ content: [{ type: 'text', text: 'working' }], details: { progress: 1 } });
      return { content: [{ type: 'text', text: String(params.value) }] };
    });
    const tools: CoreToolDefinition[] = [
      { name: 'echo', label: 'Echo value', description: 'Echo the input', execute: executed,
        parameters: { type: 'object', properties: { value: { type: 'string' } }, required: ['value'] } },
      { name: 'broken', label: 'Broken', description: 'Fails in the tool', parameters: { type: 'object' },
        async execute() { throw new Error('broken tool'); } },
    ];
    const core = runtime((model, context) => {
      const results = context.messages.filter((message) => message.role === 'toolResult');
      if (results.length) {
        expect(results).toMatchObject([
          { toolCallId: 'call-echo', content: [{ type: 'text', text: 'hello' }], isError: false },
          { toolCallId: 'call-broken', content: [{ type: 'text', text: 'broken tool' }], isError: true },
        ]);
        return reply(model, 'Recovered');
      }
      const stream = createAssistantMessageEventStream();
      stream.push({ type: 'done', reason: 'toolUse', message: assistant(model, [
        { type: 'toolCall', id: 'call-echo', name: 'echo', arguments: { value: 'hello' } },
        { type: 'toolCall', id: 'call-broken', name: 'broken', arguments: {} },
      ], 'toolUse') });
      return stream;
    }, { tools });
    const session = await core.createSession();
    const events: AgentCoreEvent[] = [];
    session.subscribe((event) => events.push(event));
    const summary = await session.prompt('Run both tools');
    expect(summary).toMatchObject({ state: 'completed', text: 'Recovered', usage: { totalTokens: 24 } });
    expect(executed).toHaveBeenCalledWith({ value: 'hello' }, expect.objectContaining({
      sessionId: session.id, runId: summary.runId, toolCallId: 'call-echo', signal: expect.any(AbortSignal),
    }));
    expect(events).toEqual(expect.arrayContaining([
      expect.objectContaining({ type: 'tool_start', toolName: 'echo', label: 'Echo value' }),
      expect.objectContaining({ type: 'tool_update', partial: 'working', details: { progress: 1 } }),
      expect.objectContaining({ type: 'tool_end', toolName: 'echo', isError: false, durationMs: expect.any(Number) }),
      expect.objectContaining({ type: 'tool_end', toolName: 'broken', isError: true, summary: 'broken tool' }),
    ]));
    expect(events.filter((event) => event.type === 'turn_start' || event.type === 'turn_end')
      .map((event) => event.type)).toEqual(['turn_start', 'turn_end', 'turn_start', 'turn_end']);
  });

  it('propagates abort to the stream signal and emits a cancelled terminal turn', async () => {
    const started = Promise.withResolvers<AbortSignal>();
    const core = runtime((model, _context, options) => {
      const stream = createAssistantMessageEventStream();
      const signal = options?.signal;
      if (!signal) throw new Error('Missing abort signal');
      signal.addEventListener('abort', () => stream.push({
        type: 'error', reason: 'aborted', error: { ...assistant(model, [], 'aborted'), errorMessage: 'cancelled' },
      }), { once: true });
      started.resolve(signal);
      return stream;
    });
    const session = await core.createSession();
    const events: AgentCoreEvent[] = [];
    session.subscribe((event) => events.push(event));
    const pending = session.prompt('Wait for abort');
    const signal = await started.promise;
    session.abort();
    expect(signal.aborted).toBe(true);
    await expect(pending).resolves.toMatchObject({ state: 'cancelled' });
    expect(events.at(-1)).toMatchObject({ type: 'turn_end', state: 'cancelled' });
  });

  it('settles an abort while an asynchronous stream factory is still opening', async () => {
    const started = Promise.withResolvers<void>();
    const core = runtime(() => {
      started.resolve();
      return new Promise(() => {});
    });
    const session = await core.createSession();
    const pending = session.prompt('Wait for the connection');
    await started.promise;
    session.abort();
    await expect(pending).resolves.toMatchObject({ state: 'cancelled' });
  });

  it('drains steering before follow-up and starts a new run for an idle follow-up', async () => {
    const started = Promise.withResolvers<() => void>();
    const seen: string[][] = [];
    const core = runtime((model, context) => {
      seen.push(userTexts(context));
      if (seen.length > 1) return reply(model, 'next');
      const stream = createAssistantMessageEventStream();
      started.resolve(() => stream.push({
        type: 'done', reason: 'stop', message: assistant(model, [{ type: 'text', text: 'first' }]),
      }));
      return stream;
    });
    const session = await core.createSession();
    const pending = session.prompt('start');
    const resume = await started.promise;
    session.followUp('later');
    session.steer('first correction');
    resume();
    await expect(pending).resolves.toMatchObject({ state: 'completed', usage: { totalTokens: 36 } });
    const idleFinished = Promise.withResolvers<void>();
    session.subscribe((event) => {
      if (event.type === 'turn_end') idleFinished.resolve();
    });
    session.followUp('idle follow-up');
    await idleFinished.promise;
    await core.dispose();
    expect(seen).toEqual([
      ['start'], ['start', 'first correction'], ['start', 'first correction', 'later'],
      ['start', 'first correction', 'later', 'idle follow-up'],
    ]);
  });

  it('persists full history, restores mode and model, and continues after opening a fresh runtime', async () => {
    const stream = vi.fn<StreamFn>((model) => reply(model, 'persisted answer'));
    const config = { defaultAlias: 'test', profiles: [profile, { ...profile, alias: 'fast', modelId: 'fast-model' }],
      streamFnOverride: stream };
    const core = runtime(stream, { baseSystemPrompt: 'Be concise', models: config });
    const session = await core.createSession({ title: 'History' });
    session.setMode('plan');
    await session.prompt('First', { modelAlias: 'fast' });
    await session.prompt('Second');
    await core.dispose();

    const reopened = runtime(stream, { baseSystemPrompt: 'Be concise', models: config });
    const restored = await reopened.openSession(session.id);
    expect(await reopened.openSession(session.id)).toBe(restored);
    expect(restored.mode).toBe('plan');
    expect(await reopened.listSessions()).toMatchObject([{ id: session.id, title: 'History', modelAlias: 'fast' }]);
    await restored.prompt('Third');
    const [model, { messages }] = stream.mock.calls.at(-1)!;
    expect(model.id).toBe('fast-model');
    expect(messages.map((message) => message.role)).toEqual(['system', 'system', 'user', 'assistant', 'user', 'assistant', 'user', 'user']);
    expect(userTexts({ messages })).toEqual(['First', 'Second', expect.stringContaining('Plan mode is active'), 'Third']);
    await reopened.deleteSession(restored.id);
    expect(await reopened.listSessions()).toEqual([]);
  });

  it('branches an open session and resumes with the selected transcript, mode and model', async () => {
    const storage = new JsonlTreeSessionStore(dataDir);
    const stream = vi.fn<StreamFn>((model) => reply(model, 'answer'));
    const core = runtime(stream, {
      sessionStorage: storage,
      models: {
        defaultAlias: 'test',
        profiles: [profile, { ...profile, alias: 'fast', modelId: 'fast-model' }],
        streamFnOverride: stream,
      },
    });
    const session = await core.createSession();
    await session.prompt('Root');
    const root = (await storage.getBranch(session.id)).at(-1)!;
    await session.setMode('plan');
    await session.prompt('Original branch', { modelAlias: 'fast' });
    const formerLeaf = (await storage.getBranch(session.id)).at(-1)!;

    await session.branch(root.id);
    expect(session.mode).toBe('build');
    await session.prompt('Alternative branch');
    const [model, transcript] = stream.mock.calls.at(-1)!;
    expect(model.id).toBe('test-model');
    expect(userTexts(transcript)).toEqual(['Root', 'Alternative branch']);
    expect(await storage.getBranch(session.id, formerLeaf.id)).toEqual(expect.arrayContaining([
      expect.objectContaining({ type: 'model', modelAlias: 'fast' }),
    ]));
  });

  it('retains the selected alias credentials when prepareRequest clones a shared gateway model', async () => {
    const first = vi.fn(() => 'fixture-first');
    const second = vi.fn(() => 'fixture-second');
    const stream = vi.fn<StreamFn>((model) => reply(model, 'ok'));
    const core = runtime(stream, {
      models: { defaultAlias: 'first', profiles: [
        { ...profile, alias: 'first', apiKey: first }, { ...profile, alias: 'second', apiKey: second },
      ], streamFnOverride: stream },
      hooks: { prepareRequest: ({ model }) => ({ model: { ...structuredClone(model), maxTokens: 42 } }) },
    });
    const session = await core.createSession();
    await expect(session.prompt('Hello', { modelAlias: 'second' })).resolves.toMatchObject({ state: 'completed' });
    expect(second).toHaveBeenCalledOnce();
    expect(first).not.toHaveBeenCalled();
    expect(stream.mock.calls[0][0].maxTokens).toBe(42);
  });

  it('applies runtime hooks to registered tools and requests for an already open session', async () => {
    const before = vi.fn();
    const stream = vi.fn<StreamFn>((model, transcript, options) => {
      expect(options?.reasoning).toBe('high');
      expect(userTexts(transcript)).toContain('request-only note');
      if (transcript.messages.some((message) => message.role === 'toolResult')) return reply(model, 'done');
      const stream = createAssistantMessageEventStream();
      stream.push({ type: 'done', reason: 'toolUse', message: assistant(model, [
        { type: 'toolCall', id: 'hook-call', name: 'hooked', arguments: { value: 'test' } },
      ], 'toolUse') });
      return stream;
    });
    const core = runtime(stream, {
      hooks: {
        transformContext: async (messages) =>
          [...messages, { role: 'user', content: 'request-only note', timestamp: Date.now() }],
        prepareRequest: () => ({ thinkingLevel: 'high' }),
        beforeToolCall: async ({ toolCall }) => {
          before(toolCall.name);
          return undefined;
        },
        afterToolCall: async () => ({ content: [{ type: 'text', text: 'rewritten' }] }),
      },
    });
    const session = await core.createSession();
    const tool: CoreToolDefinition<{ value: string }> = {
      name: 'hooked', label: 'Hooked', description: 'Used by the hook test',
      parameters: { type: 'object', properties: { value: { type: 'string' } }, required: ['value'] },
      resolveResources: ({ value }) => [value],
      async execute({ value }) { return { content: [{ type: 'text', text: value }] }; },
    };
    const builtIns = core.getCatalog().tools;
    const unregister = core.registerTools([tool]);
    expect(core.getCatalog().tools).toMatchObject([...builtIns, { name: 'hooked' }]);
    const events: AgentCoreEvent[] = [];
    session.subscribe((event) => events.push(event));
    await expect(session.prompt('Use hooks')).resolves.toMatchObject({ state: 'completed', text: 'done' });
    expect(before).toHaveBeenCalledExactlyOnceWith('hooked');
    expect(events).toContainEqual(expect.objectContaining({ type: 'tool_end', summary: 'rewritten' }));
    unregister();
    expect(core.getCatalog().tools).toEqual(builtIns);
  });

  it('maps provider failure to a failed turn and leaves the session available for retry', async () => {
    const stream = vi.fn<StreamFn>((model) => reply(model, 'retry succeeded'))
      .mockImplementationOnce((model) => {
        const failed = createAssistantMessageEventStream();
        failed.push({
          type: 'error', reason: 'error', error: { ...assistant(model, [], 'error'), errorMessage: 'provider failed' },
        });
        return failed;
      });
    const session = await runtime(stream).createSession();
    const events: AgentCoreEvent[] = [];
    session.subscribe((event) => events.push(event));
    await expect(session.prompt('First try')).resolves.toMatchObject({ state: 'failed', error: 'provider failed' });
    expect(events.at(-1)).toMatchObject({ type: 'turn_end', state: 'failed', error: 'provider failed' });
    await expect(session.prompt('Retry')).resolves.toMatchObject({ state: 'completed', text: 'retry succeeded' });
  });
});
