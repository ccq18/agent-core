import { createAssistantMessageEventStream } from '@earendil-works/pi-ai';
import { getEventListeners } from 'node:events';
import { lstat, rm } from 'node:fs/promises';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import type { AgentRuntime, AgentRuntimeConfig, PermissionRule } from '../src/index.js';
import { Semaphore } from '../src/runtime/concurrency.js';
import { call, reply, runtime, testDirectory, textResult, tool } from './harness-fixtures.js';

let dir: string;
let core: AgentRuntime | undefined;
beforeEach(async () => { dir = await testDirectory(); });
afterEach(async () => {
  try {
    await core?.dispose();
  } finally {
    core = undefined;
    await rm(dir, { recursive: true, force: true });
    await expect(lstat(dir)).rejects.toMatchObject({ code: 'ENOENT' });
  }
});

const done = { type: 'text' as const, text: 'Done' };
const delegate = (id: string, prompt = 'Child', background = false) =>
  call('subagent', { agent: 'nested', description: id, prompt, background }, id);

it('removes a cancelled waiter and its listener before transferring the slot', async () => {
  const slots = new Semaphore(1);
  const cancelled = new AbortController();
  const next = new AbortController();
  const release = await slots.acquire(next.signal);
  const waiting = slots.acquire(cancelled.signal);
  const successor = slots.acquire(next.signal);
  const reason = new Error('cancel queued operation');
  cancelled.abort(reason);
  await expect(waiting).rejects.toBe(reason);
  expect(getEventListeners(cancelled.signal, 'abort')).toEqual([]);
  release();
  const releaseNext = await successor;
  expect(getEventListeners(next.signal, 'abort')).toEqual([]);
  releaseNext();
  expect(slots.idle).toBe(true);
});

it('transfers one slot only once when a releaser is called twice', async () => {
  const slots = new Semaphore(1);
  const controller = new AbortController();
  const release = await slots.acquire(controller.signal);
  const next = slots.acquire(controller.signal);
  const third = slots.acquire(controller.signal);
  release();
  release();
  const releaseNext = await next;
  controller.abort();
  await expect(third).rejects.toBe(controller.signal.reason);
  releaseNext();
  releaseNext();
  expect(slots.idle).toBe(true);
});

it.each([
  { limits: { maxSubagentsPerRuntime: 3, maxSubagentsPerParent: 2 }, runtimePeak: 3, parentPeak: 2 },
  { limits: { maxSubagentsPerParent: 1 }, runtimePeak: 2, parentPeak: 1 },
  { limits: undefined, runtimePeak: 8, parentPeak: 4 },
])('enforces independent parent and runtime limits with $limits', async ({ limits, runtimePeak, parentPeak }) => {
  const release = Promise.withResolvers<void>();
  let total = 0;
  let peak = 0;
  const active = new Map<string, number>();
  const peaks = new Map<string, number>();
  core = runtime(dir, async (model, context) => {
    const last = context.messages.at(-1);
    if (last?.role !== 'user') return reply(model, done);
    const input = typeof last.content === 'string' ? last.content : last.content[0]?.type === 'text' ? last.content[0].text : '';
    if (input.startsWith('Parent')) {
      return reply(model, ...Array.from({ length: 4 }, (_, i) => delegate(`${input}-${i}`, `Child ${input}`)));
    }
    peak = Math.max(peak, ++total);
    const count = (active.get(input) ?? 0) + 1;
    active.set(input, count);
    peaks.set(input, Math.max(peaks.get(input) ?? 0, count));
    try {
      await release.promise;
      return reply(model, done);
    } finally {
      total--;
      active.set(input, count - 1);
    }
  }, { subagents: { nested: {} }, concurrency: limits });
  const parents = await Promise.all([core.createSession(), core.createSession()]);
  const turns = parents.map((parent, i) => parent.prompt(`Parent ${i}`));
  try {
    await vi.waitFor(() => expect(total).toBe(runtimePeak), { timeout: 600 });
  } finally {
    release.resolve();
    await Promise.allSettled(turns);
  }
  expect(peak).toBe(runtimePeak);
  expect([...peaks.values()].every((n) => n <= parentPeak)).toBe(true);
  expect(Math.max(...peaks.values())).toBe(parentPeak);
  expect(await core.listSessions()).toHaveLength(10);
});

it('rejects nested foreground delegation when two ancestors own all runtime slots', async () => {
  const failures: string[] = [];
  core = runtime(dir, (model, context) => {
    const last = context.messages.at(-1);
    if (last?.role === 'toolResult') {
      for (const message of context.messages) {
        if (message.role === 'toolResult' && message.isError && message.content[0]?.type === 'text') {
          failures.push(message.content[0].text);
        }
      }
      return reply(model, done);
    }
    const input = JSON.stringify(last?.content);
    if (input.includes('Root')) return reply(model, delegate('one', 'First depth'));
    if (input.includes('First depth')) return reply(model, delegate('two', 'Second depth'));
    if (input.includes('Second depth')) return reply(model, delegate('blocked', 'Should not start'));
    throw new Error('Unexpected descendant');
  }, { concurrency: { maxSubagentsPerRuntime: 2 }, maxSubagentDepth: 3, subagents: { nested: {} } });
  const parent = await core.createSession();
  const turn = parent.prompt('Root');
  try {
    await expect(Promise.race([turn, new Promise((resolve) => {
      const timer = setTimeout(() => resolve('blocked'), 500);
      void turn.then(() => clearTimeout(timer), () => clearTimeout(timer));
    })])).resolves.toMatchObject({ state: 'completed' });
  } finally {
    parent.abort();
    await turn.catch(() => undefined);
  }
  expect(failures).toHaveLength(1);
  expect(failures.every((failure) => failure.includes('SUBAGENT_CONCURRENCY_LIMIT'))).toBe(true);
  expect(await core.listSessions()).toHaveLength(3);
});

it('breaks the slot deadlock between parallel foreground branches', async () => {
  const bothRunning = Promise.withResolvers<void>();
  const outcomes: string[] = [];
  let running = 0;
  core = runtime(dir, async (model, { messages }) => {
    const last = messages.at(-1);
    if (last?.role === 'toolResult') {
      if (last.toolName === 'subagent' &&
          messages.some((message) => message.role === 'user' && JSON.stringify(message.content).includes('Branch '))) {
        outcomes.push(last.isError ? JSON.stringify(last.content) : 'grandchild completed');
      }
      return reply(model, done);
    }
    const input = JSON.stringify(last?.content);
    if (input.includes('Root request')) {
      return reply(model, delegate('a', 'Branch A'), delegate('b', 'Branch B'));
    }
    if (input.includes('Branch ')) {
      if (++running === 2) bothRunning.resolve();
      await bothRunning.promise;
      return reply(model, delegate('leaf', 'Grandchild request'));
    }
    return reply(model, done);
  }, { concurrency: { maxSubagentsPerRuntime: 2 }, maxSubagentDepth: 2, subagents: { nested: {} } });
  const root = await core.createSession();
  const turn = root.prompt('Root request');
  try {
    await vi.waitFor(() => expect(outcomes).toEqual([
      expect.stringContaining('SUBAGENT_CONCURRENCY_LIMIT'), 'grandchild completed',
    ]), { timeout: 1000 });
    await expect(turn).resolves.toMatchObject({ state: 'completed' });
    expect(await core.listSessions()).toHaveLength(4);
  } finally {
    bothRunning.resolve();
    root.abort();
    await turn.catch(() => undefined);
  }
});

it('runs a queued background descendant after its parent finishes with one runtime slot', async () => {
  const order: string[] = [];
  core = runtime(dir, (model, { messages }) => {
    const last = messages.at(-1);
    if (last?.role === 'toolResult') {
      if (last.toolCallId === 'background') order.push(last.isError ? 'rejected' : 'parent finished');
      return reply(model, done);
    }
    const input = JSON.stringify(last?.content);
    if (input.includes('Root request')) return reply(model, delegate('a', 'Child request'));
    if (input.includes('Child request')) return reply(model, delegate('background', 'Grandchild request', true));
    if (input.includes('Grandchild request')) {
      order.push('background started');
      return reply(model, { type: 'text', text: 'Grandchild result' });
    }
    if (input.includes('Grandchild result')) order.push('background result delivered');
    return reply(model, done);
  }, { concurrency: { maxSubagentsPerRuntime: 1 }, maxSubagentDepth: 2, subagents: { nested: {} } });
  const root = await core.createSession();
  const turn = root.prompt('Root request');
  try {
    await vi.waitFor(() => expect(order).toEqual([
      'parent finished', 'background started', 'background result delivered',
    ]), { timeout: 1000 });
    await expect(turn).resolves.toMatchObject({ state: 'completed' });
    expect(await core.listSessions()).toHaveLength(3);
  } finally {
    root.abort();
    await turn.catch(() => undefined);
  }
});

it.each([
  { maxSubagentsPerRuntime: 1 },
  { maxSubagentsPerParent: 1 },
  { maxSubagentsPerRuntime: 1, maxSubagentsPerParent: 1 },
])('cancels queued subagents and reuses quota %j without creating cancelled sessions', async (limits) => {
  const entered = Promise.withResolvers<void>();
  const release = Promise.withResolvers<void>();
  core = runtime(dir, async (model, context) => {
    const last = context.messages.at(-1);
    if (last?.role !== 'user') return reply(model, done);
    const input = JSON.stringify(last.content);
    if (input.includes('Holding child')) {
      entered.resolve();
      await release.promise;
    } else if (input.includes('First batch')) {
      return reply(model, delegate('holder', 'Holding child'), delegate('queued-1'), delegate('queued-2'));
    } else if (input.includes('Retry')) {
      return reply(model, delegate('retry', 'Successful child'));
    }
    return reply(model, done);
  }, { concurrency: limits, subagents: { nested: {} } });
  const parent = await core.createSession();
  const first = parent.prompt('First batch');
  try {
    await entered.promise;
    parent.abort();
  } finally {
    release.resolve();
    await first;
  }
  expect(await core.listSessions()).toHaveLength(2);
  await expect(parent.prompt('Retry')).resolves.toMatchObject({ state: 'completed' });
  expect(await core.listSessions()).toHaveLength(3);
});

it('cancels tool waiters immediately but keeps the running callback slot through abort until it settles', async () => {
  const entered = Promise.withResolvers<void>();
  const release = Promise.withResolvers<void>();
  const started: string[] = [];
  core = runtime(dir, (model, context, options) => context.messages.at(-1)?.role === 'user'
    ? reply(model, ...Array.from({ length: options?.sessionId === 'queued' ? 2 : 1 }, (_, i) => call('work', {}, `work-${i}`)))
    : reply(model, done), {
    concurrency: { maxToolExecutions: 1 },
    tools: [tool('work', { async execute(_, { sessionId }) {
      started.push(sessionId);
      if (sessionId === 'holder') {
        entered.resolve();
        await release.promise;
      }
      return textResult('worked');
    } })],
  });
  const [holder, queued, next] = await Promise.all(['holder', 'queued', 'next'].map((id) => core!.createSession({ id })));
  const queuedStarted = vi.fn();
  queued.subscribe((event) => { if (event.type === 'tool_start') queuedStarted(); });
  const holdingTurn = holder.prompt('Hold');
  let queuedTurn: ReturnType<typeof queued.prompt> | undefined;
  let nextTurn: ReturnType<typeof next.prompt> | undefined;
  try {
    await entered.promise;
    queuedTurn = queued.prompt('Queue');
    await vi.waitFor(() => expect(queuedStarted).toHaveBeenCalledTimes(2));
    queued.abort();
    await expect(queuedTurn).resolves.toMatchObject({ state: 'cancelled' });
    holder.abort();
    nextTurn = next.prompt('Retry');
    const enteredNext = vi.fn();
    next.subscribe((event) => { if (event.type === 'tool_start') enteredNext(); });
    await vi.waitFor(() => expect(enteredNext).toHaveBeenCalled());
    expect(started).toEqual(['holder']);
  } finally {
    release.resolve();
    await Promise.allSettled([holdingTurn, queuedTurn, nextTurn]);
  }
  expect(started).toEqual(['holder', 'next']);
});

it('keeps approvals, host beforeToolCall and the built-in question off the ordinary tool quota', async () => {
  const hostEntered = Promise.withResolvers<void>();
  const releaseHost = Promise.withResolvers<void>();
  const ordinary = vi.fn(async () => textResult('worked'));
  const config: Partial<AgentRuntimeConfig> = {
    concurrency: { maxToolExecutions: 1 },
    permissions: [{ action: 'approval', resource: '*', effect: 'ask' }],
    hooks: { async beforeToolCall({ toolCall }) {
      if (toolCall.name === 'host') {
        hostEntered.resolve();
        await releaseHost.promise;
      }
      return undefined;
    } },
    tools: [tool('host'), tool('approval'), tool('ordinary', { execute: ordinary })],
  };
  core = runtime(dir, (model, context, options) => {
    if (context.messages.at(-1)?.role !== 'user') return reply(model, done);
    const name = options?.sessionId ?? '';
    return reply(model, name === 'question' ? call(name, { questions: [{ id: 'q', question: 'Which?' }] }) : call(name));
  }, config);
  const [asking, hosting, questioning, running] = await Promise.all(
    ['approval', 'host', 'question', 'ordinary'].map((id) => core!.createSession({ id })),
  );
  const blocked = [asking.prompt('Approval'), hosting.prompt('Host approval'), questioning.prompt('Question')];
  try {
    await hostEntered.promise;
    await vi.waitFor(() => {
      expect(asking.getPendingApprovals()).toHaveLength(1);
      expect(questioning.getPendingQuestions()).toHaveLength(1);
    });
    expect(core.getCatalog().tools.find(({ name }) => name === 'question')?.description).toContain('on its own');
    await expect(running.prompt('Ordinary')).resolves.toMatchObject({ state: 'completed' });
    expect(ordinary).toHaveBeenCalledOnce();
  } finally {
    asking.abort();
    questioning.abort();
    releaseHost.resolve();
    await Promise.allSettled(blocked);
  }
});

it('denies a queued tool when its resource is forbidden before slot admission', async () => {
  const holderEntered = Promise.withResolvers<void>();
  const release = Promise.withResolvers<void>();
  const queued = Promise.withResolvers<void>();
  const permissions: PermissionRule[] = [{ action: 'write', resource: '/queued', effect: 'allow' }];
  const writes: string[] = [];
  const results: unknown[] = [];
  core = runtime(dir, (model, { messages }, options) => {
    const last = messages.at(-1);
    if (last?.role === 'toolResult') results.push(last);
    if (last?.role !== 'user') return reply(model, done);
    return reply(model, options?.sessionId === 'holder' ? call('hold') : call('write', { path: '/queued' }));
  }, {
    concurrency: { maxToolExecutions: 1 }, permissions,
    tools: [
      tool('hold', { async execute() { holderEntered.resolve(); await release.promise; return textResult('held'); } }),
      tool('write', {
        resolveResources: ({ path }) => [String(path)],
        async execute({ path }) { writes.push(String(path)); return textResult('written'); },
      }),
    ],
  });
  const holder = await core.createSession({ id: 'holder' });
  const waiting = await core.createSession({ id: 'waiting' });
  const holdingTurn = holder.prompt('Hold');
  await holderEntered.promise;
  const acquire = Semaphore.prototype.acquire;
  const admission = vi.spyOn(Semaphore.prototype, 'acquire').mockImplementationOnce(function (this: Semaphore, signal) {
    const pending = acquire.call(this, signal);
    queued.resolve();
    return pending;
  });
  const waitingTurn = waiting.prompt('Write');
  try {
    await queued.promise;
    permissions[0].effect = 'deny';
  } finally {
    release.resolve();
    await Promise.allSettled([holdingTurn, waitingTurn]);
    admission.mockRestore();
  }
  expect(writes).toEqual([]);
  expect(results).toContainEqual(expect.objectContaining({
    toolName: 'write', isError: true,
    content: [expect.objectContaining({ text: expect.stringContaining('Permission denied for "write" (write) on /queued') })],
  }));
});

it('re-evaluates deny after slot admission when permissionAction changes while queued', async () => {
  const holderEntered = Promise.withResolvers<void>();
  const release = Promise.withResolvers<void>();
  const queued = Promise.withResolvers<void>();
  const executions: string[] = [];
  const results: unknown[] = [];
  const mutable = tool('mutable', {
    permissionAction: 'safe',
    async execute() { executions.push('ran'); return textResult('ok'); },
  });
  core = runtime(dir, (model, { messages }, options) => {
    const last = messages.at(-1);
    const toolResult = [...messages].reverse().find((message) => message.role === 'toolResult');
    if (toolResult) results.push(toolResult);
    if (last?.role !== 'user') return reply(model, done);
    return reply(model, options?.sessionId === 'holder' ? call('hold') : call('mutable'));
  }, {
    concurrency: { maxToolExecutions: 1 },
    permissions: [
      { action: 'safe', resource: '*', effect: 'allow' },
      { action: 'forbidden', resource: '*', effect: 'deny' },
    ],
    tools: [
      tool('hold', { async execute() { holderEntered.resolve(); await release.promise; return textResult('held'); } }),
      mutable,
    ],
  });
  const holder = await core.createSession({ id: 'holder' });
  const waiting = await core.createSession({ id: 'waiting' });
  const holdingTurn = holder.prompt('Hold');
  await holderEntered.promise;
  const acquire = Semaphore.prototype.acquire;
  const admission = vi.spyOn(Semaphore.prototype, 'acquire').mockImplementationOnce(function (this: Semaphore, signal) {
    const pending = acquire.call(this, signal);
    queued.resolve();
    return pending;
  });
  const waitingTurn = waiting.prompt('Run mutable');
  try {
    await queued.promise;
    mutable.permissionAction = 'forbidden';
  } finally {
    release.resolve();
    await Promise.allSettled([holdingTurn, waitingTurn]);
    admission.mockRestore();
  }
  expect(executions).toEqual([]);
  expect(results).toContainEqual(expect.objectContaining({
    toolName: 'mutable', isError: true,
    content: [expect.objectContaining({ text: expect.stringContaining('Permission denied for "mutable" (forbidden)') })],
  }));
});

it('keeps background-result followUp turns within maxSubagentsPerRuntime', async () => {
  const grandchildDone = Promise.withResolvers<void>();
  const peerEntered = Promise.withResolvers<void>();
  const releasePeer = Promise.withResolvers<void>();
  let activeSubagentModels = 0;
  let peakSubagentModels = 0;
  core = runtime(dir, (model, { messages }, options) => {
    const id = options?.sessionId ?? '';
    const last = messages.at(-1);
    if (id === 'root-a') {
      return reply(model, last?.role === 'user' ? call('subagent', {
        agent: 'general', description: 'Child one', prompt: 'Start grandchild',
      }) : { type: 'text', text: 'Root A done' });
    }
    if (id === 'root-b') {
      return reply(model, last?.role === 'user' ? call('subagent', {
        agent: 'general', description: 'Peer child', prompt: 'Hold peer slot',
      }) : { type: 'text', text: 'Root B done' });
    }
    activeSubagentModels++;
    peakSubagentModels = Math.max(peakSubagentModels, activeSubagentModels);
    const finish = (text: string) => {
      activeSubagentModels--;
      return reply(model, { type: 'text', text });
    };
    const textOf = (content: unknown) => typeof content === 'string'
      ? content
      : Array.isArray(content) ? content.map((block: { text?: string }) => block.text ?? '').join('') : '';
    const prompt = last?.role === 'user' ? textOf(last.content) : '';
    if (prompt.includes('<background-result')) return finish('FollowUp consumed');
    if (prompt === 'Start grandchild') {
      activeSubagentModels--;
      return reply(model, call('subagent', {
        agent: 'general', description: 'Grandchild', prompt: 'Background work', background: true,
      }));
    }
    if (prompt === 'Background work') {
      grandchildDone.resolve();
      return finish('Grandchild done');
    }
    if (prompt === 'Hold peer slot') {
      peerEntered.resolve();
      const stream = createAssistantMessageEventStream();
      void releasePeer.promise.then(() => {
        activeSubagentModels--;
        stream.push({
          type: 'done', reason: 'stop',
          message: {
            role: 'assistant', content: [{ type: 'text', text: 'Peer done' }],
            api: model.api, provider: model.provider, model: model.id,
            usage: { input: 10, output: 5, cacheRead: 0, cacheWrite: 0, totalTokens: 15, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
            stopReason: 'stop', timestamp: Date.now(),
          },
        });
      });
      return stream;
    }
    return finish('Child one initial');
  }, {
    maxSubagentDepth: 2,
    concurrency: { maxSubagentsPerRuntime: 1 },
    subagents: { general: { deniedTools: [] } },
  });
  const rootA = await core.createSession({ id: 'root-a' });
  const rootB = await core.createSession({ id: 'root-b' });
  const turnA = rootA.prompt('Go A');
  await grandchildDone.promise;
  const turnB = rootB.prompt('Go B');
  try {
    await peerEntered.promise;
    releasePeer.resolve();
    const [resA, resB] = await Promise.all([turnA, turnB]);
    expect(resA.state).toBe('completed');
    expect(resB.state).toBe('completed');
    expect(peakSubagentModels).toBe(1);
  } finally {
    releasePeer.resolve();
    await Promise.allSettled([turnA, turnB]);
  }
});
