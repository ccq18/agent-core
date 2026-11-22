import { rm } from 'node:fs/promises';
import { getCurrentSystemPrompt, getCurrentTools } from '@earendil-works/pi-ai';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { JsonlTreeSessionStore, type AgentCoreEvent, type AgentRuntime } from '../src/index.js';
import { call, reply, runtime, testDirectory, textResult, tool } from './harness-fixtures.js';

let dir: string;
let core: AgentRuntime | undefined;
beforeEach(async () => { dir = await testDirectory(); });
afterEach(async () => {
  await core?.dispose();
  core = undefined;
  await rm(dir, { recursive: true, force: true });
});

const text = (text: string) => ({ type: 'text' as const, text });
const delegate = (agent: string, overrides = {}) => call('subagent', {
  agent, prompt: 'Child task with explicit context', description: 'Inspect', ...overrides,
}, `delegate-${agent}`);

it('delegates in an isolated session, saves ancestry and returns only the child final text', async () => {
  const storage = new JsonlTreeSessionStore(dir);
  core = runtime(dir, (model, context) => {
    const messages = JSON.stringify(context.messages);
    if (!messages.includes('Parent private history')) {
      expect(messages).toContain('Child task with explicit context');
      expect(getCurrentSystemPrompt(context.messages)).toContain('Investigate the issue carefully');
      if (context.messages.some(({ role }) => role === 'toolResult')) return reply(model, text('Child conclusion'));
      return reply(model, text('Intermediate reasoning'), call('read'));
    }
    if (context.messages.at(-1)?.role !== 'toolResult') return reply(model, delegate('explore'));
    expect(context.messages.at(-1)).toMatchObject({ isError: false, content: [text('Child conclusion')] });
    expect(messages).not.toContain('Intermediate reasoning');
    expect(messages).not.toContain('Raw tool data');
    return reply(model, text('Parent conclusion'));
  }, {
    sessionStorage: storage,
    tools: [tool('read', { readOnly: true, execute: async () => textResult('Raw tool data') })],
    subagents: { explore: { name: 'explore', description: 'Inspect', systemPrompt: 'Investigate the issue carefully' } },
  });
  const parent = await core.createSession();
  const events: AgentCoreEvent[] = [];
  parent.subscribe((event) => events.push(event));
  await expect(parent.prompt('Parent private history')).resolves.toMatchObject({ state: 'completed', text: 'Parent conclusion' });
  const child = (await core.listSessions()).find(({ id }) => id !== parent.id);
  expect(child).toMatchObject({ metadata: { parentSessionId: parent.id, subagentProfile: 'explore', depth: 1 } });
  expect(events).toEqual(expect.arrayContaining([
    expect.objectContaining({ type: 'subagent_started', subagentId: child?.id, background: false }),
    expect.objectContaining({ type: 'subagent_completed', subagentId: child?.id, summary: 'Child conclusion' }),
  ]));
  expect(JSON.stringify((await storage.getBranch(child!.id)))).not.toContain('Parent private history');
});

it('returns immediately in background mode and automatically follows up after the parent is idle', async () => {
  const releaseChild = Promise.withResolvers<void>();
  const parentUpdated = Promise.withResolvers<string>();
  core = runtime(dir, async (model, context) => {
    const messages = JSON.stringify(context.messages);
    if (!messages.includes('Parent request')) {
      await releaseChild.promise;
      return reply(model, text('Background conclusion'));
    }
    const result = context.messages.find(({ role }) => role === 'toolResult');
    if (!result) return reply(model, delegate('explore', { background: true }));
    expect(result.content).toEqual([text(expect.stringContaining('"status":"running"'))]);
    const resultMessage = context.messages.flatMap((message) => message.role !== 'user' ? [] :
      typeof message.content === 'string' ? [message.content] :
        message.content.flatMap((block) => block.type === 'text' ? [block.text] : []))
      .find((content) => content.startsWith('<background-result'));
    if (resultMessage) parentUpdated.resolve(resultMessage);
    return reply(model, text(resultMessage ? 'Parent received result' : 'Parent continued'));
  });
  const parent = await core.createSession();
  const completed = Promise.withResolvers<Extract<AgentCoreEvent, { type: 'subagent_completed' }>>();
  parent.subscribe((event) => { if (event.type === 'subagent_completed') completed.resolve(event); });
  await expect(parent.prompt('Parent request')).resolves.toMatchObject({ text: 'Parent continued' });
  releaseChild.resolve();
  const { subagentId } = await completed.promise;
  expect(await parentUpdated.promise).toBe(
    `<background-result subagentId="${subagentId}" agent="explore" state="completed">Background conclusion</background-result>`,
  );
});

it('forces a tool-free final response at maxSteps and honors profile tool filters', async () => {
  const runTool = vi.fn(async () => textResult('used'));
  let childSteps = 0;
  core = runtime(dir, (model, context, options) => {
    const messages = JSON.stringify(context.messages);
    if (messages.includes('Parent budget task')) {
      const result = context.messages.at(-1);
      if (result?.role !== 'toolResult') return reply(model, delegate('limited'));
      expect(result.content).toEqual([text('Final summary')]);
      return reply(model, text('Done'));
    }
    const names = getCurrentTools(context.messages).map(({ name }) => name);
    if (++childSteps === 1) {
      expect(names).toEqual(['lookup']);
      return reply(model, call('lookup'));
    }
    expect(names).toEqual([]);
    expect(options?.toolChoice).toBe('none');
    expect(messages).toContain('final summary');
    return reply(model, text('Final summary'));
  }, {
    subagents: { limited: {
      name: 'limited', description: 'Limited', maxSteps: 2,
      allowedTools: ['lookup', 'sensitive'], deniedTools: ['sensitive'],
    } },
    tools: [tool('lookup', { execute: runTool }), tool('sensitive')],
  });
  const parent = await core.createSession();
  await expect(parent.prompt('Parent budget task')).resolves.toMatchObject({ text: 'Done' });
  expect(childSteps).toBe(2);
  expect(runTool).toHaveBeenCalledTimes(1);
});

it('hides delegation at the configured depth while allowing custom profiles to delegate below it', async () => {
  const observed: { level: number; hasDelegate: boolean }[] = [];
  core = runtime(dir, (model, context) => {
    const messages = JSON.stringify(context.messages);
    const level = messages.includes('Depth zero') ? 0 : messages.includes('Depth one') ? 1 : 2;
    if (context.messages.at(-1)?.role === 'toolResult') return reply(model, text(`Depth ${level} done`));
    const hasDelegate = getCurrentTools(context.messages).some(({ name }) => name === 'subagent');
    observed.push({ level, hasDelegate });
    if (level < 2) return reply(model, delegate('nested', { prompt: level === 0 ? 'Depth one' : 'Depth two' }));
    return reply(model, text('Leaf done'));
  }, {
    maxSubagentDepth: 2, subagents: { nested: { name: 'nested', description: 'Custom nesting' } },
  });
  const parent = await core.createSession();
  await expect(parent.prompt('Depth zero')).resolves.toMatchObject({ text: 'Depth 0 done' });
  expect(observed).toEqual([
    { level: 0, hasDelegate: true }, { level: 1, hasDelegate: true }, { level: 2, hasDelegate: false },
  ]);
  expect((await core.listSessions()).map(({ metadata }) => metadata?.depth).filter(Boolean).sort()).toEqual([1, 2]);
});

it('allows explore in Plan mode and refuses the write-capable general profile', async () => {
  core = runtime(dir, (model, context) => {
    const messages = JSON.stringify(context.messages);
    const tools = getCurrentTools(context.messages).map(({ name }) => name);
    if (!messages.includes('Parent plan')) {
      expect(tools).toContain('read');
      expect(tools).not.toContain('write');
      expect(tools).not.toContain('subagent');
      return reply(model, text('Read-only exploration'));
    }
    expect(tools).toContain('subagent');
    const results = context.messages.filter(({ role }) => role === 'toolResult');
    if (!results.length) return reply(model, delegate('explore'), delegate('general'));
    expect(results).toMatchObject([
      { isError: false, content: [text('Read-only exploration')] },
      { isError: true, content: [text(expect.stringContaining('Plan mode'))] },
    ]);
    return reply(model, text('Planned'));
  }, { tools: [tool('read', { readOnly: true }), tool('write')] });
  const session = await core.createSession({ mode: 'plan' });
  await expect(session.prompt('Parent plan')).resolves.toMatchObject({ state: 'completed', text: 'Planned' });
  expect(await core.listSessions()).toHaveLength(2);
});

it.each(['abort', 'close'] as const)('cascades parent %s to a background child after the parent turn has finished', async (action) => {
  const childStarted = Promise.withResolvers<void>();
  const childAborted = Promise.withResolvers<void>();
  core = runtime(dir, async (model, context, options) => {
    if (JSON.stringify(context.messages).includes('Parent cancellation request')) {
      return context.messages.at(-1)?.role === 'toolResult'
        ? reply(model, text('Parent idle')) : reply(model, delegate('general', { background: true }));
    }
    const signal = options!.signal!;
    const stopped = Promise.withResolvers<never>();
    signal.addEventListener('abort', () => {
      childAborted.resolve();
      stopped.reject(signal.reason);
    }, { once: true });
    childStarted.resolve();
    return stopped.promise;
  });
  const parent = await core.createSession();
  const completed = Promise.withResolvers<AgentCoreEvent>();
  parent.subscribe((event) => { if (event.type === 'subagent_completed') completed.resolve(event); });
  await expect(parent.prompt('Parent cancellation request')).resolves.toMatchObject({ text: 'Parent idle' });
  await childStarted.promise;
  await parent[action]();
  await childAborted.promise;
  expect(await completed.promise).toMatchObject({ type: 'subagent_completed', state: 'cancelled' });
  if (action === 'close') expect(await core.openSession(parent.id)).not.toBe(parent);
});

it('resumes the same child transcript when delegated with its sessionId', async () => {
  core = runtime(dir, (model, context) => {
    const messages = JSON.stringify(context.messages);
    if (!messages.includes('Parent delegation')) {
      if (messages.includes('Follow-up question')) {
        expect(messages).toContain('Child task with explicit context');
        expect(messages).toContain('First child summary');
        return reply(model, text('Resumed child summary'));
      }
      return reply(model, text('First child summary'));
    }
    const result = context.messages.at(-1);
    if (result?.role === 'toolResult') {
      expect(result.isError).toBe(false);
      return reply(model, text(result.content[0].type === 'text' ? result.content[0].text : ''));
    }
    const request = context.messages.findLast((message) => message.role === 'user');
    const id = JSON.stringify(request?.content).match(/Resume ([a-f0-9-]+)/)?.[1];
    return reply(model, delegate('general', id ? { sessionId: id, prompt: 'Follow-up question' } : {}));
  });
  const parent = await core.createSession();
  await expect(parent.prompt('Parent delegation')).resolves.toMatchObject({ text: 'First child summary' });
  const child = (await core.listSessions()).find(({ id }) => id !== parent.id)!;
  await expect(parent.prompt(`Resume ${child.id}`)).resolves.toMatchObject({ text: 'Resumed child summary' });
  expect(await core.listSessions()).toHaveLength(2);
});
