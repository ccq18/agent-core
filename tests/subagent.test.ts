import { renameSync } from 'node:fs';
import { readFile, rm, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { getCurrentSystemPrompt, getCurrentTools } from '@earendil-works/pi-ai';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { JsonlTreeSessionStore, SubagentManager, type AgentCoreEvent, type AgentRuntime } from '../src/index.js';
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
  }
});

const text = (text: string) => ({ type: 'text' as const, text });
const delegate = (agent: string, overrides = {}) => call('subagent', {
  agent, prompt: 'Child task with explicit context', description: 'Inspect', ...overrides,
}, `delegate-${agent}`);

it('delegates in an isolated session, saves ancestry and returns only the child final text', async () => {
  const storage = new JsonlTreeSessionStore(dir);
  const write = vi.fn(async () => textResult('Modified'));
  const childMessages: string[] = [];
  core = runtime(dir, (model, context) => {
    const messages = JSON.stringify(context.messages);
    if (!messages.includes('Parent private history')) {
      childMessages.push(messages);
      expect(messages).toContain('Child task with explicit context');
      expect(getCurrentSystemPrompt(context.messages)).toContain('Investigate the issue carefully');
      expect(getCurrentTools(context.messages).map(({ name }) => name)).not.toContain('write');
      const results = context.messages.filter(({ role }) => role === 'toolResult');
      if (results.length) {
        expect(results).toMatchObject([
          { toolName: 'read', isError: false, content: [text('Raw tool data')] },
          { toolName: 'write', isError: true },
        ]);
        return reply(model, text('Child conclusion'));
      }
      return reply(model, text('Intermediate reasoning'), call('read'), call('write'));
    }
    if (context.messages.at(-1)?.role !== 'toolResult') return reply(model, delegate('explore'));
    expect(context.messages.at(-1)).toMatchObject({ isError: false, content: [text('Child conclusion')] });
    expect(messages).not.toContain('Intermediate reasoning');
    expect(messages).not.toContain('Raw tool data');
    return reply(model, text('Parent conclusion'));
  }, {
    sessionStorage: storage,
    tools: [
      tool('read', { readOnly: true, execute: async () => textResult('Raw tool data') }),
      tool('write', { execute: write }),
    ],
    subagents: { explore: { name: 'explore', description: 'Inspect', systemPrompt: 'Investigate the issue carefully' } },
  });
  const parent = await core.createSession();
  const events: AgentCoreEvent[] = [];
  parent.subscribe((event) => events.push(event));
  await expect(parent.prompt('Parent private history')).resolves.toMatchObject({ state: 'completed', text: 'Parent conclusion' });
  expect(write).not.toHaveBeenCalled();
  expect(childMessages.join('\n')).not.toContain('Plan mode is active');
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

it.each(['finish', 'abort', 'dispose'] as const)(
  'keeps a depth-two background child alive through normal cleanup and handles %s',
  async (action) => {
    const storage = new JsonlTreeSessionStore(dir, { maxIdleTrees: 0 });
    const oneSaved = Promise.withResolvers<void>();
    const leafStarted = Promise.withResolvers<void>();
    const releaseLeaf = Promise.withResolvers<void>();
    const leafAborted = Promise.withResolvers<void>();
    const delivering = Promise.withResolvers<void>();
    const releaseDelivery = Promise.withResolvers<void>();
    const completed = Promise.withResolvers<Extract<AgentCoreEvent, { type: 'subagent_completed' }>>();
    const append = storage.append.bind(storage);
    vi.spyOn(storage, 'append').mockImplementation(async (id, entry) => {
      const saved = await append(id, entry);
      if (entry.type === 'message' && entry.message.role === 'assistant' &&
          entry.message.content.some((block) => block.type === 'text' && block.text === 'Depth one initial')) {
        oneSaved.resolve();
      }
      return saved;
    });
    core = runtime(dir, async (model, context, options) => {
      const contents = JSON.stringify(context.messages);
      const last = context.messages.at(-1);
      const delivered = context.messages.some((message) =>
        message.role === 'user' && JSON.stringify(message.content).includes('<background-result'));
      if (contents.includes('Root request')) {
        if (last?.role === 'toolResult') return reply(model, text(last.isError ? JSON.stringify(last.content) : 'Root idle'));
        if (delivered) return reply(model, text('Root idle'));
        return reply(model, delegate('nested', { background: true, prompt: 'Depth one request' }));
      }
      if (contents.includes('Depth one request')) {
        if (delivered) {
          expect(contents).toContain('Depth two done');
          delivering.resolve();
          await releaseDelivery.promise;
          return reply(model, text('Depth one received nested result'));
        }
        if (last?.role === 'toolResult') {
          if (last.isError) return reply(model, text(JSON.stringify(last.content)));
          await leafStarted.promise;
          return reply(model, text('Depth one initial'));
        }
        return reply(model, delegate('nested', { background: true, prompt: 'Depth two request' }));
      }
      const signal = options!.signal!;
      const cancellation = Promise.withResolvers<never>();
      signal.addEventListener('abort', () => {
        leafAborted.resolve();
        cancellation.reject(signal.reason);
      }, { once: true });
      leafStarted.resolve();
      await Promise.race([releaseLeaf.promise, cancellation.promise]);
      return reply(model, text('Depth two done'));
    }, {
      sessionStorage: storage, maxSubagentDepth: 2, subagents: { nested: { description: 'Custom nesting' } },
    });
    const root = await core.createSession({ id: 'root' });
    root.subscribe((event) => { if (event.type === 'subagent_completed') completed.resolve(event); });

    try {
      await expect(root.prompt('Root request')).resolves.toMatchObject({ text: 'Root idle' });
      await expect(Promise.race([
        Promise.all([leafStarted.promise, oneSaved.promise]).then(() => 'nested turn ready'),
        completed.promise.then(({ state, summary }) => `released early: ${state}: ${summary}`),
      ])).resolves.toBe('nested turn ready');
      const sessions = await core.listSessions();
      const child = sessions.find(({ metadata }) => metadata?.depth === 1)!;
      const leaf = sessions.find(({ metadata }) => metadata?.depth === 2)!;

      if (action === 'finish') {
        releaseLeaf.resolve();
        await expect(Promise.race([
          delivering.promise.then(() => 'delivering'),
          completed.promise.then(({ state }) => `released early: ${state}`),
        ])).resolves.toBe('delivering');
        expect(storage['references'].has(child.id)).toBe(true);
        releaseDelivery.resolve();
        expect(await completed.promise).toMatchObject({ state: 'completed', summary: 'Depth one initial' });
        const transcript = JSON.stringify(await storage.getBranch(child.id));
        expect(transcript).toContain('Depth two done');
        expect(transcript).toContain('Depth one received nested result');
      } else {
        if (action === 'dispose') await core.dispose();
        else root.abort();
        await leafAborted.promise;
        expect(await completed.promise).toMatchObject({ state: 'cancelled' });
        const inputs = (await storage.getBranch(child.id)).filter((entry) =>
          entry.type === 'message' && entry.message.role === 'user');
        expect(JSON.stringify(inputs)).not.toContain('<background-result');
      }
      expect(storage['references'].has(child.id)).toBe(false);
      expect(storage['references'].has(leaf.id)).toBe(false);
      expect([...Reflect.get(core, 'sessions').keys()]).toEqual(action === 'dispose' ? [] : [root.id]);
      await root.close();
      expect(storage['references'].size).toBe(0);
      expect(storage['trees'].size).toBe(0);
    } finally {
      releaseLeaf.resolve();
      releaseDelivery.resolve();
    }
  },
);

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
  const storage = new JsonlTreeSessionStore(dir, { maxIdleTrees: 0 });
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
  }, { sessionStorage: storage });
  const parent = await core.createSession();
  const completed = Promise.withResolvers<AgentCoreEvent>();
  parent.subscribe((event) => { if (event.type === 'subagent_completed') completed.resolve(event); });
  await expect(parent.prompt('Parent cancellation request')).resolves.toMatchObject({ text: 'Parent idle' });
  await childStarted.promise;
  await parent[action]();
  await childAborted.promise;
  const event = await completed.promise;
  expect(event).toMatchObject({ type: 'subagent_completed', state: 'cancelled' });
  if (event.type === 'subagent_completed') expect(storage['trees'].has(event.subagentId)).toBe(false);
  if (action === 'close') expect(await core.openSession(parent.id)).not.toBe(parent);
});

it('resumes the same child transcript when delegated with its sessionId', async () => {
  const storage = new JsonlTreeSessionStore(dir, { maxIdleTrees: 0 });
  const project = vi.spyOn(storage, 'projectActiveMessages');
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
  }, { sessionStorage: storage });
  const parent = await core.createSession();
  await expect(parent.prompt('Parent delegation')).resolves.toMatchObject({ text: 'First child summary' });
  const child = (await core.listSessions()).find(({ id }) => id !== parent.id)!;
  expect(storage['trees'].has(child.id)).toBe(false);
  await expect(parent.prompt(`Resume ${child.id}`)).resolves.toMatchObject({ text: 'Resumed child summary' });
  expect(storage['trees'].has(child.id)).toBe(false);
  expect(project.mock.calls.filter(([id]) => id === child.id)).toHaveLength(2);
  expect(await core.listSessions()).toHaveLength(2);
});

it('releases the managed handle when a child prompt throws', async () => {
  const storage = new JsonlTreeSessionStore(dir, { maxIdleTrees: 0 });
  const id = 'throwing-child';
  core = runtime(dir, (model, context) => context.messages.at(-1)?.role === 'toolResult'
    ? reply(model, text('Parent handled failure')) : reply(model, delegate('general', { sessionId: id, modelAlias: 'unknown' })), {
    sessionStorage: storage,
  });
  const parent = await core.createSession();
  const completed = Promise.withResolvers<Extract<AgentCoreEvent, { type: 'subagent_completed' }>>();
  parent.subscribe((event) => { if (event.type === 'subagent_completed') completed.resolve(event); });
  // An existing child's model override is resolved by prompt(), after acquisition.
  await storage.create({ id, metadata: { parentSessionId: parent.id, subagentProfile: 'general', depth: 1 } });
  await expect(parent.prompt('Parent failure request')).resolves.toMatchObject({ state: 'completed' });
  expect(await completed.promise).toMatchObject({ subagentId: id, state: 'failed', summary: expect.stringContaining('Unknown model alias') });
  expect(storage['trees'].has(id)).toBe(false);
});

it('validates resume ownership, preserves a host-owned child and waits for close before reopening once', async () => {
  const storage = new JsonlTreeSessionStore(dir, { maxIdleTrees: 0 });
  const blocked = Promise.withResolvers<void>();
  const responding = Promise.withResolvers<void>();
  core = runtime(dir, (model, context) => {
    if (context.messages.at(-1)?.role === 'toolResult') {
      return reply(model, text(JSON.stringify(context.messages.at(-1))));
    }
    const lastUser = context.messages.findLast(({ role }) => role === 'user');
    const resume = /Resume (owned|wrong-parent|wrong-profile|wrong-depth)/.exec(JSON.stringify(lastUser?.content) ?? '');
    if (resume) return reply(model, delegate('general', { sessionId: resume[1] }));
    return reply(model, text('Child response'));
  }, {
    sessionStorage: storage,
    hooks: {
      prepareRequest: async ({ context }) => {
        const lastUser = context.messages.findLast(({ role }) => role === 'user');
        if (!JSON.stringify(lastUser?.content)?.includes('Hold child close')) return;
        responding.resolve();
        await blocked.promise;
      },
    },
  });
  const parent = await core.createSession();
  const violations = [
    ['wrong-parent', { parentSessionId: 'different' }],
    ['wrong-profile', { subagentProfile: 'explore' }],
    ['wrong-depth', { depth: 2 }],
  ] as const;
  for (const [id, metadata] of [...violations, ['owned', {}] as const]) {
    await storage.create({ id, metadata: { parentSessionId: parent.id, subagentProfile: 'general', depth: 1, ...metadata } });
  }
  for (const [id] of violations) {
    await expect(parent.prompt(`Resume ${id}`)).resolves.toMatchObject({ text: expect.stringContaining('does not belong') });
    expect(storage['trees'].has(id)).toBe(false);
  }
  const hostChild = await core.openSession('owned');
  await expect(parent.prompt('Resume owned')).resolves.toMatchObject({ text: expect.stringContaining('Child response') });
  expect(await core.openSession('owned')).toBe(hostChild);
  expect(storage['trees'].has('owned')).toBe(true);

  const active = hostChild.prompt('Hold child close');
  const pending: Promise<unknown>[] = [active];
  try {
    await responding.promise;
    let closeFinished = false;
    const closing = hostChild.close().then(() => { closeFinished = true; });
    const opens = [core.openSession('owned'), core.openSession('owned')].map(async (opening) => {
      const handle = await opening;
      expect(closeFinished).toBe(true);
      return handle;
    });
    const starting = Promise.withResolvers<void>();
    const unsubscribe = parent.subscribe((event) => {
      if (event.type === 'subagent_started' && event.subagentId === 'owned') starting.resolve();
    });
    const resumed = parent.prompt('Resume owned');
    pending.push(closing, ...opens, resumed);
    await starting.promise;
    unsubscribe();
    expect(storage['trees'].has('owned')).toBe(true);
    blocked.resolve();
    await closing;
    const [first, second] = await Promise.all(opens);
    expect(first).toBe(second);
    expect(first).not.toBe(hostChild);
    await expect(resumed).resolves.toMatchObject({ text: expect.stringContaining('Child response') });
    expect(await core.openSession('owned')).toBe(first);
  } finally {
    blocked.resolve();
    await Promise.allSettled(pending);
  }
});

it.each([0, 32])('rejects a resume when the reopened handle belongs to a different parent (idle tree cap %i)', async (maxIdleTrees) => {
  const storage = new JsonlTreeSessionStore(dir, { maxIdleTrees });
  const oldRequest = Promise.withResolvers<void>();
  const finishOld = Promise.withResolvers<void>();
  const waitingForClose = Promise.withResolvers<void>();
  const childPrompt = vi.fn();
  const id = 'closing-child';
  core = runtime(dir, (model, context) => {
    const contents = JSON.stringify(context.messages);
    if (contents.includes('Resume while closing')) {
      const last = context.messages.at(-1);
      return last?.role === 'toolResult'
        ? reply(model, text(JSON.stringify(last))) : reply(model, delegate('general', { sessionId: id }));
    }
    childPrompt();
    return reply(model, text('Child response'));
  }, {
    sessionStorage: storage,
    hooks: {
      prepareRequest: async ({ context }) => {
        if (!JSON.stringify(context.messages).includes('Hold old handle')) return;
        oldRequest.resolve();
        await finishOld.promise;
      },
    },
  });
  const parent = await core.createSession({ id: 'parent' });
  const ancestry = { parentSessionId: parent.id, subagentProfile: 'general', depth: 1 };
  const oldHeader = await storage.create({ id, metadata: ancestry });
  const oldChild = await core.openSession(id);
  const closeCompletion = Object.getOwnPropertyDescriptor(Object.getPrototypeOf(oldChild), 'closeCompletion')!.get!;
  vi.spyOn(oldChild as unknown as { closeCompletion: Promise<void> }, 'closeCompletion', 'get')
    .mockImplementation(() => {
      waitingForClose.resolve();
      return closeCompletion.call(oldChild);
    });
  const path = join(dir, 'sessions', id + '.jsonl');
  const replacement = JSON.stringify({ ...oldHeader, metadata: { ...ancestry, parentSessionId: 'other-parent' } }) + '\n';
  await writeFile(path + '.replacement', replacement);
  const release = storage.releaseSession.bind(storage);
  let replaced = false;
  vi.spyOn(storage, 'releaseSession').mockImplementation((sessionId) => {
    release(sessionId);
    if (sessionId === id && !replaced) {
      renameSync(path + '.replacement', path);
      replaced = true;
    }
  });
  const running = oldChild.prompt('Hold old handle');
  const pending: Promise<unknown>[] = [running];

  try {
    await oldRequest.promise;
    pending.push(oldChild.close());
    const resumed = parent.prompt('Resume while closing');
    pending.push(resumed);
    await waitingForClose.promise;
    finishOld.resolve();
    await expect(resumed).resolves.toMatchObject({ text: expect.stringContaining('does not belong') });
    expect(childPrompt).not.toHaveBeenCalled();
    expect(await readFile(path, 'utf8')).toBe(replacement);
    expect(storage['references'].has(id)).toBe(false);
    expect(Reflect.get(core, 'sessions').has(id)).toBe(false);
  } finally {
    finishOld.resolve();
    await Promise.allSettled(pending);
  }
});

it('rejects opening the same closing session from its hook while outside opens wait for close', async () => {
  const enteredHook = Promise.withResolvers<void>();
  const reopenInHook = Promise.withResolvers<void>();
  const deadlock = Promise.withResolvers<never>();
  const hookError = Promise.withResolvers<string>();
  core = runtime(dir, (model) => reply(model, text('Reply')), {
    hooks: {
      prepareRequest: async () => {
        enteredHook.resolve();
        await reopenInHook.promise;
        try {
          await Promise.race([core!.openSession('reentrant'), deadlock.promise]);
          hookError.resolve('Open unexpectedly succeeded');
        } catch (error) {
          hookError.resolve(error instanceof Error ? error.message : String(error));
        }
      },
    },
  });
  const session = await core.createSession({ id: 'reentrant' });
  const turn = session.prompt('Enter the hook');
  const pending: Promise<unknown>[] = [turn];
  let watchdog: ReturnType<typeof setTimeout> | undefined;
  try {
    await enteredHook.promise;
    let closeFinished = false;
    const closing = session.close().then(() => { closeFinished = true; });
    const outside = core.openSession(session.id).then((handle) => {
      expect(closeFinished).toBe(true);
      return handle;
    });
    pending.push(closing, outside);
    // Only used to let a broken implementation exit the hook and clean up.
    watchdog = setTimeout(() => deadlock.reject(new Error('Hook open deadlocked')), 1000);
    reopenInHook.resolve();
    expect(await hookError.promise).toMatch(/reentrant.*closing.*(?:own|active) turn/i);
    expect(await outside).not.toBe(session);
    expect(await turn).toMatchObject({ state: 'cancelled' });
  } finally {
    clearTimeout(watchdog);
    reopenInHook.resolve();
    await Promise.allSettled(pending);
  }
});

it('rejects a resume when a custom opener does not validate the handle ancestry', async () => {
  const storage = new JsonlTreeSessionStore(dir);
  core = runtime(dir, (model) => reply(model, text('Unexpected child response')), { sessionStorage: storage });
  const child = await core.createSession({
    id: 'custom-child', metadata: { parentSessionId: 'custom-parent', subagentProfile: 'general', depth: 1 },
  });
  const prompt = vi.spyOn(child, 'prompt');
  const manager = new SubagentManager(storage, {
    create: (options) => core!.createSession(options),
    open: (id) => core!.openSession(id),
    getParent: async (id) => ({
      id, mode: 'build', depth: 0, readOnly: false, emit: () => {}, followUp: () => {},
    }),
  });
  const result = await manager.tool.execute({
    agent: 'general', description: 'Resume', prompt: 'Retry', sessionId: child.id,
  }, {
    sessionId: 'custom-parent', runId: 'run', toolCallId: 'call', signal: new AbortController().signal,
    onUpdate: () => {},
  });
  expect(result).toMatchObject({ isError: true, content: [text(expect.stringContaining('without validating subagent ancestry'))] });
  expect(prompt).not.toHaveBeenCalled();
});

it('releases a managed child cancelled while its handle is being opened', async () => {
  const storage = new JsonlTreeSessionStore(dir, { maxIdleTrees: 0 });
  const opening = Promise.withResolvers<void>();
  const unblock = Promise.withResolvers<void>();
  const open = storage.open.bind(storage);
  const pauseOpen = vi.spyOn(storage, 'open').mockImplementation(async (id) => {
    if (id !== 'parent') {
      opening.resolve();
      await unblock.promise;
    }
    return open(id);
  });
  const childPrompt = vi.fn();
  core = runtime(dir, (model, context) => {
    if (!JSON.stringify(context.messages).includes('Cancel child during open')) {
      childPrompt();
      return reply(model, text('Unexpected child response'));
    }
    return context.messages.at(-1)?.role === 'toolResult'
      ? reply(model, text('Parent idle')) : reply(model, delegate('general', { background: true }));
  }, { sessionStorage: storage });
  const parent = await core.createSession({ id: 'parent' });
  const completed = Promise.withResolvers<Extract<AgentCoreEvent, { type: 'subagent_completed' }>>();
  parent.subscribe((event) => { if (event.type === 'subagent_completed') completed.resolve(event); });
  try {
    await expect(parent.prompt('Cancel child during open')).resolves.toMatchObject({ text: 'Parent idle' });
    await opening.promise;
    parent.abort();
    unblock.resolve();
    const event = await completed.promise;
    expect(event.state).toBe('cancelled');
    expect(childPrompt).not.toHaveBeenCalled();
    expect(storage['trees'].has(event.subagentId)).toBe(false);
  } finally {
    unblock.resolve();
    pauseOpen.mockRestore();
  }
});
