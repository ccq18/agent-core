import { rm } from 'node:fs/promises';
import { getCurrentTools, type TranscriptContext } from '@earendil-works/pi-ai';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { DEFAULT_PROMPT_TEMPLATES, JsonlTreeSessionStore, ModeController, wrapSystemUpdate,
  SessionBatchAppendError, type AgentCoreEvent, type AgentRuntime, type PlanItem } from '../src/index.js';
import { call, legacyStorage, reply, runtime, testDirectory, textResult, tool } from './harness-fixtures.js';

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

it('distinguishes a Plan reminder from identical text submitted as the user task', async () => {
  const task = wrapSystemUpdate(DEFAULT_PROMPT_TEMPLATES.planModeEnter);
  const contexts: TranscriptContext[] = [];
  const core = runtime(dir, (model, context) => {
    contexts.push(context);
    return reply(model, { type: 'text', text: 'ok' });
  });
  cores.push(core);
  const session = await core.createSession({ mode: 'plan' });
  await session.prompt(task);
  const messages = contexts[0].messages.filter(({ role }) => role === 'user');
  expect(messages).toHaveLength(2);
  expect(messages[0]).toMatchObject({ content: task, systemUpdate: true });
  expect(messages[1]).toMatchObject({ content: [{ type: 'text', text: task }] });
  expect(messages[1]).not.toHaveProperty('systemUpdate');
});

it('uses Plan permission rules for each tool even when read and write share an action', async () => {
  const items = [{ id: 'one', content: 'Inspect input', status: 'in_progress', priority: 'high' }] satisfies PlanItem[];
  const write = vi.fn(async () => textResult('modified'));
  let step = 0;
  const core = runtime(dir, (model, context) => {
    expect(getCurrentTools(context.messages).map(({ name }) => name))
      .toEqual(['todowrite', 'todoread', 'load_skill', 'subagent', 'question', 'read']);
    switch (step++) {
      case 0:
        expect(context.messages.at(-2)).toMatchObject({
          role: 'user', content: expect.stringMatching(/<system-update>\s*Plan mode is active/),
        });
        return reply(model, call('write'), call('read'), call('todowrite', { items }));
      case 1: {
        const results = context.messages.filter(({ role }) => role === 'toolResult');
        expect(results).toMatchObject([
          { toolName: 'write', isError: true }, { toolName: 'read', isError: false },
          { toolName: 'todowrite', isError: false, content: [{ text: expect.stringContaining('Inspect input') }] },
        ]);
        return reply(model, call('todoread'));
      }
      default:
        expect(context.messages.findLast(({ role }) => role === 'toolResult')).toMatchObject({
          role: 'toolResult', isError: false, content: [{ text: expect.stringContaining('Inspect input') }],
        });
        return reply(model, { type: 'text', text: 'planned' });
    }
  }, {
    permissions: [{ action: '*', resource: '*', effect: 'deny' }],
    tools: [tool('write', { permissionAction: 'files', execute: write }),
      tool('read', { readOnly: true, permissionAction: 'files' })],
  });
  cores.push(core);
  const session = await core.createSession({ mode: 'plan' });
  const events: AgentCoreEvent[] = [];
  session.subscribe((event) => events.push(event));
  const turn = await session.prompt('Make a plan');
  expect(turn).toMatchObject({ state: 'completed', text: 'planned' });
  expect(write).not.toHaveBeenCalled();
  expect(session.getPlan()).toEqual(items);
  expect(events).toContainEqual(expect.objectContaining({ type: 'plan_update', items, runId: turn.runId }));
  await core.dispose();

  const fresh = runtime(dir, (model) => reply(model, { type: 'text', text: 'ok' }));
  cores.push(fresh);
  expect((await fresh.openSession(session.id)).getPlan()).toEqual(items);
});

it.each([{ depth: 1 }, { subagentProfile: 'explore' }])('opens a persisted helper marked %o without mode reminders', async (metadata) => {
  const { id } = await new JsonlTreeSessionStore(dir).create({ id: 'host-helper', mode: 'plan', metadata });
  const storage = new JsonlTreeSessionStore(dir);
  const write = vi.fn(async () => textResult('modified'));
  const contexts: TranscriptContext[] = [];
  const core = runtime(dir, (model, context) => {
    contexts.push(context);
    return contexts.length === 1 ? reply(model, call('read'), call('write')) : reply(model, { type: 'text', text: 'ok' });
  }, { sessionStorage: storage, tools: [tool('read', { readOnly: true }), tool('write', { execute: write })] });
  cores.push(core);

  const helper = await core.openSession(id);
  expect(helper.mode).toBe('plan');
  await expect(helper.prompt('Inspect')).resolves.toMatchObject({ state: 'completed' });
  const tools = getCurrentTools(contexts[0].messages).map(({ name }) => name);
  expect(tools).toContain('read');
  expect(tools).not.toContain('write');
  expect(contexts[1].messages.filter(({ role }) => role === 'toolResult')).toMatchObject([
    { toolName: 'read', isError: false },
    { toolName: 'write', isError: true },
  ]);
  expect(write).not.toHaveBeenCalled();

  await helper.close();
  const reopened = await core.openSession(id);
  await expect(reopened.prompt('Resume inspection')).resolves.toMatchObject({ state: 'completed' });
  const planLeaf = (await storage.getBranch(id)).at(-1)!;
  await reopened.setMode('build');
  await expect(reopened.prompt('Work')).resolves.toMatchObject({ state: 'completed' });
  await reopened.branch(planLeaf.id);
  expect(reopened.mode).toBe('plan');
  await expect(reopened.prompt('Resume plan branch')).resolves.toMatchObject({ state: 'completed' });

  for (const { messages } of contexts) {
    expect(JSON.stringify(messages.filter(({ role }) => role === 'user'))).not.toMatch(/(?:Plan|Build) mode is active/);
  }
});

it('inserts mode reminders before the latest user message while keeping the leading system message equal across switches and branches', async () => {
  const contexts: TranscriptContext[] = [];
  const storage = new JsonlTreeSessionStore(dir);
  const core = runtime(dir, (model, context) => {
    contexts.push(context);
    return reply(model, { type: 'text', text: 'ok' });
  }, { baseSystemPrompt: 'Stable prefix', sessionStorage: storage });
  cores.push(core);
  const session = await core.createSession();
  await session.prompt('Initial Build');
  const leaf = (await storage.getBranch(session.id)).at(-1)!;
  await session.setMode('plan');
  await session.prompt('Plan');
  const planLeaf = (await storage.getBranch(session.id)).at(-1)!;
  await session.setMode('build');
  await session.prompt('Execute');
  const changedBuild = (await storage.getBranch(session.id)).at(-1)!;
  await session.branch(planLeaf.id);
  await session.prompt('Plan branch');
  await session.branch(leaf.id);
  await session.prompt('Build branch');

  const updates = contexts.slice(1).map(({ messages }) => messages.at(-2));
  expect(updates).toMatchObject([
    { role: 'user', content: expect.stringMatching(/<system-update>.*plan/is) },
    { role: 'user', content: expect.stringMatching(/<system-update>.*build/is) },
    { role: 'user', content: expect.stringMatching(/<system-update>.*plan/is) },
    { role: 'assistant', content: [{ type: 'text', text: 'ok' }] },
  ]);
  expect(contexts.every(({ messages }) => JSON.stringify(messages[0]) === JSON.stringify(contexts[0].messages[0]))).toBe(true);
  expect(new ModeController('plan').transformContext([])).toHaveLength(1);
  const controller = new ModeController('plan');
  const corrected = controller.transformContext([]);
  expect(controller.transformContext(corrected)).toEqual(corrected);

  // Open on the initial Build branch, then switch directly to the marked Build branch.
  await session.close();
  const fromInitial = await core.openSession(session.id);
  await fromInitial.branch(changedBuild.id);
  await fromInitial.prompt('Resume');
  const afterSwitch = contexts.at(-1)!.messages.at(-2);
  await fromInitial.branch(changedBuild.id);
  await fromInitial.close();
  const reopened = await core.openSession(session.id);
  await reopened.prompt('Resume');
  const afterOpen = contexts.at(-1)!.messages.at(-2);
  expect(afterOpen).toMatchObject({ role: 'user', content: expect.stringContaining('Build mode is active') });
  expect(afterSwitch).toMatchObject({ role: afterOpen!.role, content: afterOpen!.content });
});

it('blocks a write from a stale Build snapshot and refreshes the next step tool list', async () => {
  const atModel = Promise.withResolvers<void>();
  const resumeModel = Promise.withResolvers<void>();
  const write = vi.fn(async () => textResult('modified'));
  let step = 0;
  const core = runtime(dir, async (model, context) => {
    const names = getCurrentTools(context.messages).map(({ name }) => name);
    if (step++ === 0) {
      expect(names).toContain('write');
      atModel.resolve();
      await resumeModel.promise;
      return reply(model, call('write'));
    }
    expect(names).not.toContain('write');
    expect(context.messages.findLast(({ role }) => role === 'toolResult')).toMatchObject({
      role: 'toolResult', isError: true, content: [{ text: expect.stringContaining('Plan mode') }],
    });
    return reply(model, { type: 'text', text: 'blocked' });
  }, { tools: [tool('write', { execute: write })] });
  cores.push(core);
  const session = await core.createSession();
  const pending = session.prompt('write');
  await atModel.promise;
  try {
    await session.setMode('plan');
  } finally {
    resumeModel.resolve();
  }
  await expect(pending).resolves.toMatchObject({ state: 'completed', text: 'blocked' });
  expect(write).not.toHaveBeenCalled();
});

it('lets host tools override built-ins and restores them when unregistered', async () => {
  const seen: string[] = [];
  const core = runtime(dir, (model, context) => {
    seen.push(getCurrentTools(context.messages).find(({ name }) => name === 'todoread')!.description);
    return reply(model, { type: 'text', text: 'ok' });
  });
  cores.push(core);
  const session = await core.createSession();
  const dispose = core.registerTools([tool('todoread', { description: 'host todo' })]);
  await session.prompt('Overridden');
  dispose();
  await session.prompt('Restored');
  expect(seen[0]).toBe('host todo');
  expect(seen[1]).not.toBe('host todo');
  expect(core.getCatalog().tools.filter(({ name }) => name === 'todoread')).toHaveLength(1);
});

it('rebuilds the Build reminder after reopening a compacted transcript', async () => {
  const storage = new JsonlTreeSessionStore(dir);
  const contexts: TranscriptContext[] = [];
  const stream: Parameters<typeof runtime>[1] = (model, context) => {
    contexts.push(context);
    return reply(model, { type: 'text', text: 'ok' });
  };
  const core = runtime(dir, stream, { sessionStorage: storage, baseSystemPrompt: 'Stable prefix' });
  cores.push(core);
  const session = await core.createSession({ mode: 'plan' });
  await session.prompt('Plan');
  await session.setMode('build');
  await core.dispose();
  await storage.append(session.id, {
    type: 'compaction', summary: 'The plan is ready.', firstKeptEntryId: null, tokensBefore: 300,
  });

  const fresh = runtime(dir, stream);
  cores.push(fresh);
  const restored = await fresh.openSession(session.id);
  await expect(restored.prompt('Execute')).resolves.toMatchObject({ state: 'completed' });
  expect(contexts[1].messages[0]).toEqual(contexts[0].messages[0]);
  expect(contexts[1].messages).toEqual(expect.arrayContaining([
    expect.objectContaining({ role: 'user', content: expect.stringContaining('<compaction-summary>') }),
  ]));
  expect(contexts[1].messages.at(-2)).toMatchObject({
    role: 'user', content: expect.stringMatching(/<system-update>.*build/is),
  });
});

it.each(['batch', 'legacy'] as const)('keeps todowrite, mode and checkpoint journal order with %s storage', async (kind) => {
  const storage = new JsonlTreeSessionStore(dir);
  const batch = vi.spyOn(storage, 'appendBatch');
  const single = vi.spyOn(storage, 'append');
  const items = [{ id: 'one', content: 'Inspect input', status: 'in_progress' }] satisfies PlanItem[];
  let calls = 0;
  const core = runtime(dir, (model, context) => {
    if (context.messages.at(-1)?.role === 'toolResult' || calls++ > 0) return reply(model, { type: 'text', text: 'Complete' });
    return reply(model, call('todowrite', { items }));
  }, { sessionStorage: kind === 'batch' ? storage : legacyStorage(storage), defaultPermissionEffect: 'allow',
    compaction: { keepRecentTokens: 4, reserveTokens: 0 } });
  cores.push(core);
  const session = await core.createSession();
  await expect(session.prompt('Record the initial plan')).resolves.toMatchObject({ state: 'completed' });
  const written = await new JsonlTreeSessionStore(dir).getBranch(session.id);
  const planIndex = written.findIndex(({ type }) => type === 'plan');
  expect(written[planIndex - 1]).toMatchObject({ type: 'message', message: {
    role: 'assistant', content: [expect.objectContaining({ type: 'toolCall', name: 'todowrite' })],
  } });
  expect(written[planIndex + 1]).toMatchObject({ type: 'message', message: { role: 'toolResult' } });
  expect(written.at(-1)).toMatchObject({ type: 'message', message: { role: 'assistant' } });
  if (kind === 'batch') expect(batch).toHaveBeenCalled();
  else {
    expect(batch).not.toHaveBeenCalled();
    expect(single.mock.calls.filter(([, entry]) => entry.type === 'message')).toHaveLength(planIndex + 2);
  }
  await session.setMode('plan');
  await session.prompt('A recent request to retain');
  await expect(session.compact()).resolves.toMatchObject({ status: 'compacted' });
  const branch = await storage.getBranch(session.id);
  expect(branch.findIndex(({ type }) => type === 'mode')).toBe(written.length);
  expect(branch.at(-1)).toMatchObject({ type: 'compaction', parentId: branch.at(-2)?.id });
  await session.branch(written.at(-1)!.id);
  expect(session.mode).toBe('build');
});

it('retries only the unwritten suffix when a turn batch reports a committed prefix', async () => {
  const storage = new JsonlTreeSessionStore(dir);
  const appendBatch = storage.appendBatch.bind(storage);
  const committedIds: string[] = [];
  vi.spyOn(storage, 'appendBatch').mockImplementationOnce(async (id, entries) => {
    expect(entries).toHaveLength(3);
    committedIds.push(...(await appendBatch(id, entries.slice(0, 2))).map(({ id }) => id));
    throw new SessionBatchAppendError(2, true, new Error('interrupted write'));
  });
  const core = runtime(dir, (model) => reply(model, { type: 'text', text: 'Done' }), {
    sessionStorage: storage, compaction: { enabled: false },
  });
  cores.push(core);
  const session = await core.createSession();
  await expect(session.prompt('First input')).rejects.toMatchObject({ committedCount: 2 });
  await expect(session.prompt('Second input')).resolves.toMatchObject({ state: 'completed' });
  const branch = (await new JsonlTreeSessionStore(dir).getBranch(session.id))
    .filter((entry) => entry.type === 'message');
  expect(branch.slice(1, 3).map(({ id }) => id)).toEqual(committedIds);
  const dialogue = branch.filter(({ message }) => message.role !== 'system');
  expect(dialogue.map(({ message }) => message.role)).toEqual(['user', 'assistant', 'user', 'assistant']);
  expect(dialogue[0]).toMatchObject({ message: { content: [{ type: 'text', text: 'First input' }] } });
});
