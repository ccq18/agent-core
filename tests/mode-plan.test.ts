import { rm } from 'node:fs/promises';
import { getCurrentTools, type TranscriptContext } from '@earendil-works/pi-ai';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { JsonlTreeSessionStore, ModeController, type AgentCoreEvent, type AgentRuntime, type PlanItem } from '../src/index.js';
import { call, reply, runtime, testDirectory, textResult, tool } from './harness-fixtures.js';

let dir: string;
const cores: AgentRuntime[] = [];
beforeEach(async () => { dir = await testDirectory(); });
afterEach(async () => {
  await Promise.all(cores.splice(0).map((core) => core.dispose()));
  await rm(dir, { recursive: true, force: true });
});

it('uses Plan permission rules for each tool even when read and write share an action', async () => {
  const items = [{ id: 'one', content: 'Inspect input', status: 'in_progress', priority: 'high' }] satisfies PlanItem[];
  const write = vi.fn(async () => textResult('modified'));
  let step = 0;
  const core = runtime(dir, (model, context) => {
    expect(getCurrentTools(context.messages).map(({ name }) => name))
      .toEqual(['todowrite', 'todoread', 'load_skill', 'subagent', 'read']);
    switch (step++) {
      case 0:
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

it('appends mode reminders while keeping the leading system message equal across switches and branches', async () => {
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
  await session.branch(planLeaf.id);
  await session.prompt('Plan branch');
  await session.branch(leaf.id);
  await session.prompt('Build branch');

  const updates = contexts.slice(1).map(({ messages }) => messages.at(-1));
  expect(updates).toMatchObject([
    { role: 'user', content: expect.stringMatching(/<system-update>.*plan/is) },
    { role: 'user', content: expect.stringMatching(/<system-update>.*build/is) },
    { role: 'user', content: expect.stringMatching(/<system-update>.*plan/is) },
    { role: 'user', content: expect.stringMatching(/<system-update>.*build/is) },
  ]);
  expect(contexts.every(({ messages }) => JSON.stringify(messages[0]) === JSON.stringify(contexts[0].messages[0]))).toBe(true);
  expect(new ModeController('plan').transformContext([])).toHaveLength(1);
  const controller = new ModeController('plan');
  const corrected = controller.transformContext([]);
  expect(controller.transformContext(corrected)).toEqual(corrected);
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
  expect(contexts[1].messages.at(-1)).toMatchObject({
    role: 'user', content: expect.stringMatching(/<system-update>.*build/is),
  });
});
