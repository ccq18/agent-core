import { readFile, rm } from 'node:fs/promises';
import { join } from 'node:path';
import { getCurrentTools } from '@earendil-works/pi-ai';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import {
  JsonlTreeSessionStore, matchWildcard, PermissionEngine,
  type AgentCoreEvent, type AgentRuntime, type ApprovalRequest, type PermissionRule,
} from '../src/index.js';
import { call, reply, runtime, testDirectory, textResult, tool } from './harness-fixtures.js';

let dataDir: string;
let core: AgentRuntime | undefined;
beforeEach(async () => { dataDir = await testDirectory(); });
afterEach(async () => {
  try {
    await core?.dispose();
    core = undefined;
  } finally {
    await rm(dataDir, { recursive: true, force: true });
  }
});

it('matches complete glob patterns and applies the last matching rule across ordered rulesets', () => {
  const engine = new PermissionEngine();
  expect(matchWildcard('*.?s', 'lib/index.ts')).toBe(true);
  expect(matchWildcard('a[1].ts', 'a[1].ts')).toBe(true);
  expect(matchWildcard('Build*', 'build.log', true)).toBe(true);
  expect(matchWildcard('file', 'file\n')).toBe(false);
  expect(matchWildcard('?', '😀')).toBe(true);
  const deny: PermissionRule = { action: 'ed?t', resource: '*', effect: 'deny' };
  const allow: PermissionRule = { action: 'edit', resource: '/Repo/*', effect: 'allow' };
  expect(engine.evaluate('edit', '/Repo/x', [deny], [allow])).toBe('allow');
  expect(engine.evaluate('edit', '/repo/x', [deny], [allow]))
    .toBe(process.platform === 'darwin' || process.platform === 'win32' ? 'allow' : 'deny');
  expect(engine.evaluate('edit', '/Repo/x', [allow, deny])).toBe('deny');
  expect(engine.evaluate('read', 'x', [deny])).toBe('allow');
  expect(new PermissionEngine(undefined, 'ask').evaluate('read', 'x', [deny])).toBe('ask');
});

it('keeps a tool visible when only single-character resources are denied', () => {
  const engine = new PermissionEngine();
  const deny: PermissionRule = { action: 'edit', resource: '?', effect: 'deny' };
  expect(engine.isToolGloballyDenied(tool('edit'), [[deny]])).toBe(false);
  expect(engine.evaluate('edit', 'ab', [deny])).toBe('allow');
  expect(engine.evaluate('edit', 'a', [deny])).toBe('deny');
  expect(engine.isToolGloballyDenied(tool('edit'), [[{ ...deny, resource: '**' }]])).toBe(true);
});

it('hides a globally denied tool while retaining a tool with a narrower exception', async () => {
  const visible: string[][] = [];
  const executed = vi.fn(async () => textResult('edited'));
  const rules: PermissionRule[] = [
    { action: 'shell', resource: '*', effect: 'deny' },
    { action: 'edit', resource: '*', effect: 'deny' },
    { action: 'edit', resource: 'docs/*', effect: 'allow' },
  ];
  core = runtime(dataDir, (model, transcript) => {
    visible.push(getCurrentTools(transcript.messages).map(({ name }) => name));
    const results = transcript.messages.filter(({ role }) => role === 'toolResult');
    if (results.length) {
      expect(results).toMatchObject([
        { toolName: 'edit', isError: true, content: [{ text: expect.stringContaining('private/key') }] },
        { toolName: 'edit', isError: false },
      ]);
      return reply(model, { type: 'text', text: 'done' });
    }
    return reply(model, call('edit', { paths: ['docs/a', 'private/key'] }, 'denied'),
      call('edit', { paths: ['docs/b'] }, 'allowed'));
  }, { permissions: rules, tools: [
    tool('shell', { permissionAction: 'shell' }),
    tool('edit', {
      parameters: { type: 'object', properties: { paths: { type: 'array', items: { type: 'string' } } }, required: ['paths'] },
      resolveResources: ({ paths }) => paths as string[], execute: executed,
    }),
  ] });
  const session = await core.createSession();
  await expect(session.prompt('edit')).resolves.toMatchObject({ text: 'done' });
  expect(visible[0]).toContain('edit');
  expect(visible[0]).not.toContain('shell');
  expect(executed).toHaveBeenCalledTimes(1);
  const engine = new PermissionEngine();
  expect(engine.isToolGloballyDenied(tool('edit'), [[
    ...rules, { action: 'edit', resource: 'docs/*', effect: 'deny' },
  ]])).toBe(true);
});

it('allows unmatched tools with local rules and supports once with an explicit ask default', async () => {
  const executed = vi.fn(async () => textResult('ran'));
  const stream: Parameters<typeof runtime>[1] = (model, context) =>
    context.messages.at(-1)?.role === 'toolResult'
      ? reply(model, { type: 'text', text: 'done' }) : reply(model, call('work'));
  const toolset = [tool('work', { execute: executed, resolveResources: () => [] })];
  const legacy = runtime(dataDir, stream, { tools: toolset, permissions: [
    { action: 'other', resource: '*', effect: 'ask' },
  ] });
  try {
    const open = await legacy.createSession({
      permissions: [{ action: 'unrelated', resource: '*', effect: 'deny' }],
    });
    open.subscribe((event) => {
      if (event.type === 'approval_requested') open.replyApproval(event.request.id, { kind: 'reject' });
    });
    await expect(open.prompt('default')).resolves.toMatchObject({ text: 'done' });
  } finally {
    await legacy.dispose();
  }

  core = runtime(dataDir, stream, { tools: toolset, permissions: [], defaultPermissionEffect: 'ask' });
  const session = await core.createSession();
  const first = Promise.withResolvers<ApprovalRequest>();
  const second = Promise.withResolvers<ApprovalRequest>();
  const events: AgentCoreEvent[] = [];
  session.subscribe((event) => {
    events.push(event);
    if (event.type === 'approval_requested') {
      (events.filter(({ type }) => type === 'approval_requested').length === 1 ? first : second).resolve(event.request);
    }
  });
  const run = session.prompt('explicit');
  const request = await first.promise;
  expect(request).toMatchObject({
    sessionId: session.id, runId: expect.any(String), toolCallId: 'call-work',
    toolName: 'work', action: 'work', resources: ['*'], args: {}, createdAt: expect.any(Number),
  });
  expect(session.getPendingApprovals()).toEqual([request]);
  expect(executed).toHaveBeenCalledTimes(1);
  session.replyApproval(request.id, { kind: 'once' });
  await expect(run).resolves.toMatchObject({ text: 'done' });
  expect(executed).toHaveBeenCalledTimes(2);
  expect(events).toContainEqual(expect.objectContaining({
    type: 'approval_resolved', requestId: request.id, decision: { kind: 'once' },
  }));
  const next = session.prompt('another call');
  session.replyApproval((await second.promise).id, { kind: 'once' });
  await next;
  expect(session.getPendingApprovals()).toEqual([]);
});

it('persists always approvals and only re-evaluates pending requests from the same session', async () => {
  const requests: ApprovalRequest[] = [];
  const engine = new PermissionEngine(dataDir, 'ask');
  const gated = tool('write', { permissionAction: 'edit', resolveResources: ({ path }) => [String(path)] });
  const ask = (sessionId: string, runId: string, path: string) => engine.beforeToolCall({
    sessionId, runId, toolCallId: path, tool: gated, args: { path },
    rulesets: () => [], onApprovalRequested: (request) => requests.push(request),
  });
  const first = ask('a', 'run1', 'docs/a');
  const sameSession = ask('a', 'run2', 'docs/b');
  const otherSession = ask('b', 'run1', 'docs/c');
  engine.replyApproval(requests[0].id, { kind: 'always', savePattern: 'docs/*' }, 'a');
  await expect(Promise.all([first, sameSession])).resolves.toEqual([undefined, undefined]);
  expect(engine.getPendingApprovals('b')).toEqual([requests[2]]);
  expect(JSON.parse(await readFile(join(dataDir, 'approvals.json'), 'utf8')))
    .toEqual([{ action: 'edit', resource: 'docs/*', effect: 'allow' }]);
  engine.replyApproval(requests[2].id, { kind: 'once' }, 'b');
  await otherSession;
  const reloaded = new PermissionEngine(dataDir, 'ask');
  await expect(reloaded.beforeToolCall({
    sessionId: 'a', runId: 'reloaded', toolCallId: 'read', tool: gated, args: { path: 'docs/d' },
    rulesets: () => [],
    onApprovalRequested: () => { throw Error('saved permission should avoid prompting'); },
  })).resolves.toBeUndefined();
});

it('rejects other pending requests from the same run and includes correction feedback', async () => {
  const engine = new PermissionEngine(undefined, 'ask');
  const gated = tool('work');
  const ask = (runId: string, toolCallId: string) => engine.beforeToolCall({
    sessionId: 'session', runId, toolCallId, tool: gated, args: {}, rulesets: () => [],
  });
  const first = ask('run', 'a');
  const second = ask('run', 'b');
  const differentRun = ask('other-run', 'c');
  const requests = engine.getPendingApprovals('session');
  engine.replyApproval(requests[0].id, { kind: 'reject', feedback: 'Use a read-only query.' }, 'session');
  for (const result of await Promise.all([first, second])) {
    expect(result).toMatchObject({ block: true, reason: expect.stringContaining('Use a read-only query.') });
  }
  expect(engine.getPendingApprovals('session')).toEqual([requests[2]]);
  engine.replyApproval(requests[2].id, { kind: 'once' }, 'session');
  await differentRun;
});

it('returns rejection feedback to the model and emits approval_resolved', async () => {
  const feedback = 'Use a read-only query.';
  core = runtime(dataDir, (model, context) => {
    const result = context.messages.at(-1);
    if (result?.role !== 'toolResult') return reply(model, call('work'));
    expect(result).toMatchObject({ isError: true, content: [{ text: expect.stringContaining(feedback) }] });
    return reply(model, { type: 'text', text: 'adjusted' });
  }, { tools: [tool('work', { execute: async () => { throw Error('must not run'); } })], permissions: [], defaultPermissionEffect: 'ask' });
  const session = await core.createSession();
  const events: AgentCoreEvent[] = [];
  session.subscribe((event) => {
    events.push(event);
    if (event.type === 'approval_requested') session.replyApproval(event.request.id, { kind: 'reject', feedback });
  });
  await expect(session.prompt('try')).resolves.toMatchObject({ text: 'adjusted' });
  expect(events).toContainEqual(expect.objectContaining({ type: 'approval_resolved', decision: { kind: 'reject', feedback } }));
});

it('cancels an unanswered approval without running the tool and preserves messages', async () => {
  const execute = vi.fn(async () => textResult('unexpected'));
  const storage = new JsonlTreeSessionStore(dataDir);
  core = runtime(dataDir, (model) => reply(model, call('work')), {
    tools: [tool('work', { execute })], permissions: [], defaultPermissionEffect: 'ask', sessionStorage: storage,
  });
  const session = await core.createSession();
  const requested = Promise.withResolvers<void>();
  session.subscribe((event) => { if (event.type === 'approval_requested') requested.resolve(); });
  const run = session.prompt('abort me');
  await requested.promise;
  session.abort();
  await expect(run).resolves.toMatchObject({ state: 'cancelled' });
  expect(session.getPendingApprovals()).toEqual([]);
  expect(execute).not.toHaveBeenCalled();
  expect((await storage.projectActiveMessages(session.id)).messages.map(({ role }) => role))
    .toEqual(expect.arrayContaining(['user', 'assistant', 'toolResult']));
});

it('applies a session ruleset above runtime defaults and reloads always approval through SessionHandle', async () => {
  const execute = vi.fn(async () => textResult('executed'));
  const stream: Parameters<typeof runtime>[1] = (model, context) => {
    if (context.messages.at(-1)?.role !== 'toolResult') return reply(model, call('work', { path: 'docs/file' }));
    expect(context.messages.at(-1)).toMatchObject({ role: 'toolResult', isError: false });
    return reply(model, { type: 'text', text: 'done' });
  };
  const config = {
    permissions: [{ action: 'write', resource: '*', effect: 'deny' as const }],
    tools: [tool('work', {
      permissionAction: 'write',
      parameters: { type: 'object', properties: { path: { type: 'string' } }, required: ['path'] },
      resolveResources: ({ path }) => [String(path)], execute,
    })],
  };
  core = runtime(dataDir, stream, config);
  const session = await core.createSession({ permissions: [{ action: 'write', resource: 'docs/*', effect: 'ask' }] });
  const events: AgentCoreEvent[] = [];
  session.subscribe((event) => {
    events.push(event);
    if (event.type === 'approval_requested') session.replyApproval(event.request.id, { kind: 'always' });
  });
  await expect(session.prompt('First time')).resolves.toMatchObject({ text: 'done' });
  expect(events).toContainEqual(expect.objectContaining({ type: 'approval_resolved', decision: { kind: 'always' } }));
  expect(JSON.parse(await readFile(join(dataDir, 'approvals.json'), 'utf8')))
    .toEqual([{ action: 'write', resource: 'docs/file', effect: 'allow' }]);
  await core.dispose();

  core = runtime(dataDir, stream, config);
  const restored = await core.openSession(session.id);
  const reopened: ApprovalRequest[] = [];
  restored.subscribe((event) => {
    if (event.type === 'approval_requested') {
      reopened.push(event.request);
      restored.replyApproval(event.request.id, { kind: 'reject', feedback: 'An approval should not be needed.' });
    }
  });
  await expect(restored.prompt('New runtime')).resolves.toMatchObject({ text: 'done' });
  expect(reopened).toEqual([]);
  expect(restored.getPendingApprovals()).toEqual([]);
  expect(execute).toHaveBeenCalledTimes(2);
});
