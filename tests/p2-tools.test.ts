import { lstat, rm } from 'node:fs/promises';
import { getCurrentTools } from '@earendil-works/pi-ai';
import { expect, it, vi } from 'vitest';
import { PermissionEngine, type AgentRuntime, type PermissionRule } from '../src/index.js';
import { call, reply, runtime, testDirectory, textResult, tool } from './harness-fixtures.js';

it('reuses permission filtering while binding tool callbacks to each run, and invalidates on all visibility changes', async () => {
  const dir = await testDirectory();
  const filter = vi.spyOn(PermissionEngine.prototype, 'isToolGloballyDenied');
  const visible: string[][] = [];
  const runs: Array<{ sessionId: string; runId: string; toolCallId: string; label: string }> = [];
  const done = { type: 'text' as const, text: 'Done' };
  let mcpTool = 'first';
  const approvalTool = tool('approve', { readOnly: true });
  const mutableRules: PermissionRule[] = [{ action: 'approve', resource: '*', effect: 'ask' }];
  const allowedTools = ['approve'];
  const deniedTools: string[] = [];
  const makeWork = (label: string) => tool('work', {
    label,
    async execute(_, { sessionId, runId, toolCallId }) {
      runs.push({ sessionId, runId, toolCallId, label });
      return textResult(label);
    },
  });
  let core: AgentRuntime | undefined;
  try {
    core = runtime(dir, (model, context) => {
      const tools = getCurrentTools(context.messages).map(({ name }) => name);
      visible.push(tools);
      const last = context.messages.at(-1);
      if (last?.role !== 'user') return reply(model, done);
      const user = JSON.stringify(last.content);
      if (user.includes('Execute work')) return reply(model, call('work', {}, 'tracked-work'));
      if (user.includes('Approval request')) return reply(model, call('approve'));
      return reply(model, done);
    }, {
      tools: [makeWork('original'), approvalTool],
      permissions: mutableRules,
      subagents: { narrow: { allowedTools, deniedTools }, readonly: { readOnly: true, allowedTools: ['approve'] } },
      mcpServers: [{ id: 'docs', transport: 'stdio' }],
      mcpClientFactory: () => ({
        listTools: async () => [{ name: mcpTool, inputSchema: {} }],
        callTool: async () => textResult('mcp'), close: async () => undefined,
      }),
    });
    const session = await core.createSession();
    const first = await session.prompt('Execute work');
    const calls = filter.mock.calls.length;
    expect(calls).toBeGreaterThan(0);
    const second = await session.prompt('Execute work');
    expect(filter).toHaveBeenCalledTimes(calls);
    expect(first.runId).not.toBe(second.runId);

    const unregister = core.registerTools([makeWork('override')]);
    const third = await session.prompt('Execute work');
    expect(filter.mock.calls.length).toBeGreaterThan(calls);
    unregister();
    const fourth = await session.prompt('Execute work');
    expect(runs).toEqual([first, second, third, fourth].map(({ runId }, i) => ({
      runId, sessionId: session.id, toolCallId: 'tracked-work', label: i === 2 ? 'override' : 'original',
    })));

    mcpTool = 'second';
    await core.reloadMcp();
    await session.prompt('After reload');
    expect(visible.at(-1)).toContain('docs__second');
    expect(visible.at(-1)).not.toContain('docs__first');

    await session.setMode('plan');
    await session.prompt('In Plan');
    expect(visible.at(-1)).not.toContain('work');
    expect(visible.at(-1)).toContain('approve');
    await session.setMode('build');
    await session.prompt('In Build');
    expect(visible.at(-1)).toContain('work');

    const beforeApproval = filter.mock.calls.length;
    let approvals = 0;
    session.subscribe((event) => {
      if (event.type !== 'approval_requested') return;
      approvals++;
      session.replyApproval(event.request.id, { kind: 'always' });
    });
    await session.prompt('Approval request');
    expect(filter.mock.calls.length).toBeGreaterThan(beforeApproval);
    const afterApproval = filter.mock.calls.length;
    await session.prompt('Approval request again');
    expect(approvals).toBe(1);
    expect(filter).toHaveBeenCalledTimes(afterApproval);

    mutableRules.push({ action: 'work', resource: '*', effect: 'deny' });
    await session.prompt('Edited rules');
    expect(visible.at(-1)).not.toContain('work');
    mutableRules.at(-1)!.effect = 'allow';
    await session.prompt('Edited rule in place');
    expect(visible.at(-1)).toContain('work');

    const helper = await core.createSession({
      metadata: { parentSessionId: session.id, subagentProfile: 'narrow', depth: 1 },
    });
    await helper.prompt('Before profile change');
    expect(visible.at(-1)).toEqual(['approve']);
    allowedTools.push('work');
    deniedTools.push('approve');
    await helper.prompt('After profile change');
    expect(visible.at(-1)).toEqual(['work']);

    const readOnlyHelper = await core.createSession({
      metadata: { parentSessionId: session.id, subagentProfile: 'readonly', depth: 1 },
    });
    await readOnlyHelper.prompt('Before mutating the tool in place');
    expect(visible.at(-1)).toEqual(['approve']);
    approvalTool.readOnly = false;
    await readOnlyHelper.prompt('After mutating the tool in place');
    expect(visible.at(-1)).toEqual([]);
    approvalTool.readOnly = true;
    await readOnlyHelper.prompt('After restoring read-only');
    expect(visible.at(-1)).toEqual(['approve']);
  } finally {
    filter.mockRestore();
    try {
      await core?.dispose();
    } finally {
      await rm(dir, { recursive: true, force: true });
      await expect(lstat(dir)).rejects.toMatchObject({ code: 'ENOENT' });
    }
  }
});
