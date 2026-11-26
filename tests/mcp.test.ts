import { rm } from 'node:fs/promises';
import { getCurrentTools } from '@earendil-works/pi-ai';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import type { AgentRuntime, McpBridgeClient, McpServerConfig } from '../src/index.js';
import { call, reply, runtime, testDirectory, textResult } from './harness-fixtures.js';

const servers: McpServerConfig[] = [
  { id: 'docs', transport: 'stdio', command: 'unused', timeoutMs: 1000 },
  { id: 'broken', transport: 'http', url: 'http://localhost/mcp' },
  { id: 'off', enabled: false, transport: 'stdio' },
];
let dir: string;
let core: AgentRuntime | undefined;
beforeEach(async () => { dir = await testDirectory(); });
afterEach(async () => {
  try {
    await core?.dispose();
    core = undefined;
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

it('registers tools on demand, forwards args and abort signal, and reports connection state', async () => {
  const close = vi.fn(async () => undefined);
  const brokenClose = vi.fn(async () => undefined);
  const callTool = vi.fn<McpBridgeClient['callTool']>(async () => textResult('MCP result'));
  const factory = vi.fn((server: McpServerConfig): McpBridgeClient => server.id === 'broken' ? {
    listTools: async () => { throw Error('Offline'); }, callTool, close: brokenClose,
  } : {
    listTools: async () => [{ name: 'search', inputSchema: {
      properties: { query: { type: 'string' } }, required: ['query'],
    } }],
    callTool, close,
  });
  core = runtime(dir, (model, context) => {
    const last = context.messages.at(-1);
    if (last?.role !== 'toolResult') {
      expect(getCurrentTools(context.messages).map(({ name }) => name)).toContain('docs__search');
      return reply(model, call('docs__search', { query: 'needle' }));
    }
    expect(last).toMatchObject({ isError: false, content: [expect.objectContaining({ text: 'MCP result' })] });
    return reply(model, { type: 'text', text: 'Done' });
  }, { mcpServers: servers, mcpClientFactory: factory });
  const session = await core.createSession();
  await expect(session.prompt('Search')).resolves.toMatchObject({ text: 'Done' });
  expect(callTool).toHaveBeenCalledWith('search', { query: 'needle' }, {
    signal: expect.any(AbortSignal), timeoutMs: 1000,
  });
  expect(factory.mock.calls.map(([server]) => server.id)).toEqual(['docs', 'broken']);
  expect(core.getCatalog().mcpServers).toEqual([
    { id: 'docs', transport: 'stdio', state: 'ready', toolCount: 1 },
    { id: 'broken', transport: 'http', state: 'error', toolCount: 0, error: 'Offline' },
    { id: 'off', transport: 'stdio', state: 'disabled', toolCount: 0 },
  ]);
  expect(brokenClose).toHaveBeenCalledTimes(1);
  await core.dispose();
  expect(close).toHaveBeenCalledTimes(1);
  expect(core.getCatalog().tools.map(({ name }) => name)).not.toContain('docs__search');
});

it('wraps MCP server and transport errors and applies rules to the original tool name', async () => {
  const tools = ['server_error', 'transport_error', 'protected', 'denied'];
  const callTool = vi.fn<McpBridgeClient['callTool']>(async (name) => {
    if (name === 'transport_error') throw Error('Connection lost');
    return { ...textResult(name === 'server_error' ? 'Invalid query' : 'Allowed'), isError: name === 'server_error' };
  });
  core = runtime(dir, (model, context) => {
    const results = context.messages.filter(({ role }) => role === 'toolResult');
    if (!results.length) return reply(model, ...tools.map((name) => call(`docs__${name}`)));
    expect(results).toMatchObject([
      { isError: true, content: [{ text: expect.stringContaining('Invalid query') }] },
      { isError: true, content: [{ text: expect.stringContaining('Connection lost') }] },
      { isError: false },
      { isError: true, content: [{ text: expect.stringContaining('Permission denied') }] },
    ]);
    return reply(model, { type: 'text', text: 'Handled' });
  }, {
    mcpServers: [servers[0]],
    mcpClientFactory: () => ({
      listTools: async () => tools.map((name) => ({ name, inputSchema: {} })),
      callTool, close: async () => undefined,
    }),
    permissions: [
      { action: 'mcp:docs', resource: 'protected', effect: 'ask' },
      { action: 'mcp:docs', resource: 'denied', effect: 'deny' },
    ],
  });
  const session = await core.createSession();
  const approvals: unknown[] = [];
  session.subscribe((event) => {
    if (event.type !== 'approval_requested') return;
    approvals.push(event.request);
    session.replyApproval(event.request.id, { kind: 'once' });
  });
  await expect(session.prompt('Check the tools')).resolves.toMatchObject({ text: 'Handled' });
  expect(approvals).toEqual([expect.objectContaining({ action: 'mcp:docs', resources: ['protected'] })]);
  expect(callTool.mock.calls.map(([name]) => name).sort()).toEqual(['protected', 'server_error', 'transport_error']);
});

it('unregisters tools when the transport closes and reloads a fresh tool catalog', async () => {
  const close = vi.fn(async () => undefined);
  let connection = 0;
  let notifyClosed: (() => void) | undefined;
  core = runtime(dir, (model) => reply(model, { type: 'text', text: 'Done' }), {
    mcpServers: [servers[0]],
    mcpClientFactory: () => {
      const name = `tool_${++connection}`;
      return {
        listTools: async () => [{ name, inputSchema: {} }],
        callTool: async () => textResult(name),
        onClose(listener) { notifyClosed = listener; return () => { notifyClosed = undefined; }; },
        close,
      };
    },
  });
  await core.createSession();
  expect(core.getCatalog().tools.map(({ name }) => name)).toContain('docs__tool_1');
  notifyClosed!();
  expect(core.getCatalog().mcpServers).toMatchObject([{ state: 'error', toolCount: 0 }]);
  expect(core.getCatalog().tools.map(({ name }) => name)).not.toContain('docs__tool_1');
  await core.reloadMcp();
  expect(core.getCatalog().mcpServers).toMatchObject([{ state: 'ready', toolCount: 1 }]);
  expect(core.getCatalog().tools.map(({ name }) => name)).toContain('docs__tool_2');
  expect(close).toHaveBeenCalledTimes(1);
});
