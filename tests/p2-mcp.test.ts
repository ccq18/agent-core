import { lstat, rm } from 'node:fs/promises';
import { setTimeout as delay } from 'node:timers/promises';
import { getCurrentTools } from '@earendil-works/pi-ai';
import { expect, it, vi } from 'vitest';
import { McpBridge, ToolRegistry, type AgentRuntime, type McpBridgeClient, type McpServerConfig } from '../src/index.js';
import { reply, runtime, testDirectory, textResult } from './harness-fixtures.js';

const server = (id: string, connectTimeoutMs?: number): McpServerConfig =>
  ({ id, transport: 'stdio', command: 'unused', connectTimeoutMs });
const client = (name: string, options: Partial<McpBridgeClient> = {}): McpBridgeClient => ({
  listTools: async () => [{ name, inputSchema: {} }],
  callTool: async () => textResult(name),
  close: vi.fn(async () => undefined),
  ...options,
});

it('isolates a 400ms factory at the 80ms deadline, registers the fast server immediately and closes the late client', async () => {
  const registry = new ToolRegistry();
  const slow = client('late');
  let slowSignal: AbortSignal | undefined;
  const bridge = new McpBridge(registry, [server('slow', 80), server('fast')], async ({ id }, signal) => {
    if (id === 'slow') slowSignal = signal;
    await delay(id === 'slow' ? 400 : 10);
    return id === 'slow' ? slow : client('early');
  }, { connectTimeoutMs: 1000 });
  try {
    const start = performance.now();
    const init = bridge.initialize();
    await vi.waitFor(() => expect(registry.snapshot().definitions.has('fast__early')).toBe(true), { interval: 5 });
    await init;
    expect(performance.now() - start).toBeLessThan(350);
    expect(slowSignal?.aborted).toBe(true);
    expect(bridge.getCatalog()).toMatchObject([
      { id: 'slow', state: 'error', error: expect.stringContaining('80ms'), toolCount: 0 },
      { id: 'fast', state: 'ready', toolCount: 1 },
    ]);
    await vi.waitFor(() => expect(slow.close).toHaveBeenCalledOnce());
    expect(registry.snapshot().definitions.has('slow__late')).toBe(false);
  } finally {
    await bridge.dispose();
  }
});

it('uses one deadline across connect and listTools, closes an acquired client, and admits another connection', async () => {
  const registry = new ToolRegistry();
  const listed = Promise.withResolvers<Awaited<ReturnType<McpBridgeClient['listTools']>>>();
  const closing = Promise.withResolvers<void>();
  const slowClose = vi.fn(() => closing.promise);
  const listTools = vi.fn(() => listed.promise);
  const factory = vi.fn(async ({ id }: McpServerConfig) => {
    await delay(20);
    return id === 'slow' ? client('late', { listTools, close: slowClose }) : client('ok');
  });
  const bridge = new McpBridge(registry, [server('slow'), server('fast')], factory, {
    connectTimeoutMs: 80, maxConcurrentConnects: 1,
  });
  try {
    const initial = bridge.initialize();
    await vi.waitFor(() => expect(listTools).toHaveBeenCalled(), { interval: 5 });
    expect(factory).toHaveBeenCalledTimes(1);
    await initial;
    expect(slowClose).toHaveBeenCalledOnce();
    expect(listTools).toHaveBeenCalledWith({ timeoutMs: undefined, signal: expect.objectContaining({ aborted: true }) });
    expect(registry.snapshot().visible.map(({ name }) => name)).toEqual(['fast__ok']);
    listed.resolve([{ name: 'late', inputSchema: {} }]);
  } finally {
    listed.resolve([]);
    closing.resolve();
    await bridge.dispose();
  }
});

it('applies the deadline to paginated discovery over the actual Pi stdio transport', async () => {
  const registry = new ToolRegistry();
  const script = `
    const lines = require('node:readline').createInterface({ input: process.stdin });
    const slow = process.argv[1] === 'slow';
    lines.on('line', (line) => {
      const { id, method, params } = JSON.parse(line);
      const write = (result) => console.log(JSON.stringify({ jsonrpc: '2.0', id, result }));
      if (method === 'initialize') {
        write({ protocolVersion: params.protocolVersion, capabilities: { tools: {} }, serverInfo: { name: 'fixture', version: '1' } });
      } else if (method === 'tools/list' && !params?.cursor) {
        write({ tools: [{ name: 'first', inputSchema: { type: 'object' } }], ...(slow && { nextCursor: 'next' }) });
      }
    });
  `;
  const bridge = new McpBridge(registry, ['slow', 'fast'].map((id) => ({
    ...server(id, id === 'slow' ? 200 : 2000), command: process.execPath, args: ['-e', script, id],
  })));
  try {
    await bridge.initialize();
    expect(bridge.getCatalog()).toMatchObject([
      { state: 'error', error: expect.stringContaining('200ms'), toolCount: 0 },
      { state: 'ready', toolCount: 1 },
    ]);
    expect(registry.snapshot().visible.map(({ name }) => name)).toEqual(['fast__first']);
  } finally {
    await bridge.dispose();
  }
});

it.each(['on-first-session', 'on-first-prompt', 'background'] as const)(
  'starts and waits according to the %s policy', async (policy) => {
    const dir = await testDirectory();
    const releaseSlow = Promise.withResolvers<void>();
    const seen: string[][] = [];
    const factory = vi.fn(async ({ id }: McpServerConfig) => {
      if (id === 'slow') await releaseSlow.promise;
      return client('tool');
    });
    let core: AgentRuntime | undefined;
    try {
      const runtimeHandle = core = runtime(dir, (model, context) => {
        seen.push(getCurrentTools(context.messages).map(({ name }) => name));
        return reply(model, { type: 'text', text: 'Done' });
      }, { mcpServers: [server('fast'), server('slow')], mcpClientFactory: factory, mcpOptions: { initPolicy: policy } });
      expect(await core.listSessions()).toEqual([]);
      if (policy !== 'background') expect(factory).not.toHaveBeenCalled();
      const opening = core.createSession();
      if (policy === 'on-first-session') {
        const open = vi.fn();
        void opening.then(open, () => undefined);
        await vi.waitFor(() => expect(factory).toHaveBeenCalledTimes(2));
        expect(open).not.toHaveBeenCalled();
        releaseSlow.resolve();
      }
      const session = await opening;
      await session.close();
      const reopened = await core.openSession(session.id);
      if (policy === 'on-first-prompt') expect(factory).not.toHaveBeenCalled();
      if (policy === 'background') {
        await vi.waitFor(() => expect(runtimeHandle.getCatalog().tools.map(({ name }) => name)).toContain('fast__tool'));
        await reopened.prompt('Before slow finishes');
        expect(seen[0]).toContain('fast__tool');
        expect(seen[0]).not.toContain('slow__tool');
      }
      const prompt = reopened.prompt('After requesting tools');
      if (policy === 'on-first-prompt') {
        await vi.waitFor(() => expect(factory).toHaveBeenCalledTimes(2));
        expect(seen).toEqual([]);
      }
      releaseSlow.resolve();
      await prompt;
      await vi.waitFor(() => expect(runtimeHandle.getCatalog().tools.map(({ name }) => name)).toContain('slow__tool'));
      await reopened.prompt('Tools ready');
      expect(seen.at(-1)).toEqual(expect.arrayContaining(['fast__tool', 'slow__tool']));
    } finally {
      releaseSlow.resolve();
      try {
        await core?.dispose();
      } finally {
        await rm(dir, { recursive: true, force: true });
        await expect(lstat(dir)).rejects.toMatchObject({ code: 'ENOENT' });
      }
    }
  },
);

it('invalidates an older tool list on reload and closes a factory result arriving after dispose', async () => {
  const registry = new ToolRegistry();
  const list = Promise.withResolvers<Awaited<ReturnType<McpBridgeClient['listTools']>>>();
  const listed = Promise.withResolvers<void>();
  const lateFactory = Promise.withResolvers<McpBridgeClient>();
  const old = client('old', { listTools() { listed.resolve(); return list.promise; } });
  const fresh = client('fresh');
  const late = client('late');
  const factory = vi.fn().mockResolvedValueOnce(old).mockResolvedValueOnce(fresh).mockReturnValueOnce(lateFactory.promise);
  const bridge = new McpBridge(registry, [server('reload')], factory);
  try {
    const initial = bridge.initialize();
    await listed.promise;
    await bridge.reload();
    await initial;
    expect(old.close).toHaveBeenCalledOnce();
    expect(registry.snapshot().visible.map(({ name }) => name)).toEqual(['reload__fresh']);
    list.resolve([{ name: 'old', inputSchema: {} }]);
    const pending = bridge.reload();
    await vi.waitFor(() => expect(factory).toHaveBeenCalledTimes(3));
    await bridge.dispose();
    await pending;
    lateFactory.resolve(late);
    await vi.waitFor(() => expect(late.close).toHaveBeenCalledOnce());
    expect(registry.snapshot().visible).toEqual([]);
    expect(fresh.close).toHaveBeenCalledOnce();
  } finally {
    list.resolve([]);
    lateFactory.resolve(late);
    await bridge.dispose();
  }
});

it('reconnects healthy servers while isolating hanging and rejected ready-client closes', async () => {
  const registry = new ToolRegistry();
  const release = Promise.withResolvers<void>();
  const hanging = client('old', { close: () => release.promise });
  const rejecting = client('old', { close: async () => { throw new Error('Could not close transport'); } });
  let initial = true;
  const bridge = new McpBridge(registry, ['hanging', 'rejecting', 'healthy'].map((id) => server(id)),
    ({ id }) => initial && id === 'hanging' ? hanging : initial && id === 'rejecting' ? rejecting
      : client(initial ? 'old' : 'new'));
  let reload: Promise<void> | undefined;
  try {
    await bridge.initialize();
    initial = false;
    reload = bridge.reload();
    void reload.catch(() => undefined);
    await vi.waitFor(() => expect(registry.snapshot().definitions.has('healthy__new')).toBe(true));
    await expect(reload).resolves.toBeUndefined();
    expect(registry.snapshot().visible.map(({ name }) => name).sort())
      .toEqual(['hanging__new', 'healthy__new', 'rejecting__new']);
  } finally {
    release.resolve();
    await reload?.catch(() => undefined);
    await bridge.dispose();
  }
});

it('drains a close already in flight when a second reload and disposal supersede it', async () => {
  const entered = Promise.withResolvers<void>();
  const release = Promise.withResolvers<void>();
  const old = client('old', { close() { entered.resolve(); return release.promise; } });
  const factory = vi.fn().mockReturnValueOnce(old).mockImplementation(() => client('new'));
  const bridge = new McpBridge(new ToolRegistry(), [server('reload')], factory);
  const disposed = vi.fn();
  let first: Promise<void> | undefined;
  let second: Promise<void> | undefined;
  try {
    await bridge.initialize();
    first = bridge.reload();
    await entered.promise;
    second = bridge.reload();
    void bridge.dispose().then(disposed, () => undefined);
    await new Promise<void>((resolve) => setImmediate(resolve));
    expect(disposed).not.toHaveBeenCalled();
  } finally {
    release.resolve();
    await Promise.allSettled([first, second, bridge.dispose()]);
  }
  expect(disposed).toHaveBeenCalledOnce();
});

it('finishes disposal when a ready-client close never settles before its deadline', async () => {
  const release = Promise.withResolvers<void>();
  const bridge = new McpBridge(new ToolRegistry(), [server('docs')], () => client('old', { close: () => release.promise }));
  vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
  try {
    await bridge.initialize();
    const disposed = vi.fn();
    void bridge.dispose().then(disposed, () => undefined);
    await vi.advanceTimersByTimeAsync(2000);
    expect(disposed).toHaveBeenCalledOnce();
  } finally {
    release.resolve();
    await bridge.dispose();
    vi.useRealTimers();
  }
});

it.each(['on-first-session', 'on-first-prompt'] as const)(
  'keeps the first %s waiter on the newest MCP generation', async (initPolicy) => {
    const dir = await testDirectory();
    const old = Promise.withResolvers<McpBridgeClient>();
    const latest = Promise.withResolvers<McpBridgeClient>();
    const factory = vi.fn().mockReturnValueOnce(old.promise).mockReturnValueOnce(latest.promise);
    const seen: string[][] = [];
    const core = runtime(dir, (model, context) => {
      seen.push(getCurrentTools(context.messages).map(({ name }) => name));
      return reply(model, { type: 'text', text: 'Done' });
    }, { mcpServers: [server('docs')], mcpClientFactory: factory, mcpOptions: { initPolicy } });
    let first: Promise<unknown> | undefined;
    let reloading: Promise<void> | undefined;
    try {
      const opening = core.createSession();
      const waiting = initPolicy === 'on-first-session' ? opening : (await opening).prompt('First');
      first = initPolicy === 'on-first-session' ? opening.then((session) => session.prompt('First')) : waiting;
      await vi.waitFor(() => expect(factory).toHaveBeenCalledTimes(1));
      reloading = core.reloadMcp();
      await vi.waitFor(() => expect(factory).toHaveBeenCalledTimes(2));
      await expect(Promise.race([waiting.then(() => 'released'), delay(50, 'waiting')])).resolves.toBe('waiting');
      latest.resolve(client('fresh'));
      await reloading;
      await expect(first).resolves.toMatchObject({ state: 'completed' });
      expect(seen).toEqual([expect.arrayContaining(['docs__fresh'])]);
    } finally {
      old.resolve(client('stale'));
      latest.resolve(client('fresh'));
      try {
        await Promise.allSettled([first, reloading]);
        await core.dispose();
      } finally {
        await rm(dir, { recursive: true, force: true });
        await expect(lstat(dir)).rejects.toMatchObject({ code: 'ENOENT' });
      }
    }
  },
);
