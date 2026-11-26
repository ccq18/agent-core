import {
  McpClient, normalizeMcpInputSchema, StdioTransport, StreamableHttpTransport, toLlmContent,
  type McpServerConfig as PiMcpServerConfig, type Tool,
} from '@earendil-works/pi-mcp';
import type { JsonValue } from '@earendil-works/pi-ai';
import { Semaphore, waitWithSignal } from '../runtime/concurrency.js';
import type { McpOptions, McpServerConfig, McpServerStatus } from '../types/config.js';
import type { CoreToolDefinition } from '../types/tool.js';
import type { ToolRegistry } from './tool-registry.js';

export type McpBridgeClient = Pick<McpClient, 'listTools' | 'callTool' | 'close'> &
  Partial<Pick<McpClient, 'onClose' | 'serverCapabilities'>>;
export type McpClientFactory = (server: McpServerConfig, signal?: AbortSignal) => McpBridgeClient | Promise<McpBridgeClient>;

interface Connection {
  client: McpBridgeClient;
  ready: boolean;
  unregister?: () => void;
  unsubscribe?: () => void;
}

interface McpServer {
  config: McpServerConfig;
  status: McpServerStatus;
  connection?: Connection;
  closing?: Promise<void>;
}

const CLOSE_TIMEOUT_MS = 1_000;

function asServerConfig(config: McpServerConfig | PiMcpServerConfig, index: number): McpServerConfig {
  if ('transport' in config) return config;
  return {
    ...config, id: `mcp-${index + 1}`, transport: 'url' in config ? 'http' : 'stdio',
    timeoutMs: config.timeout === undefined ? undefined : config.timeout * 1000,
  };
}

const errorMessage = (error: unknown) => error instanceof Error ? error.message : String(error);
const closeClient = async (client: McpBridgeClient) => { await client.close(); };

async function connectClient(server: McpServerConfig, signal?: AbortSignal): Promise<McpClient> {
  const { transport, command, args, env, cwd, url, headers, timeoutMs } = server;
  const client = new McpClient({ name: 'agent-core', version: '0.1.0', requestTimeoutMs: timeoutMs });
  const abort = () => { void closeClient(client).catch(() => undefined); };
  signal?.throwIfAborted();
  signal?.addEventListener('abort', abort, { once: true });
  try {
    if (transport === 'stdio') {
      if (!command) throw new Error('MCP stdio transport requires a command');
      await client.connect(new StdioTransport({ command, args, env, cwd }));
    } else {
      if (!url) throw new Error('MCP HTTP transport requires a URL');
      await client.connect(new StreamableHttpTransport({ url, headers }));
    }
    return client;
  } finally {
    signal?.removeEventListener('abort', abort);
  }
}

function createTool(server: McpServerConfig, client: McpBridgeClient, tool: Tool): CoreToolDefinition {
  const name = `${server.id}__${tool.name}`;
  return {
    name,
    label: tool.title ?? tool.annotations?.title ?? name,
    description: tool.description ?? name,
    parameters: normalizeMcpInputSchema(tool.inputSchema),
    permissionAction: `mcp:${server.id}`,
    resolveResources: () => [tool.name],
    async execute(args, { signal }) {
      try {
        const result = await client.callTool(tool.name, args, { signal, timeoutMs: server.timeoutMs });
        const content = toLlmContent(result);
        return {
          content: result.isError && !content.length
            ? [{ type: 'text', text: `MCP tool "${name}" returned an error` }] : content,
          structuredContent: result.structuredContent as JsonValue | undefined,
          isError: result.isError === true,
        };
      } catch (error) {
        return { content: [{ type: 'text', text: `MCP tool "${name}": ${errorMessage(error)}` }], isError: true };
      }
    },
  };
}

export class McpBridge {
  private readonly servers: McpServer[];
  private readonly slots?: Semaphore;
  private readonly closing = new Set<Promise<void>>();
  private generation = 0;
  private controller = new AbortController();
  private initialization?: Promise<void>;
  private disposal?: Promise<void>;

  constructor(
    private readonly registry: ToolRegistry,
    configs: readonly (McpServerConfig | PiMcpServerConfig)[] = [],
    private readonly clientFactory: McpClientFactory = connectClient,
    private readonly options: McpOptions = {},
  ) {
    if (options.maxConcurrentConnects !== undefined) this.slots = new Semaphore(options.maxConcurrentConnects);
    this.servers = configs.map(asServerConfig).map((config) => ({
      config,
      status: {
        id: config.id, transport: config.transport, toolCount: 0,
        ...(config.enabled === false ? { state: 'disabled' } : { state: 'error', error: 'Not connected' }),
      },
    }));
  }

  getCatalog(): McpServerStatus[] {
    return this.servers.map(({ status }) => ({ ...status }));
  }

  initialize(): Promise<void> {
    if (this.disposal) return this.disposal;
    this.initialization ??= this.startGeneration();
    return this.initialization;
  }

  reload(): Promise<void> {
    if (this.disposal) return this.disposal;
    const previous = this.nextGeneration();
    for (const server of this.servers) this.disconnect(server);
    const initializing = this.startGeneration();
    this.initialization = initializing;
    previous.abort(new Error('MCP reloaded'));
    return initializing;
  }

  /** @internal Let runtime handles shut down before closing ready transports. */
  cancelInitialization(): void {
    this.controller.abort(new Error('MCP startup cancelled'));
  }

  dispose(): Promise<void> {
    if (!this.disposal) {
      const previous = this.nextGeneration();
      for (const server of this.servers) this.disconnect(server);
      this.disposal = Promise.all(this.closing).then(() => undefined);
      previous.abort(new Error('MCP disposed'));
    }
    return this.disposal;
  }

  private nextGeneration(): AbortController {
    this.generation++;
    this.registry.invalidate();
    const previous = this.controller;
    this.controller = new AbortController();
    return previous;
  }

  private async startGeneration(): Promise<void> {
    const { generation, controller: { signal } } = this;
    // Publish the initialization promise before calling host code.
    await Promise.resolve();
    if (signal.aborted) return;
    await Promise.all(this.servers.filter(({ config }) => config.enabled !== false)
      .map((server) => this.connect(server, generation, signal)));
  }

  private async connect(server: McpServer, generation: number, generationSignal: AbortSignal): Promise<void> {
    const { config } = server;
    const timeout = config.connectTimeoutMs ?? this.options.connectTimeoutMs ?? 30_000;
    let release: (() => void) | undefined;
    let timer: ReturnType<typeof setTimeout> | undefined;
    try {
      // Only a server's own ready transport can delay reconnecting that server.
      if (server.closing) await waitWithSignal(server.closing, generationSignal);
      if (this.slots) release = await this.slots.acquire(generationSignal);
      generationSignal.throwIfAborted();
      if (!Number.isInteger(timeout) || timeout < 1 || timeout > 2_147_483_647) throw new RangeError('Invalid MCP connectTimeoutMs');
      const attempt = new AbortController();
      const signal = AbortSignal.any([generationSignal, attempt.signal]);
      timer = setTimeout(() => attempt.abort(new Error(`MCP startup timed out after ${timeout}ms`)), timeout);
      const { connection, tools } = await waitWithSignal(this.discover(server, attempt, signal), signal);
      signal.throwIfAborted();
      connection.unregister = this.registry.register(tools.map((tool) => createTool(config, connection.client, tool)));
      connection.ready = true;
      server.status = { id: config.id, transport: config.transport, state: 'ready', toolCount: tools.length };
    } catch (error) {
      if (generation !== this.generation) return;
      // A stale or unresponsive transport must not extend the startup deadline.
      this.disconnect(server);
      server.status = { id: config.id, transport: config.transport, state: 'error', error: errorMessage(error), toolCount: 0 };
    } finally {
      clearTimeout(timer);
      release?.();
    }
  }

  private async discover(server: McpServer, attempt: AbortController, signal: AbortSignal) {
    const { config } = server;
    const client = await this.clientFactory(config, signal);
    if (signal.aborted) {
      this.trackClose(client);
      signal.throwIfAborted();
    }
    const connection: Connection = { client, ready: false };
    server.connection = connection;
    connection.unsubscribe = client.onClose?.(() => {
      if (server.connection !== connection) return;
      if (!connection.ready) {
        attempt.abort(new Error('Connection closed'));
        return;
      }
      connection.unregister?.();
      server.status = { id: config.id, transport: config.transport, state: 'error', error: 'Connection closed', toolCount: 0 };
    });
    signal.throwIfAborted();
    const tools = client.serverCapabilities && !client.serverCapabilities.tools
      ? [] : await client.listTools({ signal, timeoutMs: config.timeoutMs });
    return { connection, tools };
  }

  private disconnect(server: McpServer): void {
    const { connection, config } = server;
    if (!connection) return;
    server.connection = undefined;
    connection.unregister?.();
    server.status = { id: config.id, transport: config.transport, state: 'error', toolCount: 0, error: 'Not connected' };
    const closing = this.trackClose(connection.client, connection.unsubscribe);
    if (!connection.ready) return;
    server.closing = closing;
    void closing.then(() => { if (server.closing === closing) server.closing = undefined; });
  }

  private trackClose(client: McpBridgeClient, unsubscribe?: () => void): Promise<void> {
    // Detach registry state and publish tracking before invoking custom transport callbacks.
    const attempt = Promise.resolve().then(async () => {
      try {
        unsubscribe?.();
      } finally {
        await closeClient(client);
      }
    });
    let timer: ReturnType<typeof setTimeout>;
    const deadline = new Promise<void>((resolve) => { timer = setTimeout(resolve, CLOSE_TIMEOUT_MS); });
    const closing = Promise.race([attempt, deadline]).catch(() => undefined).finally(() => {
      clearTimeout(timer);
      this.closing.delete(closing);
    });
    this.closing.add(closing);
    return closing;
  }
}
