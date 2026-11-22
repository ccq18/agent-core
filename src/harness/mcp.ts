import {
  McpClient, normalizeMcpInputSchema, StdioTransport, StreamableHttpTransport, toLlmContent,
  type McpServerConfig as PiMcpServerConfig, type Tool,
} from '@earendil-works/pi-mcp';
import type { JsonValue } from '@earendil-works/pi-ai';
import type { McpServerConfig, McpServerStatus } from '../types/config.js';
import type { CoreToolDefinition } from '../types/tool.js';
import type { ToolRegistry } from './tool-registry.js';

export type McpBridgeClient = Pick<McpClient, 'listTools' | 'callTool' | 'close'> &
  Partial<Pick<McpClient, 'onClose' | 'serverCapabilities'>>;
export type McpClientFactory = (server: McpServerConfig) => McpBridgeClient | Promise<McpBridgeClient>;

interface Connection {
  client: McpBridgeClient;
  unregister?: () => void;
  unsubscribe?: () => void;
}

interface McpServer {
  config: McpServerConfig;
  status: McpServerStatus;
  connection?: Connection;
}

function asServerConfig(config: McpServerConfig | PiMcpServerConfig, index: number): McpServerConfig {
  if ('transport' in config) return config;
  return {
    ...config, id: `mcp-${index + 1}`, transport: 'url' in config ? 'http' : 'stdio',
    timeoutMs: config.timeout === undefined ? undefined : config.timeout * 1000,
  };
}

const errorMessage = (error: unknown) => error instanceof Error ? error.message : String(error);

async function connectClient(server: McpServerConfig): Promise<McpClient> {
  const { transport, command, args, env, cwd, url, headers, timeoutMs } = server;
  const client = new McpClient({ name: 'agent-core', version: '0.1.0', requestTimeoutMs: timeoutMs });
  if (transport === 'stdio') {
    if (!command) throw new Error('MCP stdio transport requires a command');
    await client.connect(new StdioTransport({ command, args, env, cwd }));
  } else {
    if (!url) throw new Error('MCP HTTP transport requires a URL');
    await client.connect(new StreamableHttpTransport({ url, headers }));
  }
  return client;
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
  private initialization?: Promise<void>;
  private disposal?: Promise<void>;

  constructor(
    private readonly registry: ToolRegistry,
    configs: readonly (McpServerConfig | PiMcpServerConfig)[] = [],
    private readonly clientFactory: McpClientFactory = connectClient,
  ) {
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
    this.initialization ??= this.connectAll();
    return this.initialization;
  }

  reload(): Promise<void> {
    if (this.disposal) return this.disposal;
    const previous = this.initialization;
    this.initialization = (async () => {
      await previous?.catch(() => undefined);
      await this.disconnectAll();
      if (!this.disposal) await this.connectAll();
    })();
    return this.initialization;
  }

  dispose(): Promise<void> {
    this.disposal ??= (async () => {
      await this.initialization?.catch(() => undefined);
      await this.disconnectAll();
    })();
    return this.disposal;
  }

  private async connectAll(): Promise<void> {
    await Promise.all(this.servers.filter(({ config }) => config.enabled !== false).map(async (server) => {
      const { config } = server;
      try {
        const client = await this.clientFactory(config);
        const connection: Connection = { client };
        server.connection = connection;
        let closed = false;
        connection.unsubscribe = client.onClose?.(() => {
          closed = true;
          connection.unregister?.();
          if (server.connection === connection) {
            server.status = { id: config.id, transport: config.transport, state: 'error', error: 'Connection closed', toolCount: 0 };
          }
        });
        const tools = client.serverCapabilities && !client.serverCapabilities.tools
          ? [] : await client.listTools({ timeoutMs: config.timeoutMs });
        if (closed) throw new Error('Connection closed');
        connection.unregister = this.registry.register(tools.map((tool) => createTool(config, client, tool)));
        server.status = { id: config.id, transport: config.transport, state: 'ready', toolCount: tools.length };
      } catch (error) {
        await this.disconnect(server).catch(() => undefined);
        server.status = { id: config.id, transport: config.transport, state: 'error', error: errorMessage(error), toolCount: 0 };
      }
    }));
  }

  private async disconnect(server: McpServer): Promise<void> {
    const { connection, config } = server;
    if (!connection) return;
    server.connection = undefined;
    connection.unsubscribe?.();
    connection.unregister?.();
    server.status = { id: config.id, transport: config.transport, state: 'error', toolCount: 0, error: 'Not connected' };
    await connection.client.close();
  }

  private async disconnectAll(): Promise<void> {
    const results = await Promise.allSettled(this.servers.map((server) => this.disconnect(server)));
    const failures = results.flatMap((result) => result.status === 'rejected' ? [result.reason] : []);
    if (failures.length) throw new AggregateError(failures, 'Could not close all MCP connections');
  }
}
