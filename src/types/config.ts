import type { AgentOptions, StreamFn, ThinkingLevel } from '@earendil-works/pi-agent-core';
import type { ImageContent } from '@earendil-works/pi-ai';
import type { McpServerConfig as PiMcpServerConfig } from '@earendil-works/pi-mcp';
import type { McpClientFactory } from '../harness/mcp.js';
import type { PermissionRule } from './permission.js';
import type { AgentMode, SessionStorageAdapter } from './session.js';
import type { CoreToolDefinition } from './tool.js';

export type ModelProtocol =
  | 'openai-completions'
  | 'openai-responses'
  | 'anthropic-messages'
  | 'google-generative-ai';

export interface ModelProfile {
  alias: string;
  modelId: string;
  providerId: string;
  protocol: ModelProtocol;
  baseUrl: string;
  apiKey?: string | (() => Promise<string | undefined> | string | undefined);
  headers?: Record<string, string>;
  contextWindow: number;
  maxTokens: number;
  reasoning?: boolean;
  defaultThinkingLevel?: 'off' | 'minimal' | 'low' | 'medium' | 'high';
  compat?: Record<string, unknown>;
}

export interface ModelRouterConfig {
  defaultAlias: string;
  profiles: ModelProfile[];
  streamFnOverride?: StreamFn;
}

export type AgentRuntimeHooks = Pick<
  AgentOptions, 'transformContext' | 'prepareRequest' | 'beforeToolCall' | 'afterToolCall'
>;

export interface SubagentProfile {
  name: string;
  description: string;
  modelAlias?: string;
  systemPrompt?: string;
  readOnly?: boolean;
  maxSteps?: number;
  allowedTools?: string[];
  deniedTools?: string[];
}

export interface PromptTemplates {
  /** Base persona for the embedded assistant. */
  basePersona: string;
  /** Runtime updates, background results and tool coordination. */
  harnessProtocol: string;
  /** Verification, editing and communication conventions. */
  workingDiscipline: string;
  /** Reminder shown in Plan mode, optionally wrapped in <system-update>. */
  planModeEnter: string;
  /** Reminder after switching back to Build mode. */
  buildModeEnter: string;
  todoWriteDescription: string;
  subagentDescription: string;
  questionDescription: string;
  compactionSummary: string;
  maxStepsSummary: string;
}

export interface CompactionOptions {
  enabled?: boolean;
  reserveTokens?: number;
  keepRecentTokens?: number;
  modelAlias?: string;
  /** Compute a candidate summary before the threshold is reached. Defaults to false. */
  precompute?: boolean;
  /** Begin precomputing within this many tokens of the threshold. Defaults to 20,000. */
  precomputeLeadTokens?: number;
}

export interface McpServerConfig {
  id: string;
  enabled?: boolean;
  transport: 'stdio' | 'http';
  command?: string;
  args?: string[];
  env?: Record<string, string>;
  cwd?: string;
  url?: string;
  headers?: Record<string, string>;
  timeoutMs?: number;
}

export interface McpServerStatus {
  id: string;
  transport: McpServerConfig['transport'];
  state: 'ready' | 'error' | 'disabled';
  toolCount: number;
  error?: string;
}

export interface TruncationOptions {
  maxLines?: number;
  maxBytes?: number;
}

export interface AgentRuntimeConfig {
  dataDir: string;
  models: ModelRouterConfig;
  sessionStorage?: SessionStorageAdapter;
  tools?: CoreToolDefinition[];
  baseSystemPrompt?: string;
  /** Replace individual prompt segments. baseSystemPrompt remains the persona fallback. */
  prompts?: Partial<PromptTemplates>;
  /** Show the built-in question tool. Defaults to true. */
  enableQuestionTool?: boolean;
  hooks?: AgentRuntimeHooks;
  /** Rules for resources that match. Unmatched resources use defaultPermissionEffect. */
  permissions?: PermissionRule[];
  defaultPermissionEffect?: 'allow' | 'ask';
  globalRules?: string[];
  skillDirs?: string[];
  mcpServers?: Array<McpServerConfig | PiMcpServerConfig>;
  mcpClientFactory?: McpClientFactory;
  subagents?: Record<string, Partial<SubagentProfile>>;
  maxSubagentDepth?: number;
  compaction?: CompactionOptions;
  truncation?: TruncationOptions;
}

export interface CreateSessionOptions {
  id?: string;
  title?: string;
  cwd?: string;
  /** Overrides matching runtime rules. */
  permissions?: PermissionRule[];
  mode?: AgentMode;
  modelAlias?: string;
  metadata?: Record<string, unknown>;
}

export interface PromptOptions {
  modelAlias?: string;
  thinkingLevel?: ThinkingLevel;
  images?: ImageContent[];
}

export interface RuntimeCatalog {
  models: Array<Pick<ModelProfile, 'alias' | 'modelId' | 'providerId' | 'protocol'>>;
  tools: Array<Pick<CoreToolDefinition, 'name' | 'label' | 'description'>>;
  rules?: string[];
  skills?: Array<{ id?: string; name: string; description?: string }>;
  mcpServers?: McpServerStatus[];
}
