import type { ImageContent, JsonValue, TextContent, Usage } from '@earendil-works/pi-ai';

export interface CoreToolResult {
  content: (TextContent | ImageContent)[];
  details?: unknown;
  structuredContent?: JsonValue;
  usage?: Usage;
  isError?: boolean;
  terminate?: boolean;
}

export interface TruncatedOutputDetails {
  truncated: true;
  outputPath: string;
  totalLines?: number;
  totalBytes?: number;
  /** Number of complete lines retained in the preview. */
  keptLines?: number;
  /** Includes any line that is only partially visible. */
  omittedLines?: number;
  strategy?: 'head-tail' | 'head';
}

export interface ToolExecutionContext {
  sessionId: string;
  runId: string;
  toolCallId: string;
  signal: AbortSignal;
  onUpdate(result: CoreToolResult): void;
}

export interface CoreToolDefinition<TParams = Record<string, unknown>> {
  name: string;
  label: string;
  description: string;
  parameters: Record<string, unknown>;
  permissionAction?: string;
  resolveResources?(params: TParams): string[];
  readOnly?: boolean;
  executionMode?: 'parallel' | 'sequential';
  execute(params: TParams, context: ToolExecutionContext): Promise<CoreToolResult>;
}
