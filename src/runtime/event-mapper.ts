import type { AgentEvent } from '@earendil-works/pi-agent-core';
import type { AssistantMessageEvent } from '@earendil-works/pi-ai';
import type { AgentCoreEvent, AgentCoreEventData, TokenUsage } from '../types/events.js';
import type { AgentMode } from '../types/session.js';

export interface EventMappingContext {
  runId: string;
  mode: AgentMode;
  modelId: string;
  signal?: AbortSignal;
  getToolLabel?: (name: string) => string | undefined;
}

function mapBlock(event: AssistantMessageEvent): AgentCoreEventData | undefined {
  switch (event.type) {
    case 'thinking_start':
    case 'text_start':
      return { type: event.type, blockIndex: event.contentIndex };
    case 'thinking_delta':
    case 'text_delta':
      return { type: event.type, blockIndex: event.contentIndex, delta: event.delta };
    case 'thinking_end':
    case 'text_end':
      return { type: event.type, blockIndex: event.contentIndex, content: event.content };
    default:
      return undefined;
  }
}

function toolText(result: { content?: { type: string; text?: string }[] } | undefined): string {
  return result?.content?.filter(({ type }) => type === 'text').map(({ text }) => text).join('\n') ?? '';
}

export class EventMapper {
  private seq = 0;

  constructor(private readonly sessionId: string) {}

  create(runId: string, data: AgentCoreEventData): AgentCoreEvent {
    return { ...data, sessionId: this.sessionId, runId, seq: ++this.seq, timestamp: Date.now() };
  }

  map(event: AgentEvent, context: EventMappingContext): AgentCoreEvent | undefined {
    const data = this.mapData(event, context);
    return data && this.create(context.runId, data);
  }

  private mapData(event: AgentEvent, context: EventMappingContext): AgentCoreEventData | undefined {
    switch (event.type) {
      case 'turn_start':
        return { type: 'turn_start', mode: context.mode, modelId: context.modelId };
      case 'turn_end': {
        const assistant = event.message.role === 'assistant' ? event.message : undefined;
        const state = context.signal?.aborted || assistant?.stopReason === 'aborted' ? 'cancelled' :
          assistant?.stopReason === 'error' ? 'failed' : 'completed';
        return { type: 'turn_end', state, usage: assistant?.usage, error: assistant?.errorMessage };
      }
      case 'message_update':
        return mapBlock(event.assistantMessageEvent);
      case 'tool_execution_start':
        return { type: 'tool_start', toolCallId: event.toolCallId, toolName: event.toolName,
          label: context.getToolLabel?.(event.toolName) ?? event.toolName, args: event.args };
      case 'tool_execution_update':
        return { type: 'tool_update', toolCallId: event.toolCallId, toolName: event.toolName,
          partial: toolText(event.partialResult), details: event.partialResult?.details };
      case 'tool_execution_end': {
        const details = event.result?.details;
        const output = details?.truncated === true && typeof details.outputPath === 'string'
          ? { truncated: true, outputPath: details.outputPath } : {};
        return { type: 'tool_end', toolCallId: event.toolCallId, toolName: event.toolName, isError: event.isError,
          summary: toolText(event.result), durationMs: event.durationMs, ...output };
      }
      default:
        return undefined;
    }
  }
}

export function sumTokenUsage(total: TokenUsage | undefined, next: TokenUsage | undefined): TokenUsage | undefined {
  if (!next) return total;
  if (!total) return { ...next, cost: { ...next.cost } };
  return {
    input: total.input + next.input,
    output: total.output + next.output,
    cacheRead: total.cacheRead + next.cacheRead,
    cacheWrite: total.cacheWrite + next.cacheWrite,
    ...(total.cacheWrite1h !== undefined || next.cacheWrite1h !== undefined
      ? { cacheWrite1h: (total.cacheWrite1h ?? 0) + (next.cacheWrite1h ?? 0) } : {}),
    ...(total.reasoning !== undefined || next.reasoning !== undefined
      ? { reasoning: (total.reasoning ?? 0) + (next.reasoning ?? 0) } : {}),
    totalTokens: total.totalTokens + next.totalTokens,
    cost: {
      input: total.cost.input + next.cost.input,
      output: total.cost.output + next.cost.output,
      cacheRead: total.cost.cacheRead + next.cost.cacheRead,
      cacheWrite: total.cost.cacheWrite + next.cost.cacheWrite,
      total: total.cost.total + next.cost.total,
    },
  };
}
