import type { ApprovalDecision, ApprovalRequest } from './permission.js';
import type { PendingQuestionRequest, QuestionAnswer } from './question.js';
import type { AgentMode, PlanItem } from './session.js';

export interface TokenUsage {
  input: number;
  output: number;
  cacheRead: number;
  cacheWrite: number;
  cacheWrite1h?: number;
  reasoning?: number;
  totalTokens: number;
  cost: {
    input: number;
    output: number;
    cacheRead: number;
    cacheWrite: number;
    total: number;
  };
}

export type TurnState = 'completed' | 'cancelled' | 'failed';

export interface TurnSummary {
  sessionId: string;
  runId: string;
  state: TurnState;
  text: string;
  usage?: TokenUsage;
  error?: string;
}

export type AgentCoreEventData =
  | { type: 'turn_start'; mode: AgentMode; modelId: string }
  | { type: 'thinking_start'; blockIndex: number }
  | { type: 'thinking_delta'; blockIndex: number; delta: string }
  | { type: 'thinking_end'; blockIndex: number; content: string }
  | { type: 'text_start'; blockIndex: number }
  | { type: 'text_delta'; blockIndex: number; delta: string }
  | { type: 'text_end'; blockIndex: number; content: string }
  | { type: 'plan_update'; items: PlanItem[] }
  | { type: 'mode_changed'; from: AgentMode; to: AgentMode }
  | { type: 'tool_start'; toolCallId: string; toolName: string; label: string; args: unknown }
  | { type: 'tool_update'; toolCallId: string; toolName: string; partial: string; details?: unknown }
  | { type: 'tool_end'; toolCallId: string; toolName: string; isError: boolean; summary: string; truncated?: boolean; outputPath?: string; durationMs?: number }
  | { type: 'approval_requested'; request: ApprovalRequest }
  | { type: 'approval_resolved'; requestId: string; decision: ApprovalDecision }
  | { type: 'question_requested'; request: PendingQuestionRequest }
  | { type: 'question_resolved'; requestId: string; cancelled: boolean; answers?: QuestionAnswer }
  | { type: 'subagent_started'; subagentId: string; profile: string; description: string; background: boolean }
  | { type: 'subagent_completed'; subagentId: string; state: TurnState; summary: string }
  | { type: 'compaction_start'; trigger: 'auto' | 'overflow' | 'manual'; tokensBefore: number }
  | { type: 'compaction_end'; tokensBefore: number; tokensAfter: number; summary: string }
  | { type: 'turn_end'; state: TurnState; usage?: TokenUsage; error?: string };

export type AgentCoreEvent = {
  sessionId: string;
  runId: string;
  seq: number;
  timestamp: number;
} & AgentCoreEventData;
