export type PermissionEffect = 'allow' | 'ask' | 'deny';

export interface PermissionRule {
  action: string;
  resource: string;
  effect: PermissionEffect;
}

export interface ApprovalRequest {
  id: string;
  sessionId: string;
  runId: string;
  toolCallId: string;
  toolName: string;
  action: string;
  resources: string[];
  /** Validated tool arguments and creation time on Phase 2 approval requests. */
  args?: unknown;
  createdAt?: number;
  /** Timestamp kept for compatibility with Phase 1 consumers. */
  timestamp: number;
}

/** Approval payload emitted by the harness with validated arguments and creation time. */
export interface PendingApprovalRequest extends ApprovalRequest {
  args: unknown;
  createdAt: number;
}

export type ApprovalDecision =
  | { kind: 'once' }
  | { kind: 'always'; savePattern?: string }
  | { kind: 'reject'; feedback?: string };
