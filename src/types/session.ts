import type { AgentMessage } from '@earendil-works/pi-agent-core';
import type { PermissionRule } from './permission.js';

export type AgentMode = 'plan' | 'build';

export interface PlanItem {
  id: string;
  content: string;
  status: 'pending' | 'in_progress' | 'completed' | 'cancelled';
  priority?: 'low' | 'medium' | 'high';
}

export interface SessionHeader {
  type: 'session';
  version: 1;
  id: string;
  createdAt: number;
  mode: AgentMode;
  modelAlias?: string;
  title?: string;
  cwd?: string;
  permissions?: PermissionRule[];
  metadata?: Record<string, unknown>;
}

export type SessionEntryData =
  | { type: 'message'; message: AgentMessage }
  | { type: 'model'; modelAlias: string }
  | { type: 'mode'; mode: AgentMode }
  | { type: 'plan'; items: PlanItem[] }
  | { type: 'compaction'; summary: string; firstKeptEntryId: string | null; tokensBefore: number; tokensAfter?: number }
  | { type: 'metadata'; metadata: Record<string, unknown> };

export type SessionEntry = SessionEntryData & {
  id: string;
  parentId: string | null;
  timestamp: number;
};

export type SessionEntryInput = SessionEntryData;
export type SessionCompactionInput = Extract<SessionEntryInput, { type: 'compaction' }>;

export interface SessionCompactionWindow {
  entries: SessionEntry[];
  leafId: string | null;
  systemBaseline?: Extract<AgentMessage, { role: 'system' }>;
}

/** Describes how much of the input is already in the authoritative journal. */
export class SessionBatchAppendError extends Error {
  constructor(
    readonly committedCount: number | undefined,
    readonly retrySafe: boolean,
    cause: unknown,
  ) {
    const prefix = committedCount === undefined ? 'committed prefix unknown' : `${committedCount} entries committed`;
    super(`Session batch append failed (${prefix}): ${cause instanceof Error ? cause.message : String(cause)}`, { cause });
    this.name = 'SessionBatchAppendError';
  }
}

export interface SessionSnapshot {
  header: SessionHeader;
  leafId: string | null;
}

export interface SessionSummary {
  id: string;
  title?: string;
  mode: AgentMode;
  modelAlias?: string;
  createdAt: number;
  updatedAt: number;
  metadata?: Record<string, unknown>;
}

export interface SessionProjection {
  messages: AgentMessage[];
  /** Prefix whose recorded provider usage predates the latest compaction. */
  compactedMessageCount?: number;
  mode: AgentMode;
  modeChanged?: boolean;
  plan: PlanItem[];
  modelAlias?: string;
}

export type CompactionResult =
  | { status: 'compacted'; summary: string; firstKeptEntryId: string; tokensBefore: number; tokensAfter: number }
  | { status: 'skipped'; reason: 'no-cut-point' | 'aborted' }
  | { status: 'failed'; error: string };

export interface SessionStorageAdapter {
  /** Optional paired hooks for keeping an open handle's cached session resident. */
  retainSession?(sessionId: string): void;
  releaseSession?(sessionId: string): void;
  create(options?: import('./config.js').CreateSessionOptions): Promise<SessionHeader>;
  open(sessionId: string): Promise<SessionSnapshot>;
  list(): Promise<SessionSummary[]>;
  delete(sessionId: string): Promise<void>;
  append(sessionId: string, entry: SessionEntryData): Promise<SessionEntry>;
  /** In input order. Adapters with partial commits report the prefix via SessionBatchAppendError. */
  appendBatch?(sessionId: string, entries: SessionEntryInput[]): Promise<SessionEntry[]>;
  branch(sessionId: string, fromEntryId: string): Promise<void>;
  getBranch(sessionId: string, leafId?: string): Promise<SessionEntry[]>;
  getCompactionWindow?(sessionId: string): Promise<SessionCompactionWindow>;
  /** Atomically append while the expected leaf and retention ancestor are still active and the signal is live. */
  appendCompaction?(
    sessionId: string, expectedLeafId: string, entry: SessionCompactionInput, signal?: AbortSignal,
  ): Promise<SessionEntry | undefined>;
  projectActiveMessages(sessionId: string): Promise<SessionProjection>;
}
