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
  create(options?: import('./config.js').CreateSessionOptions): Promise<SessionHeader>;
  open(sessionId: string): Promise<SessionSnapshot>;
  list(): Promise<SessionSummary[]>;
  delete(sessionId: string): Promise<void>;
  append(sessionId: string, entry: SessionEntryData): Promise<SessionEntry>;
  branch(sessionId: string, fromEntryId: string): Promise<void>;
  getBranch(sessionId: string, leafId?: string): Promise<SessionEntry[]>;
  projectActiveMessages(sessionId: string): Promise<SessionProjection>;
}
