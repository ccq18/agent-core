import { AsyncLocalStorage } from 'node:async_hooks';
import { randomUUID } from 'node:crypto';
import { Semaphore } from '../runtime/concurrency.js';
import type { AgentSessionHandle } from '../runtime/session-handle.js';
import type { ConcurrencyOptions, CreateSessionOptions, SubagentProfile } from '../types/config.js';
import type { AgentCoreEventData, TurnState, TurnSummary } from '../types/events.js';
import type { PermissionRule } from '../types/permission.js';
import type { AgentMode, SessionStorageAdapter } from '../types/session.js';
import type { CoreToolDefinition, ToolExecutionContext } from '../types/tool.js';
import { DEFAULT_PROMPT_TEMPLATES } from './prompt.js';

const DEFAULT_PROFILES: Record<string, SubagentProfile> = {
  explore: {
    name: 'explore', description: 'Read-only fact gathering and code analysis', readOnly: true, maxSteps: 12,
    systemPrompt: 'Gather facts and analyze the code using read-only tools. Return a concise, evidence-based summary.',
    deniedTools: ['subagent'],
  },
  general: {
    name: 'general', description: 'General task execution', readOnly: false, maxSteps: 20,
    systemPrompt: 'Carry out the delegated task using the supplied context and report the outcome concisely.',
    deniedTools: ['subagent'],
  },
};

type SubagentEvent = Extract<AgentCoreEventData, { type: 'subagent_started' | 'subagent_completed' }>;

export interface SubagentParent {
  id: string;
  mode: AgentMode;
  cwd?: string;
  depth: number;
  readOnly: boolean;
  modelAlias?: string;
  permissions?: PermissionRule[];
  followUp(input: string): void;
  emit(runId: string, event: SubagentEvent): void;
}

interface SubagentSessions {
  create(options: CreateSessionOptions): Promise<AgentSessionHandle>;
  /** Validate the header captured by the usable handle before returning it. */
  open(id: string, validate: (metadata?: Record<string, unknown>) => void): Promise<AgentSessionHandle>;
  release?(handle: AgentSessionHandle, drainBackground: boolean): Promise<void>;
  getParent(id: string): Promise<SubagentParent | undefined>;
}

interface ActiveChild {
  controller: AbortController;
  result: Promise<ChildResult>;
}

interface ChildResult {
  state: TurnState;
  summary: string;
}

export interface SubagentTurnLease {
  readonly sessionId: string;
  readonly lease: symbol;
  readonly ancestry: ReadonlyMap<string, symbol>;
  release(): void;
}

const textResult = (text: string, isError = false) => ({ content: [{ type: 'text' as const, text }], isError });

export class SubagentManager {
  private readonly profiles: ReadonlyMap<string, SubagentProfile>;
  private readonly active = new Map<string, Map<string, ActiveChild>>();
  private readonly maxDepth: number;
  private readonly runtimeSlots?: Semaphore;
  private readonly parentLimit?: number;
  private readonly parentSlots = new Map<string, Semaphore>();
  private readonly ancestry = new AsyncLocalStorage<ReadonlyMap<string, symbol>>();
  private readonly leases = new Set<symbol>();
  private readonly waitingParents = new Map<symbol, ReadonlyMap<string, symbol>>();

  constructor(
    // Kept for callers that construct the manager directly.
    _storage: SessionStorageAdapter,
    private readonly sessions: SubagentSessions,
    profiles: Record<string, Partial<SubagentProfile>> = {},
    maxDepth = 1,
    description = DEFAULT_PROMPT_TEMPLATES.subagentDescription,
    limits: ConcurrencyOptions = {},
  ) {
    const configured = new Map(Object.entries(DEFAULT_PROFILES));
    for (const [key, profile] of Object.entries(profiles)) {
      const name = profile.name ?? key;
      const settings = { description: name, ...configured.get(key), ...profile };
      configured.set(name, { ...settings, name, maxSteps: settings.maxSteps ?? (settings.readOnly ? 12 : 20) });
    }
    this.profiles = configured;
    this.maxDepth = maxDepth;
    this.tool.description = description;
    if (limits.maxSubagentsPerRuntime !== undefined) this.runtimeSlots = new Semaphore(limits.maxSubagentsPerRuntime);
    this.parentLimit = limits.maxSubagentsPerParent;
  }

  getProfile(name: string): SubagentProfile | undefined {
    return this.profiles.get(name);
  }

  allowsTool(tool: CoreToolDefinition, depth: number, profile?: SubagentProfile): boolean {
    if (tool.name === 'subagent' && depth >= this.maxDepth) return false;
    if (profile?.readOnly && tool.readOnly !== true) return false;
    return (!profile?.allowedTools || profile.allowedTools.includes(tool.name)) &&
      !profile?.deniedTools?.includes(tool.name);
  }

  planBlockReason(args: unknown): string | undefined {
    const name = String((args as { agent?: unknown })?.agent);
    if (this.getProfile(name)?.readOnly !== true) return `Plan mode blocked subagent "${name}". This profile is not read-only.`;
    return undefined;
  }

  readonly tool: CoreToolDefinition = {
    name: 'subagent',
    label: 'Delegate subagent',
    description: DEFAULT_PROMPT_TEMPLATES.subagentDescription,
    permissionAction: 'subagent',
    resolveResources: (params) => [String(params.agent)],
    readOnly: true,
    parameters: {
      type: 'object',
      properties: {
        agent: { type: 'string' }, description: { type: 'string' }, prompt: { type: 'string' },
        background: { type: 'boolean' }, modelAlias: { type: 'string' }, sessionId: { type: 'string' },
      },
      required: ['agent', 'description', 'prompt'],
    },
    execute: (params, context) => this.delegate(params, context),
  };

  private async delegate(params: Record<string, unknown>, context: ToolExecutionContext) {
    const name = String(params.agent);
    const profile = this.getProfile(name);
    if (!profile) return textResult(`Unknown subagent profile "${name}"`, true);
    const parent = await this.sessions.getParent(context.sessionId);
    if (!parent) return textResult(`Unknown parent session "${context.sessionId}"`, true);
    if (parent.depth >= this.maxDepth) return textResult('Maximum subagent depth reached', true);
    if (parent.mode === 'plan' || parent.readOnly) {
      const reason = this.planBlockReason(params);
      if (reason) return textResult(reason, true);
    }
    context.signal.throwIfAborted();

    const background = params.background === true;
    const chain = this.ancestry.getStore();
    const ancestors = !background && chain?.has(parent.id) ? chain : undefined;
    if (!background && this.runtimeSlotsBlocked(ancestors)) {
      const code = 'SUBAGENT_CONCURRENCY_LIMIT';
      return {
        ...textResult(`${code}: All runtime subagent slots are held by sessions blocked on delegation.`, true),
        structuredContent: { code },
      };
    }

    const resume = typeof params.sessionId === 'string' ? params.sessionId : undefined;
    const id = resume ?? randomUUID();
    const children = this.active.get(parent.id) ?? new Map<string, ActiveChild>();
    if (children.has(id)) return textResult(`Subagent session "${id}" is already running`, true);
    this.active.set(parent.id, children);

    const controller = new AbortController();
    const description = String(params.description);
    const modelAlias = typeof params.modelAlias === 'string' ? params.modelAlias : undefined;
    const start = () => parent.emit(context.runId, { type: 'subagent_started', subagentId: id, profile: name, description, background });
    const run = (currentParent: SubagentParent, releaseForDrain?: () => void) =>
      this.runChild(currentParent, profile, id, resume !== undefined, description, String(params.prompt), modelAlias,
        controller.signal, releaseForDrain);
    const limited = this.runtimeSlots !== undefined || this.parentLimit !== undefined;
    const abort = () => controller.abort(context.signal.reason);
    const settle = async (execute: () => Promise<ChildResult>): Promise<ChildResult> => {
      let result: ChildResult;
      try {
        result = await execute();
      } catch (error) {
        result = {
          state: controller.signal.aborted ? 'cancelled' : 'failed',
          summary: error instanceof Error ? error.message : String(error),
        };
      }
      context.signal.removeEventListener('abort', abort);
      children.delete(id);
      if (!children.size) this.active.delete(parent.id);
      parent.emit(context.runId, {
        type: 'subagent_completed', subagentId: id, state: result.state, summary: result.summary,
      });
      if (background && !controller.signal.aborted) {
        parent.followUp(`<background-result subagentId="${id}" agent="${name}" state="${result.state}">${result.summary}</background-result>`);
      }
      return result;
    };
    const admitted = async (releaseForDrain?: () => void) => {
      const currentParent = await this.sessions.getParent(parent.id);
      if (!currentParent) throw new Error(`Unknown parent session "${parent.id}"`);
      if (currentParent.mode === 'plan' || currentParent.readOnly) {
        const reason = this.planBlockReason(params);
        if (reason) throw new Error(reason);
      }
      controller.signal.throwIfAborted();
      start();
      return run(currentParent, releaseForDrain);
    };
    const child: ActiveChild = {
      controller,
      result: limited ? this.runWithSlots(parent.id, id, ancestors, controller.signal, admitted, settle) : settle(() => run(parent)),
    };
    children.set(id, child);
    context.signal.addEventListener('abort', abort, { once: true });
    if (context.signal.aborted) abort();
    if (!limited) start();
    if (background) return textResult(JSON.stringify({ subagentId: id, status: 'running' }));
    const result = await child.result;
    return textResult(result.summary, result.state !== 'completed');
  }

  private runtimeSlotsBlocked(ancestors: ReadonlyMap<string, symbol> | undefined): boolean {
    if (!this.runtimeSlots || this.leases.size !== this.runtimeSlots.capacity) return false;
    const waitingLeases = new Set(ancestors?.values());
    for (const waiting of this.waitingParents.values()) {
      for (const lease of waiting.values()) waitingLeases.add(lease);
    }
    return [...this.leases].every((lease) => waitingLeases.has(lease));
  }

  private async runWithSlots(
    parentId: string, id: string, ancestors: ReadonlyMap<string, symbol> | undefined, signal: AbortSignal,
    run: (releaseForDrain?: () => void) => Promise<ChildResult>, settle: (execute: () => Promise<ChildResult>) => Promise<ChildResult>,
  ): Promise<ChildResult> {
    let parentSlots: Semaphore | undefined;
    let releaseParent: (() => void) | undefined;
    let releaseRuntime: (() => void) | undefined;
    const lease = Symbol();
    const releaseSlot = () => {
      this.leases.delete(lease);
      releaseRuntime?.();
    };
    try {
      return await settle(async () => {
        // Foreground ancestors cannot finish while this delegation waits for admission.
        if (ancestors && this.runtimeSlots) this.waitingParents.set(lease, ancestors);
        // Wait for the parent quota first so one parent's backlog cannot reserve runtime slots.
        if (this.parentLimit !== undefined) {
          parentSlots = this.parentSlots.get(parentId) ?? new Semaphore(this.parentLimit);
          this.parentSlots.set(parentId, parentSlots);
          releaseParent = await parentSlots.acquire(signal);
        }
        if (!this.runtimeSlots) return run();
        releaseRuntime = await this.runtimeSlots.acquire(signal);
        this.waitingParents.delete(lease);
        signal.throwIfAborted();
        this.leases.add(lease);
        return this.ancestry.run(new Map(ancestors).set(id, lease), () => run(() => {
          if (ancestors) this.waitingParents.set(lease, ancestors);
          releaseSlot();
        }));
      });
    } finally {
      this.waitingParents.delete(lease);
      releaseSlot();
      releaseParent?.();
      if (parentSlots?.idle) this.parentSlots.delete(parentId);
    }
  }

  private async runChild(
    parent: SubagentParent, profile: SubagentProfile, id: string, resume: boolean,
    description: string, prompt: string, modelAlias: string | undefined, signal: AbortSignal, releaseForDrain?: () => void,
  ) {
    const depth = parent.depth + 1;
    const metadata = { parentSessionId: parent.id, subagentProfile: profile.name, depth };
    let validated = false;
    const validate = (saved?: Record<string, unknown>) => {
      if (saved?.parentSessionId !== parent.id || saved.subagentProfile !== profile.name || saved.depth !== depth) {
        throw new Error(`Session "${id}" does not belong to subagent "${profile.name}" of this parent`);
      }
      validated = true;
    };
    let session: AgentSessionHandle;
    signal.throwIfAborted();
    if (resume) {
      session = await this.sessions.open(id, validate);
    } else {
      session = await this.sessions.create({
        id, title: description, cwd: parent.cwd, permissions: parent.permissions,
        mode: profile.readOnly ? 'plan' : 'build',
        modelAlias: modelAlias ?? profile.modelAlias ?? parent.modelAlias, metadata,
      });
    }

    const abort = () => session.abort();
    let turn: TurnSummary | undefined;
    try {
      if (resume && !validated) throw new Error(`Session "${id}" was opened without validating subagent ancestry`);
      signal.throwIfAborted();
      signal.addEventListener('abort', abort, { once: true });
      turn = await session.prompt(prompt, { modelAlias });
    } finally {
      try {
        const drainBackground = turn !== undefined && turn.state !== 'cancelled' && !signal.aborted;
        // The completed prompt's slot must be available to its queued background descendants.
        if (drainBackground && this.hasActive(id)) releaseForDrain?.();
        await this.sessions.release?.(session, drainBackground);
      } finally {
        signal.removeEventListener('abort', abort);
      }
    }
    const state = signal.aborted ? 'cancelled' : turn.state;
    return { state, summary: state === 'completed' ? turn.text : turn.error ?? turn.text };
  }

  async acquireTurnLease(
    sessionId: string, depth: number, signal: AbortSignal,
  ): Promise<SubagentTurnLease | undefined> {
    if (!this.runtimeSlots || depth <= 0) return undefined;
    const chain = this.ancestry.getStore();
    if (chain?.has(sessionId)) return undefined;
    const releaseSlot = await this.runtimeSlots.acquire(signal);
    const lease = Symbol();
    let released = false;
    const release = () => {
      if (released) return;
      released = true;
      this.leases.delete(lease);
      releaseSlot();
    };
    if (signal.aborted) {
      release();
      signal.throwIfAborted();
    }
    this.leases.add(lease);
    return { sessionId, lease, ancestry: new Map(chain).set(sessionId, lease), release };
  }

  runWithTurnLease<T>(lease: SubagentTurnLease, execute: () => Promise<T>): Promise<T> {
    return this.ancestry.run(lease.ancestry, execute);
  }

  abort(parentId: string): void {
    for (const child of this.active.get(parentId)?.values() ?? []) child.controller.abort();
  }

  hasActive(parentId: string): boolean {
    return this.active.has(parentId);
  }

  async wait(parentId: string): Promise<void> {
    await Promise.all([...(this.active.get(parentId)?.values() ?? [])].map(({ result }) => result));
  }
}
