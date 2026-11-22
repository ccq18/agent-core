import { randomUUID } from 'node:crypto';
import type { AgentSessionHandle } from '../runtime/session-handle.js';
import type { CreateSessionOptions, SubagentProfile } from '../types/config.js';
import type { AgentCoreEventData, TurnState } from '../types/events.js';
import type { PermissionRule } from '../types/permission.js';
import type { AgentMode, SessionStorageAdapter } from '../types/session.js';
import type { CoreToolDefinition, ToolExecutionContext } from '../types/tool.js';

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
  open(id: string): Promise<AgentSessionHandle>;
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

const textResult = (text: string, isError = false) => ({ content: [{ type: 'text' as const, text }], isError });

export class SubagentManager {
  private readonly profiles: ReadonlyMap<string, SubagentProfile>;
  private readonly active = new Map<string, Map<string, ActiveChild>>();
  private readonly maxDepth: number;

  constructor(
    private readonly storage: SessionStorageAdapter,
    private readonly sessions: SubagentSessions,
    profiles: Record<string, Partial<SubagentProfile>> = {},
    maxDepth = 1,
  ) {
    const configured = new Map(Object.entries(DEFAULT_PROFILES));
    for (const [key, profile] of Object.entries(profiles)) {
      const name = profile.name ?? key;
      const settings = { description: name, ...configured.get(key), ...profile };
      configured.set(name, { ...settings, name, maxSteps: settings.maxSteps ?? (settings.readOnly ? 12 : 20) });
    }
    this.profiles = configured;
    this.maxDepth = maxDepth;
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
    description: 'Delegate an isolated task to a subagent. Built-in profiles: explore (read-only investigation) and general (task execution).',
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

    const resume = typeof params.sessionId === 'string' ? params.sessionId : undefined;
    const id = resume ?? randomUUID();
    const children = this.active.get(parent.id) ?? new Map<string, ActiveChild>();
    if (children.has(id)) return textResult(`Subagent session "${id}" is already running`, true);
    this.active.set(parent.id, children);

    const controller = new AbortController();
    const description = String(params.description);
    const background = params.background === true;
    const modelAlias = typeof params.modelAlias === 'string' ? params.modelAlias : undefined;
    const child: ActiveChild = {
      controller,
      result: this.runChild(parent, profile, id, resume !== undefined, description, String(params.prompt), modelAlias, controller.signal)
        .catch((error: unknown): ChildResult => ({
          state: controller.signal.aborted ? 'cancelled' : 'failed',
          summary: error instanceof Error ? error.message : String(error),
        }))
        .then((result) => {
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
        }),
    };
    children.set(id, child);
    const abort = () => controller.abort();
    context.signal.addEventListener('abort', abort, { once: true });
    parent.emit(context.runId, { type: 'subagent_started', subagentId: id, profile: name, description, background });
    if (background) return textResult(JSON.stringify({ subagentId: id, status: 'running' }));
    const result = await child.result;
    return textResult(result.summary, result.state !== 'completed');
  }

  private async runChild(
    parent: SubagentParent, profile: SubagentProfile, id: string, resume: boolean,
    description: string, prompt: string, modelAlias: string | undefined, signal: AbortSignal,
  ) {
    const depth = parent.depth + 1;
    const metadata = { parentSessionId: parent.id, subagentProfile: profile.name, depth };
    let session: AgentSessionHandle;
    if (resume) {
      const { header } = await this.storage.open(id);
      const saved = header.metadata;
      if (saved?.parentSessionId !== parent.id || saved.subagentProfile !== profile.name || saved.depth !== depth) {
        throw new Error(`Session "${id}" does not belong to subagent "${profile.name}" of this parent`);
      }
      signal.throwIfAborted();
      session = await this.sessions.open(id);
    } else {
      session = await this.sessions.create({
        id, title: description, cwd: parent.cwd, permissions: parent.permissions,
        mode: profile.readOnly ? 'plan' : 'build',
        modelAlias: modelAlias ?? profile.modelAlias ?? parent.modelAlias, metadata,
      });
    }

    signal.throwIfAborted();
    const abort = () => session.abort();
    signal.addEventListener('abort', abort, { once: true });
    try {
      const { state, text, error } = await session.prompt(prompt, { modelAlias });
      return { state, summary: state === 'completed' ? text : error ?? text };
    } finally {
      signal.removeEventListener('abort', abort);
    }
  }

  abort(parentId: string): void {
    for (const child of this.active.get(parentId)?.values() ?? []) child.controller.abort();
  }

  async wait(parentId: string): Promise<void> {
    await Promise.all([...(this.active.get(parentId)?.values() ?? [])].map(({ result }) => result));
  }
}
