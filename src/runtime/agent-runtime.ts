import { resolve } from 'node:path';
import { McpBridge } from '../harness/mcp.js';
import { PermissionEngine } from '../harness/permission.js';
import { createPlanTools } from '../harness/plan.js';
import { PromptAssembler, resolvePromptTemplates } from '../harness/prompt.js';
import { QuestionManager } from '../harness/question.js';
import { RulesLoader } from '../harness/rules.js';
import { SkillsRegistry } from '../harness/skills.js';
import { SubagentManager } from '../harness/subagent.js';
import { ToolRegistry } from '../harness/tool-registry.js';
import { OutputTruncator } from '../harness/truncation.js';
import { ModelRouter } from '../model/router.js';
import { CompactionEngine } from '../session/compaction.js';
import { JsonlTreeSessionStore } from '../session/store.js';
import type { AgentRuntimeConfig, CreateSessionOptions, ModelRouterConfig, PromptTemplates, RuntimeCatalog } from '../types/config.js';
import type { SessionStorageAdapter, SessionSummary } from '../types/session.js';
import type { CoreToolDefinition } from '../types/tool.js';
import { SessionHandle, type AgentSessionHandle } from './session-handle.js';

export interface AgentRuntime {
  createSession(options?: CreateSessionOptions): Promise<AgentSessionHandle>;
  openSession(sessionId: string): Promise<AgentSessionHandle>;
  listSessions(): Promise<SessionSummary[]>;
  deleteSession(sessionId: string): Promise<void>;
  registerTools(tools: CoreToolDefinition[]): () => void;
  updateModelConfig(config: ModelRouterConfig): void;
  reloadMcp(): Promise<void>;
  getCatalog(): RuntimeCatalog;
  dispose(): Promise<void>;
}

class Runtime implements AgentRuntime {
  private readonly storage: SessionStorageAdapter;
  private readonly router: ModelRouter;
  private readonly sessions = new Map<string, Promise<SessionHandle>>();
  private readonly creating = new Set<Promise<SessionHandle>>();
  private readonly deleting = new Map<string, Promise<void>>();
  private readonly registry: ToolRegistry;
  private readonly permissions: PermissionEngine;
  private readonly rules: RulesLoader;
  private readonly loadedRules: Set<string>;
  private readonly skills: SkillsRegistry;
  private readonly prompt = new PromptAssembler();
  private readonly prompts: PromptTemplates;
  private readonly truncator: OutputTruncator;
  private readonly compaction: CompactionEngine;
  private readonly subagents: SubagentManager;
  private readonly questions?: QuestionManager;
  private readonly mcp: McpBridge;
  private disposal?: Promise<void>;

  constructor(private readonly config: AgentRuntimeConfig) {
    this.prompts = resolvePromptTemplates(config.prompts, config.baseSystemPrompt);
    this.storage = config.sessionStorage ?? new JsonlTreeSessionStore(config.dataDir);
    this.router = new ModelRouter(config.models);
    this.permissions = new PermissionEngine(config.dataDir, config.defaultPermissionEffect);
    this.rules = new RulesLoader(config.globalRules);
    this.loadedRules = new Set(this.rules.load());
    this.skills = new SkillsRegistry(config.skillDirs);
    this.truncator = new OutputTruncator(config.dataDir, config.truncation);
    this.compaction = new CompactionEngine(this.storage, this.router, config.compaction, this.prompts);
    this.subagents = new SubagentManager(this.storage, {
      create: (options) => this.createSession(options),
      open: (id) => this.openSession(id),
      getParent: async (id) => (await this.sessions.get(id))?.getSubagentParent(),
    }, config.subagents, config.maxSubagentDepth, this.prompts.subagentDescription);
    if (config.enableQuestionTool !== false) {
      this.questions = new QuestionManager(({ sessionId, runId }, event) => {
        void this.sessions.get(sessionId)?.then((session) => session.emitRunEvent(runId, event));
      }, this.prompts.questionDescription);
    }
    const todoTools = createPlanTools(async (sessionId) => {
      const session = await this.sessions.get(sessionId);
      if (!session) throw new Error(`Unknown session "${sessionId}"`);
      return session.planStore;
    }, this.prompts.todoWriteDescription);
    const builtIns = [...todoTools, this.skills.createTool(), this.subagents.tool];
    if (this.questions) builtIns.push(this.questions.tool);
    this.registry = new ToolRegistry(builtIns);
    this.mcp = new McpBridge(this.registry, config.mcpServers, config.mcpClientFactory);
    this.registerTools(config.tools ?? []);
  }

  async createSession(options: CreateSessionOptions = {}): Promise<AgentSessionHandle> {
    this.assertOpen();
    this.router.resolve(options.modelAlias);
    const cwd = options.cwd ?? (typeof options.metadata?.cwd === 'string' ? options.metadata.cwd : undefined);
    const settings = cwd === undefined ? options : { ...options, cwd: resolve(cwd) };
    const pending = this.storage.create(settings).then((header) => this.loadSession(header.id));
    this.creating.add(pending);
    try {
      return await pending;
    } finally {
      this.creating.delete(pending);
    }
  }

  async openSession(id: string): Promise<AgentSessionHandle> {
    const deletion = this.deleting.get(id);
    if (deletion) await deletion;
    this.assertOpen();
    return this.loadSession(id);
  }

  private loadSession(id: string): Promise<SessionHandle> {
    const existing = this.sessions.get(id);
    if (existing) return existing;
    const pending = this.storage.open(id).then(async ({ header }) => {
      await this.mcp.initialize();
      const cwd = header.cwd ?? (typeof header.metadata?.cwd === 'string' ? header.metadata.cwd : undefined);
      const { rules, paths } = this.rules.loadWithPaths(cwd);
      const projection = await this.storage.projectActiveMessages(id);
      const metadata = header.metadata;
      const depth = typeof metadata?.depth === 'number' && typeof metadata.parentSessionId === 'string' ? metadata.depth : 0;
      const subagentProfile = depth > 0 && typeof metadata?.subagentProfile === 'string'
        ? this.subagents.getProfile(metadata.subagentProfile) : undefined;
      const prompts = subagentProfile?.systemPrompt ? {
        ...this.prompts,
        basePersona: [this.config.prompts?.basePersona ?? this.config.baseSystemPrompt, subagentProfile.systemPrompt]
          .filter(Boolean).join('\n\n'),
      } : this.prompts;
      const handle = new SessionHandle({
        id, storage: this.storage, router: this.router, toolRegistry: this.registry,
        systemPrompt: this.prompt.assemble({
          prompts, rules, skills: this.skills.modelIndex(), cwd,
        }),
        hooks: this.config.hooks, permissionEngine: this.permissions,
        globalPermissions: this.config.permissions, sessionPermissions: header.permissions,
        defaultPermissionEffect: this.config.defaultPermissionEffect, truncator: this.truncator,
        compaction: this.compaction, subagents: this.subagents, subagentProfile, depth, cwd,
        prompts, questions: this.questions, rulesLoader: this.rules, baselineRulePaths: paths,
        onClose: () => { if (this.sessions.get(id) === pending) this.sessions.delete(id); },
      }, projection);
      for (const rule of rules) this.loadedRules.add(rule);
      return handle;
    });
    this.sessions.set(id, pending);
    void pending.catch(() => {
      if (this.sessions.get(id) === pending) this.sessions.delete(id);
    });
    return pending;
  }

  listSessions(): Promise<SessionSummary[]> {
    return this.storage.list();
  }

  async deleteSession(id: string): Promise<void> {
    this.assertOpen();
    const existing = this.deleting.get(id);
    if (existing) return existing;
    const pending = this.deleteOpenSession(id);
    this.deleting.set(id, pending);
    try {
      await pending;
    } finally {
      this.deleting.delete(id);
    }
  }

  private async deleteOpenSession(id: string): Promise<void> {
    await Promise.allSettled(this.creating);
    const handle = await this.sessions.get(id)?.catch(() => undefined);
    if (handle) {
      try {
        await handle.close();
      } finally {
        this.sessions.delete(id);
      }
    }
    await this.storage.delete(id);
  }

  registerTools(tools: CoreToolDefinition[]): () => void {
    this.assertOpen();
    return this.registry.register(tools);
  }

  updateModelConfig(config: ModelRouterConfig): void {
    this.assertOpen();
    this.router.updateConfig(config);
  }

  reloadMcp(): Promise<void> {
    this.assertOpen();
    return this.mcp.reload();
  }

  getCatalog(): RuntimeCatalog {
    return {
      models: this.router.listProfiles().map(({ alias, modelId, providerId, protocol }) =>
        ({ alias, modelId, providerId, protocol })),
      tools: this.registry.snapshot().visible.map(({ name, label, description }) => ({ name, label, description })),
      rules: [...this.loadedRules],
      skills: this.skills.list(),
      mcpServers: this.config.mcpServers ? this.mcp.getCatalog() : undefined,
    };
  }

  dispose(): Promise<void> {
    this.disposal ??= this.closeSessions();
    return this.disposal;
  }

  private async closeSessions(): Promise<void> {
    await Promise.allSettled([...this.creating, ...this.deleting.values()]);
    const handles = await Promise.allSettled(this.sessions.values());
    const results = await Promise.allSettled(handles.flatMap((handle) =>
      handle.status === 'fulfilled' ? [handle.value.close()] : []));
    this.sessions.clear();
    const cleanup = await Promise.allSettled([this.mcp.dispose(), this.permissions.flush()]);
    this.registry.clear();
    for (const result of [...results, ...cleanup]) {
      if (result.status === 'rejected') throw result.reason;
    }
  }

  private assertOpen(): void {
    if (this.disposal) throw new Error('Agent runtime is disposed');
  }
}

export function createAgentRuntime(config: AgentRuntimeConfig): AgentRuntime {
  return new Runtime(config);
}
