import type { AsyncLocalStorage } from 'node:async_hooks';
import { randomUUID } from 'node:crypto';
import { Agent, type AgentEvent, type AgentMessage, type AgentTool, type PrepareRequest } from '@earendil-works/pi-agent-core';
import { createInitialSystemMessage, getCurrentTools, getToolStateChanges, toToolDeclaration, type Api, type AssistantMessage, type Model } from '@earendil-works/pi-ai';
import { isContextOverflow } from '@earendil-works/pi-ai/utils/overflow';
import { ModeController } from '../harness/mode.js';
import { PermissionEngine, type PermissionRulesets } from '../harness/permission.js';
import { PlanStore } from '../harness/plan.js';
import { createSystemUpdateMessage, DEFAULT_PROMPT_TEMPLATES, wrapSystemUpdate } from '../harness/prompt.js';
import type { QuestionManager } from '../harness/question.js';
import { isRuleUpdateMessage, RulesLoader, type RuleUpdateMessage } from '../harness/rules.js';
import type { SubagentManager, SubagentParent } from '../harness/subagent.js';
import { ToolRegistry, type ToolSnapshot } from '../harness/tool-registry.js';
import type { OutputTruncator } from '../harness/truncation.js';
import type { ModelRouter } from '../model/router.js';
import { CompactionEngine, estimateContextTokens, isRateLimitError, type CompactionTrigger } from '../session/compaction.js';
import type { AgentRuntimeHooks, PromptOptions, PromptTemplates, SubagentProfile } from '../types/config.js';
import type { AgentCoreEvent, AgentCoreEventData, TurnSummary } from '../types/events.js';
import type { ApprovalDecision, ApprovalRequest, PendingApprovalRequest, PermissionRule } from '../types/permission.js';
import type { PendingQuestionRequest, QuestionAnswer } from '../types/question.js';
import { SessionBatchAppendError } from '../types/session.js';
import type { AgentMode, CompactionResult, PlanItem, SessionProjection, SessionStorageAdapter } from '../types/session.js';
import type { CoreToolDefinition } from '../types/tool.js';
import type { Semaphore } from './concurrency.js';
import { EventMapper, sumTokenUsage } from './event-mapper.js';

export interface AgentSessionHandle {
  readonly id: string;
  readonly mode: AgentMode;
  prompt(input: string, options?: PromptOptions): Promise<TurnSummary>;
  steer(input: string): void;
  followUp(input: string): void;
  abort(): void;
  setMode(mode: AgentMode): Promise<void>;
  branch(fromEntryId: string): Promise<void>;
  getPlan(): PlanItem[];
  getPendingApprovals(): ApprovalRequest[];
  replyApproval(requestId: string, decision: ApprovalDecision): void;
  getPendingQuestions(): PendingQuestionRequest[];
  replyQuestion(requestId: string, answers: QuestionAnswer | { cancelled: true }): void;
  injectRulesForPath(targetPath: string): string[];
  compact(customInstructions?: string): Promise<CompactionResult>;
  close(): Promise<void>;
  subscribe(listener: (event: AgentCoreEvent) => void): () => void;
}

interface SessionHandleOptions {
  id: string;
  storage: SessionStorageAdapter;
  router: ModelRouter;
  getTools?(): CoreToolDefinition[];
  toolRegistry?: ToolRegistry;
  toolSlots?: Semaphore;
  initializeTools?(signal: AbortSignal): Promise<void>;
  permissionEngine?: PermissionEngine;
  globalPermissions?: readonly PermissionRule[];
  sessionPermissions?: readonly PermissionRule[];
  defaultPermissionEffect?: 'allow' | 'ask';
  truncator?: OutputTruncator;
  compaction?: CompactionEngine;
  subagents?: SubagentManager;
  subagentProfile?: SubagentProfile;
  depth?: number;
  subagentAncestry?: Readonly<{ parentSessionId?: string; subagentProfile?: string; depth: number }>;
  turnContext?: AsyncLocalStorage<ReadonlyMap<string, AbortController>>;
  cwd?: string;
  systemPrompt?: string;
  hooks?: AgentRuntimeHooks;
  prompts?: PromptTemplates;
  suppressModePrompt?: boolean;
  questions?: QuestionManager;
  rulesLoader?: RulesLoader;
  baselineRulePaths?: ReadonlySet<string>;
  onClose?(): void;
}

interface ActiveRun {
  summary: TurnSummary;
  controller: AbortController;
  acceptingMessages: boolean;
  promise: Promise<TurnSummary>;
  steps: number;
  finalizing: boolean;
  retryOverflow: boolean;
  overflowMessage?: AgentMessage;
  requestModel?: Model<Api>;
  toolSnapshot?: { visible: CoreToolDefinition[]; tools: AgentTool[] };
  toolPermissionVersions?: Map<string, number>;
}

interface CachedToolSnapshot {
  registryVersion: number;
  definitionFingerprint: string;
  permissionVersion: number;
  mode: AgentMode;
  profileKey: string;
  snapshot: ToolSnapshot;
}

type EventListener = (event: AgentCoreEvent) => void;
const PATH_FIELDS = ['path', 'filePath', 'file_path', 'cwd', 'directory'] as const;

function isRequestOverflow(message: AssistantMessage, contextWindow: number): boolean {
  if (message.stopReason === 'error') {
    const error = message.errorMessage?.trim() ?? '';
    if (isRateLimitError(error)) return false;
    // Older gateways may return just this diagnostic.
    if (/^maximum context reached[.!]?$/i.test(error)) return true;
  }
  return isContextOverflow(message, contextWindow);
}

export class SessionHandle implements AgentSessionHandle {
  readonly id: string;
  private readonly agent: Agent;
  private readonly mapper: EventMapper;
  private readonly listeners = new Set<EventListener>();
  private readonly detach: () => void;
  readonly planStore: PlanStore;
  private readonly modeController: ModeController;
  private readonly permissions: PermissionEngine;
  private readonly permissionRules: PermissionRulesets;
  private readonly compaction: CompactionEngine;
  private readonly rulesLoader: RulesLoader;
  private readonly pendingRules: Array<{ path: string; rule: string }> = [];
  private readonly deferredInputs = new Set<Promise<void>>();
  private loadedRulePaths = new Set<string>();
  private toolDefinitions: ReadonlyMap<string, CoreToolDefinition> = new Map();
  private cachedTools?: CachedToolSnapshot;
  private modelAlias?: string;
  private savedMessageCount: number;
  private usageAnchorStart = 0;
  private active?: ActiveRun;
  private changes = Promise.resolve();
  private persistence = Promise.resolve();
  private persistenceNeedsRetry = false;
  private persistenceBlocked?: SessionBatchAppendError;
  private closed = false;
  private closing?: Promise<void>;
  private manualCompaction?: AbortController;

  constructor(private readonly options: SessionHandleOptions, projection: SessionProjection) {
    this.id = options.id;
    const prompts = options.prompts ?? DEFAULT_PROMPT_TEMPLATES;
    this.modeController = new ModeController(projection.mode, projection.modeChanged, prompts);
    this.permissions = options.permissionEngine ?? new PermissionEngine();
    this.permissionRules = [options.globalPermissions ?? [], options.sessionPermissions ?? []];
    this.compaction = options.compaction ?? new CompactionEngine(options.storage, options.router, {}, prompts);
    this.rulesLoader = options.rulesLoader ?? new RulesLoader();
    this.restoreRulePaths(projection.messages);
    this.modelAlias = projection.modelAlias;
    this.mapper = new EventMapper(this.id);
    this.planStore = new PlanStore(this.id, options.storage, projection.plan, (items, runId) => {
      this.emit(this.mapper.create(runId ?? randomUUID(), { type: 'plan_update', items }));
    }, () => this.persistMessages());
    const hooks = options.hooks;
    this.agent = new Agent({
      ...hooks,
      sessionId: this.id,
      streamFn: (model, context, streamOptions) => options.router.streamFn(model, context,
        this.active?.finalizing ? { ...streamOptions, toolChoice: 'none' } : streamOptions),
      initialState: {
        messages: projection.messages,
        systemPrompt: options.systemPrompt,
      },
      transformContext: async (messages, signal) => {
        const withRules = this.transformRuleContext(messages);
        const context = await hooks?.transformContext?.(withRules, signal) ?? withRules;
        const transformed = options.suppressModePrompt ? context : this.modeController.transformContext(context);
        if (!this.active?.finalizing) return transformed;
        // Providers can replay historical tool declarations from system messages.
        const withoutTools = transformed.map((message) => {
          if (message.role !== 'system') return message;
          const { toolsAdded, toolsRemoved, ...rest } = message;
          return rest;
        });
        if (!prompts.maxStepsSummary.trim()) return withoutTools;
        return [...withoutTools, createSystemUpdateMessage(prompts.maxStepsSummary)];
      },
      prepareRequest: this.prepareRequest,
      finishTurn: () => this.active?.finalizing || this.active?.overflowMessage ? { action: 'end' } : undefined,
      prepareNextTurnWithContext: ({ context }) => ({
        context: { ...context, tools: this.refreshTools() },
      }),
      beforeToolCall: async (context, signal) => {
        const { toolCall, args } = context;
        const tool = this.toolDefinitions.get(toolCall.name) ?? { name: toolCall.name, readOnly: false };
        const run = this.active!;
        const reason = this.toolBlockReason(tool, args);
        if (reason) return { block: true, reason };
        const permissionVersion = options.toolSlots && tool !== options.questions?.tool && tool !== options.subagents?.tool
          ? this.permissions.refreshVersion(this.permissionRules) : undefined;
        const result = await this.permissions.beforeToolCall({
          sessionId: this.id, runId: run.summary.runId, toolCallId: toolCall.id, tool, args,
          signal: signal ?? run.controller.signal,
          rulesets: () => this.permissionRules,
          modeRules: () => this.modeController.rulesFor(tool),
          defaultEffect: options.defaultPermissionEffect,
          onApprovalRequested: (request) => this.emit(this.mapper.create(request.runId, { type: 'approval_requested', request })),
          onApprovalResolved: (request, decision) =>
            this.emit(this.mapper.create(request.runId, { type: 'approval_resolved', requestId: request.id, decision })),
        });
        if (result?.block) return result;
        const updatedReason = this.toolBlockReason(tool, args);
        if (updatedReason) return { block: true, reason: updatedReason };
        if (permissionVersion !== undefined) (run.toolPermissionVersions ??= new Map()).set(toolCall.id, permissionVersion);
        return hooks?.beforeToolCall?.(context, signal);
      },
      afterToolCall: async (context, signal) => {
        if (options.cwd && context.args && typeof context.args === 'object' && !signal?.aborted) {
          const paths = PATH_FIELDS.map((field) => (context.args as Record<string, unknown>)[field])
            .filter((path) => typeof path === 'string');
          const found = await this.rulesLoader.discoverForPathsAsync(paths, this.loadedRulePaths, options.cwd);
          if (!signal?.aborted && !this.closed) this.queueRuleUpdates(found);
        }
        const host = await hooks?.afterToolCall?.(context, signal);
        // Keep structured answers intact, including long free-form replies.
        const questionTool = options.questions?.tool;
        if (!options.truncator || (questionTool && questionTool === this.toolDefinitions.get(context.toolCall.name))) return host;
        const result = {
          content: host?.content ?? context.result.content ?? [],
          details: host?.details ?? context.result.details,
        };
        const truncated = await options.truncator.truncate(this.id, context.toolCall.id, result, context.toolCall.name);
        if (truncated === result) return host;
        return { ...host, content: truncated.content, details: truncated.details, structuredContent: undefined };
      },
    });
    // An older transcript can predate a base system message. Count the synthesized head
    // so that already persisted messages are not appended again.
    this.savedMessageCount = projection.messages.length ? this.agent.state.messages.length : 0;
    this.usageAnchorStart = projection.compactedMessageCount === undefined ? 0 :
      projection.compactedMessageCount + this.agent.state.messages.length - projection.messages.length;
    this.detach = this.agent.subscribe((event, signal) => this.onAgentEvent(event, signal));
  }

  get mode(): AgentMode {
    return this.modeController.mode;
  }

  getSubagentParent(): SubagentParent {
    return {
      id: this.id, mode: this.mode, cwd: this.options.cwd,
      depth: this.options.depth ?? 0, modelAlias: this.modelAlias,
      readOnly: this.options.subagentProfile?.readOnly === true,
      permissions: this.options.sessionPermissions?.slice(),
      followUp: (input) => this.queueInput('followUp', input, (this.options.depth ?? 0) > 0),
      emit: (runId, event) => this.emitRunEvent(runId, event),
    };
  }

  get subagentAncestry() {
    return this.options.subagentAncestry;
  }

  subscribe(listener: EventListener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  private emit(event: AgentCoreEvent): void {
    for (const listener of [...this.listeners]) {
      try {
        listener(event);
      } catch {
        // An observer can fail independently of the conversation and its persistence.
      }
    }
  }

  emitRunEvent(runId: string, event: AgentCoreEventData): void {
    this.emit(this.mapper.create(runId, event));
  }

  injectRulesForPath(targetPath: string): string[] {
    if (this.closed) return [];
    return this.queueRuleUpdates(this.rulesLoader.discoverForPath(targetPath, this.loadedRulePaths, this.options.cwd));
  }

  private queueRuleUpdates(found: Array<{ path: string; rule: string }>): string[] {
    const paths: string[] = [];
    for (const rule of found) {
      if (this.loadedRulePaths.has(rule.path)) continue;
      this.loadedRulePaths.add(rule.path);
      this.pendingRules.push(rule);
      paths.push(rule.path);
    }
    return paths;
  }

  private transformRuleContext(messages: AgentMessage[]): AgentMessage[] {
    if (!this.pendingRules.length) return messages;
    const rules = this.pendingRules.splice(0);
    const update: RuleUpdateMessage = {
      role: 'user',
      content: wrapSystemUpdate(rules.map(({ path, rule }) => `Rules from ${path}:\n\n${rule}`).join('\n\n')),
      timestamp: Date.now(),
      rulePaths: rules.map(({ path }) => path),
    };
    // Keep the update in the transcript so compaction and branches can restore the path index.
    const last = messages.at(-1);
    this.agent.state.messages.push(update);
    messages.push(update);
    if (last?.role !== 'user') return messages;
    return [...messages.slice(0, -2), update, last];
  }

  private restoreRulePaths(messages: readonly AgentMessage[]): void {
    this.loadedRulePaths = new Set(this.options.baselineRulePaths);
    for (const message of messages) {
      if (!isRuleUpdateMessage(message)) continue;
      for (const path of message.rulePaths) this.loadedRulePaths.add(path);
    }
    for (const { path } of this.pendingRules) this.loadedRulePaths.add(path);
  }

  async prompt(input: string, options: PromptOptions = {}): Promise<TurnSummary> {
    if (this.closed) throw new Error(`Session "${this.id}" is closed`);
    if (this.active) throw new Error('Session is already processing a prompt. Use steer() or followUp().');

    const summary: TurnSummary = { sessionId: this.id, runId: randomUUID(), state: 'completed', text: '' };
    const controller = new AbortController();
    const turnContext = this.options.turnContext;
    const execute = () => this.runPrompt(input, options, summary, controller.signal);
    const promise = turnContext
      ? turnContext.run(new Map(turnContext.getStore()).set(this.id, controller), execute)
      : execute();
    const run: ActiveRun = {
      summary, controller, acceptingMessages: true, steps: 0, finalizing: false, retryOverflow: this.compaction.enabled,
      promise,
    };
    this.active = run;
    try {
      return await run.promise;
    } finally {
      this.agent.clearAllQueues();
      this.active = undefined;
    }
  }

  private async runPrompt(
    input: string, options: PromptOptions, summary: TurnSummary, signal: AbortSignal,
  ): Promise<TurnSummary> {
    let subagentLease: Awaited<ReturnType<SubagentManager['acquireTurnLease']>>;
    try {
      await this.changes;
      const alias = options.modelAlias ?? this.modelAlias;
      const route = this.options.router.resolve(alias);
      if (!signal.aborted && this.options.initializeTools) {
        try {
          await this.options.initializeTools(signal);
        } catch (error) {
          if (!signal.aborted) throw error;
        }
      }
      if (!signal.aborted && this.options.subagents) {
        try {
          subagentLease = await this.options.subagents.acquireTurnLease(this.id, this.options.depth ?? 0, signal);
        } catch (error) {
          if (!signal.aborted) throw error;
        }
      }
      if (signal.aborted) {
        summary.state = 'cancelled';
        this.emit(this.mapper.create(summary.runId, { type: 'turn_start', mode: this.mode, modelId: route.model.id }));
        this.emit(this.mapper.create(summary.runId, { type: 'turn_end', state: summary.state }));
        return summary;
      }

      const executeTurn = async (): Promise<TurnSummary> => {
        await this.persistMessages();
        if (options.modelAlias !== undefined && options.modelAlias !== this.modelAlias) {
          await this.options.storage.append(this.id, { type: 'model', modelAlias: options.modelAlias });
          this.modelAlias = options.modelAlias;
        }

        this.agent.state.model = route.model;
        this.agent.state.thinkingLevel = options.thinkingLevel ?? route.thinkingLevel;
        this.refreshTools();
        await this.agent.prompt(input, options.images);
        const run = this.active!;
        while (!signal.aborted) {
          if (run.overflowMessage) {
            const failed = run.overflowMessage;
            run.overflowMessage = undefined;
            run.retryOverflow = false;
            // A provider can include tool calls even on a stop/length response. Keep their results paired.
            const removeResponse = this.agent.state.messages.at(-1) === failed;
            if (removeResponse) this.agent.state.messages.pop();
            run.acceptingMessages = true;
            let result: CompactionResult | undefined;
            try {
              result = await this.compactContext('overflow', run.requestModel ?? route.model, summary.runId, signal);
            } finally {
              if (removeResponse && result?.status !== 'compacted') this.agent.state.messages.push(failed);
            }
            if (signal.aborted) summary.state = 'cancelled';
            if (result.status !== 'compacted' || signal.aborted) break;
            summary.text = '';
            this.refreshTools();
            await this.agent.continue();
            continue;
          }
          if (summary.state !== 'completed' || run.finalizing || !this.agent.hasQueuedMessages()) break;
          // A message can arrive between Pi's final queue poll and agent_end.
          run.acceptingMessages = true;
          this.refreshTools();
          await this.agent.continue();
        }
        return summary;
      };
      return subagentLease
        ? await this.options.subagents!.runWithTurnLease(subagentLease, executeTurn)
        : await executeTurn();
    } finally {
      try {
        if (this.active) this.active.overflowMessage = undefined;
        await this.persistMessages();
        const model = this.active?.requestModel;
        if (this.compaction.precomputeEnabled && model && !signal.aborted && summary.state === 'completed') {
          this.compaction.maybePrecompute(this.id, model, estimateContextTokens(this.agent.state.messages, this.usageAnchorStart));
        }
      } finally {
        subagentLease?.release();
      }
    }
  }

  private toolBlockReason(tool: Pick<CoreToolDefinition, 'name' | 'readOnly'>, args: unknown): string | undefined {
    if (this.modeController.blocks(tool)) return `Plan mode blocked "${tool.name}". This tool is not read-only.`;
    if ((this.mode === 'plan' || this.options.subagentProfile?.readOnly) && tool === this.options.subagents?.tool) {
      return this.options.subagents.planBlockReason(args);
    }
    return undefined;
  }

  private getToolSnapshot(): ToolSnapshot {
    const { toolRegistry, getTools, subagentProfile: profile, depth = 0 } = this.options;
    const registryVersion = toolRegistry?.version ?? 0;
    const definitionFingerprint = toolRegistry?.definitionFingerprint() ?? '';
    const permissionVersion = this.permissions.refreshVersion(this.permissionRules);
    const mode = this.mode;
    const profileKey = profile ? JSON.stringify([depth, profile.readOnly, profile.allowedTools, profile.deniedTools]) : String(depth);
    const cached = this.cachedTools;
    if (cached && cached.registryVersion === registryVersion && cached.definitionFingerprint === definitionFingerprint &&
        cached.permissionVersion === permissionVersion &&
        cached.mode === mode && cached.profileKey === profileKey) return cached.snapshot;

    const savedRules = this.permissions.savedRules;
    const registry = toolRegistry ?? new ToolRegistry(getTools?.());
    const snapshot = registry.snapshot((tool) =>
      (this.options.subagents?.allowsTool(tool, depth, profile) ?? true) &&
      !this.permissions.isToolGloballyDenied(tool, [...this.permissionRules, savedRules, this.modeController.rulesFor(tool)]));
    // A direct SessionHandle caller may supply an unversioned dynamic getTools callback.
    if (toolRegistry) this.cachedTools = { registryVersion, definitionFingerprint, permissionVersion, mode, profileKey, snapshot };
    return snapshot;
  }

  private refreshTools(): AgentTool[] {
    const { definitions, visible } = this.getToolSnapshot();
    this.toolDefinitions = definitions;
    const run = this.active!;
    if (run.toolSnapshot?.visible === visible) {
      this.agent.state.tools = run.toolSnapshot.tools;
      return run.toolSnapshot.tools;
    }
    const { runId } = run.summary;
    const defaultSignal = run.controller.signal;
    const { toolSlots, questions, subagents } = this.options;
    const snapshot = visible.map((tool): AgentTool => ({
      name: tool.name,
      label: tool.label,
      description: tool.description,
      parameters: tool.parameters,
      executionMode: tool.executionMode,
      execute: async (toolCallId, params, signal = defaultSignal, onUpdate) => {
        const release = toolSlots && tool !== questions?.tool && tool !== subagents?.tool
          ? await toolSlots.acquire(signal) : undefined;
        try {
          if (release) signal.throwIfAborted();
          // Pi can prepare multiple calls before executing them. The mode can change in between.
          const reason = this.toolBlockReason(tool, params);
          if (reason) throw new Error(reason);
          if (release) {
            const denied = this.permissions.denialReason({
              tool, args: params, rulesets: () => this.permissionRules, modeRules: () => this.modeController.rulesFor(tool),
              defaultEffect: this.options.defaultPermissionEffect,
            });
            if (denied) throw new Error(denied);
          }
          const result = await tool.execute(params as Record<string, unknown>, {
            sessionId: this.id, runId, toolCallId, signal,
            onUpdate: (partial) => onUpdate?.({ ...partial, details: partial.details }),
          });
          return { ...result, details: result.details };
        } finally {
          release?.();
        }
      },
    }));
    run.toolSnapshot = { visible, tools: snapshot };
    this.agent.state.tools = snapshot;
    return snapshot;
  }

  private readonly prepareRequest: PrepareRequest = async (request, signal) => {
    const run = this.active!;
    let { context } = request;
    if (run.retryOverflow && request.model.contextWindow > 0 && !signal?.aborted) {
      const tokens = estimateContextTokens(context.messages, this.usageAnchorStart);
      if (tokens >= request.model.contextWindow - this.compaction.reserveTokens) {
        const result = await this.compactContext('auto', request.model, run.summary.runId, signal);
        if (result.status === 'compacted') {
          const messages = this.agent.state.messages.slice();
          const delta = getToolStateChanges(getCurrentTools(messages), (context.tools ?? []).map(toToolDeclaration));
          if (delta.toolsAdded.length || delta.toolsRemoved.length) {
            const firstDialogue = messages.findIndex(({ role }) => role !== 'system');
            messages.splice(firstDialogue, 0, { role: 'system', content: '', timestamp: Date.now(), ...delta });
          }
          context = { ...context, messages };
        }
      } else {
        this.compaction.maybePrecompute(this.id, request.model, tokens);
      }
    }
    let update = await this.options.hooks?.prepareRequest?.({ ...request, context }, signal);
    if (update?.model) update = { ...update, model: this.options.router.prepareModel(request.model, update.model) };
    run.requestModel = update?.model ?? request.model;
    const maxSteps = this.options.subagentProfile?.maxSteps;
    run.finalizing = maxSteps !== undefined && ++run.steps >= maxSteps;
    if (run.finalizing) return { ...update, context: { ...(update?.context ?? context), tools: [] } };
    return context !== request.context ? { ...update, context: update?.context ?? context } : update || undefined;
  };

  private onAgentEvent(event: AgentEvent, signal: AbortSignal): void {
    const run = this.active;
    if (!run) return;
    if (event.type === 'agent_start' && run.controller.signal.aborted) this.agent.abort();
    if (event.type === 'agent_end') run.acceptingMessages = false;
    // Detect the overflow before Pi decides whether to drain queued messages or start another tool turn.
    if (event.type === 'message_end' && event.message.role === 'assistant' && run.retryOverflow && !signal.aborted) {
      const contextWindow = (run.requestModel ?? this.agent.state.model).contextWindow;
      if (isRequestOverflow(event.message, contextWindow)) run.overflowMessage = event.message;
    }
    const mapped = this.mapper.map(event, {
      runId: run.summary.runId, mode: this.mode, modelId: this.agent.state.model.id, signal,
      getToolLabel: (name) => this.agent.state.tools.find((tool) => tool.name === name)?.label,
    });
    if (mapped?.type === 'turn_end' && event.type === 'turn_end') {
      const message = event.message;
      run.summary.state = mapped.state;
      run.summary.usage = sumTokenUsage(run.summary.usage, mapped.usage);
      run.summary.error = mapped.error;
      if (message.role === 'assistant') {
        const text = message.content.flatMap((block) => block.type === 'text' ? [block.text] : []).join('');
        if (text || mapped.state === 'completed') run.summary.text = text;
      }
    }
    if (mapped) this.emit(mapped);
  }

  private persistMessages(): Promise<void> {
    const messages = this.agent.state.messages;
    const overflow = this.active?.overflowMessage;
    const end = messages.length - Number(!!overflow && overflow === messages.at(-1));
    const start = this.savedMessageCount;
    const suffix = messages.slice(start, end);
    const write = this.persistence.then(async () => {
      if (this.persistenceBlocked) throw this.persistenceBlocked;
      const { storage } = this.options;
      try {
        if (storage.appendBatch && end - this.savedMessageCount > 1) {
          const entries = suffix.slice(this.savedMessageCount - start)
            .map((message) => ({ type: 'message' as const, message }));
          await storage.appendBatch(this.id, entries);
          this.savedMessageCount = end;
        } else {
          while (this.savedMessageCount < end) {
            await storage.append(this.id, { type: 'message', message: suffix[this.savedMessageCount - start] });
            this.savedMessageCount++;
          }
        }
        this.persistenceNeedsRetry = false;
      } catch (error) {
        this.persistenceNeedsRetry = true;
        if (error instanceof SessionBatchAppendError) {
          this.savedMessageCount += error.committedCount ?? 0;
          if (!error.retrySafe) this.persistenceBlocked = error;
        }
        throw error;
      }
    });
    this.persistence = write.catch(() => undefined);
    return write;
  }

  private queueInput(kind: 'steer' | 'followUp', input: string, afterTurn = false): void {
    if (this.closed) return;
    const run = this.active;
    if (!run) {
      void this.prompt(input).catch((error: unknown) => {
        this.emit(this.mapper.create(randomUUID(), {
          type: 'turn_end', state: 'failed', error: error instanceof Error ? error.message : String(error),
        }));
      });
      return;
    }
    if (run.controller.signal.aborted) return;
    if (run.acceptingMessages && !afterTurn) {
      const message: AgentMessage = { role: 'user', content: input, timestamp: Date.now() };
      this.agent[kind](message);
    } else {
      // Background results need a fresh turn when the current one can stop without draining its queue.
      const enqueue = () => {
        if (!run.controller.signal.aborted) this.queueInput(kind, input, afterTurn);
      };
      const pending = run.promise.then(enqueue, enqueue).finally(() => this.deferredInputs.delete(pending));
      this.deferredInputs.add(pending);
    }
  }

  steer(input: string): void {
    this.queueInput('steer', input);
  }

  followUp(input: string): void {
    this.queueInput('followUp', input);
  }

  abort(): void {
    this.options.subagents?.abort(this.id);
    this.compaction.clearPrecompute(this.id);
    this.manualCompaction?.abort();
    const run = this.active;
    if (!run) return;
    run.controller.abort();
    this.agent.abort();
    this.agent.clearAllQueues();
  }

  private change<T>(operation: () => Promise<T>): Promise<T> {
    if (this.closed) return Promise.reject(new Error(`Session "${this.id}" is closed`));
    const next = this.changes.then(operation);
    this.changes = next.then(() => undefined, () => undefined);
    return next;
  }

  setMode(mode: AgentMode): Promise<void> {
    return this.change(async () => {
      const from = this.mode;
      if (from === mode) return;
      await this.persistMessages();
      await this.options.storage.append(this.id, { type: 'mode', mode });
      this.modeController.setMode(mode);
      this.emit(this.mapper.create(this.active?.summary.runId ?? randomUUID(), { type: 'mode_changed', from, to: mode }));
    });
  }

  branch(fromEntryId: string): Promise<void> {
    this.compaction.clearPrecompute(this.id);
    const active = this.active?.promise;
    return this.change(async () => {
      await active?.catch(() => undefined);
      this.compaction.clearPrecompute(this.id);
      await this.persistMessages();
      await this.options.storage.branch(this.id, fromEntryId);
      const projection = await this.options.storage.projectActiveMessages(this.id);
      const from = this.mode;
      this.modeController.restore(projection.mode, projection.modeChanged);
      this.modelAlias = projection.modelAlias;
      this.planStore.restore(projection.plan);
      this.pendingRules.length = 0;
      this.restoreMessages(projection.messages, projection.compactedMessageCount);
      const runId = randomUUID();
      if (from !== this.mode) {
        this.emit(this.mapper.create(runId, { type: 'mode_changed', from, to: this.mode }));
      }
      this.emit(this.mapper.create(runId, { type: 'plan_update', items: this.getPlan() }));
    });
  }

  getPlan(): PlanItem[] {
    return this.planStore.getItems();
  }

  getPendingApprovals(): PendingApprovalRequest[] {
    return this.permissions.getPendingApprovals(this.id);
  }

  replyApproval(requestId: string, decision: ApprovalDecision): void {
    this.permissions.replyApproval(requestId, decision, this.id);
  }

  getPendingQuestions(): PendingQuestionRequest[] {
    return this.options.questions?.getPendingQuestions(this.id) ?? [];
  }

  replyQuestion(requestId: string, answers: QuestionAnswer | { cancelled: true }): void {
    this.options.questions?.replyQuestion(this.id, requestId, answers);
  }

  compact(customInstructions?: string): Promise<CompactionResult> {
    const active = this.active?.promise;
    return this.change(async () => {
      await active?.catch(() => undefined);
      if (this.closed) return { status: 'skipped', reason: 'aborted' };
      const controller = new AbortController();
      this.manualCompaction = controller;
      try {
        const { model } = this.options.router.resolve(this.modelAlias);
        return await this.compactContext('manual', model, randomUUID(), controller.signal, customInstructions);
      } finally {
        this.manualCompaction = undefined;
      }
    });
  }

  private restoreMessages(messages: AgentMessage[], compactedMessageCount?: number): void {
    this.restoreRulePaths(messages);
    const baseline = createInitialSystemMessage(this.options.systemPrompt, undefined);
    this.agent.state.messages = baseline && messages[0]?.role !== 'system' ? [baseline, ...messages] : messages;
    this.savedMessageCount = this.agent.state.messages.length;
    this.usageAnchorStart = compactedMessageCount === undefined ? 0 :
      compactedMessageCount + this.savedMessageCount - messages.length;
  }

  private async compactContext(
    trigger: CompactionTrigger, model: Model<Api>, runId: string, signal?: AbortSignal, customInstructions?: string,
  ): Promise<CompactionResult> {
    const tokensBefore = estimateContextTokens(this.agent.state.messages, this.usageAnchorStart);
    let started = false;
    const start = () => {
      if (started) return;
      started = true;
      this.emit(this.mapper.create(runId, { type: 'compaction_start', trigger, tokensBefore }));
    };
    let end: Extract<AgentCoreEventData, { type: 'compaction_end' }> = {
      type: 'compaction_end', tokensBefore, tokensAfter: tokensBefore, summary: '',
    };
    if (trigger === 'manual') start();
    try {
      await this.persistMessages();
      const result = await this.compaction.compact({
        sessionId: this.id, model, tokensBefore, trigger, signal, customInstructions,
        systemPrompt: this.options.systemPrompt, onStart: start,
      });
      if (result.status !== 'compacted') return result;
      const { messages, ...checkpoint } = result;
      this.restoreMessages(messages, messages.length);
      end = { ...end, tokensAfter: checkpoint.tokensAfter, summary: checkpoint.summary };
      return checkpoint;
    } finally {
      if (started) this.emit(this.mapper.create(runId, end));
    }
  }

  get closeCompletion(): Promise<void> | undefined {
    return this.closing;
  }

  assertCanWaitForClose(): void {
    if (this.active && this.options.turnContext?.getStore()?.get(this.id) === this.active.controller) {
      throw new Error(`Cannot open session "${this.id}" while it is closing from its own active turn`);
    }
  }

  async waitForBackgroundWork(): Promise<void> {
    const { subagents } = this.options;
    while (!this.closed && (this.active || this.deferredInputs.size || subagents?.hasActive(this.id))) {
      await Promise.allSettled([this.active?.promise, ...this.deferredInputs, subagents?.wait(this.id)]);
    }
  }

  close(): Promise<void> {
    this.closed = true;
    this.closing ??= Promise.resolve().then(() => this.closeInternal());
    return this.closing;
  }

  private async closeInternal(): Promise<void> {
    try {
      this.abort();
      const [run] = await Promise.allSettled([this.active?.promise, this.changes, this.options.subagents?.wait(this.id)]);
      if (this.persistenceNeedsRetry && this.savedMessageCount < this.agent.state.messages.length) {
        await this.persistMessages();
      }
      if (run.status === 'rejected') throw run.reason;
    } finally {
      this.detach();
      this.listeners.clear();
      this.options.onClose?.();
    }
  }
}
