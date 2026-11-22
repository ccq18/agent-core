import { randomUUID } from 'node:crypto';
import { Agent, type AgentEvent, type AgentMessage, type AgentTool, type PrepareRequest } from '@earendil-works/pi-agent-core';
import { createInitialSystemMessage, getCurrentTools, getToolStateChanges, toToolDeclaration, type Api, type Model } from '@earendil-works/pi-ai';
import { ModeController } from '../harness/mode.js';
import { PermissionEngine, type PermissionRulesets } from '../harness/permission.js';
import { PlanStore } from '../harness/plan.js';
import type { SubagentManager, SubagentParent } from '../harness/subagent.js';
import { ToolRegistry } from '../harness/tool-registry.js';
import type { OutputTruncator } from '../harness/truncation.js';
import type { ModelRouter } from '../model/router.js';
import { CompactionEngine, estimateContextTokens, type CompactionTrigger } from '../session/compaction.js';
import type { AgentRuntimeHooks, PromptOptions, SubagentProfile } from '../types/config.js';
import type { AgentCoreEvent, AgentCoreEventData, TurnSummary } from '../types/events.js';
import type { ApprovalDecision, ApprovalRequest, PendingApprovalRequest, PermissionRule } from '../types/permission.js';
import type { AgentMode, CompactionResult, PlanItem, SessionProjection, SessionStorageAdapter } from '../types/session.js';
import type { CoreToolDefinition } from '../types/tool.js';
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
  permissionEngine?: PermissionEngine;
  globalPermissions?: readonly PermissionRule[];
  sessionPermissions?: readonly PermissionRule[];
  defaultPermissionEffect?: 'allow' | 'ask';
  truncator?: OutputTruncator;
  compaction?: CompactionEngine;
  subagents?: SubagentManager;
  subagentProfile?: SubagentProfile;
  depth?: number;
  cwd?: string;
  systemPrompt?: string;
  hooks?: AgentRuntimeHooks;
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
}

type EventListener = (event: AgentCoreEvent) => void;
const OVERFLOW = /context length|maximum context|too many tokens|context_length_exceeded|prompt is too long/i;
const SUBAGENT_FINAL = '<system-update>Tool steps are exhausted. Return your final summary now using the information already gathered.</system-update>';

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
  private toolDefinitions: ReadonlyMap<string, CoreToolDefinition> = new Map();
  private modelAlias?: string;
  private savedMessageCount: number;
  private usageAnchorStart = 0;
  private active?: ActiveRun;
  private changes = Promise.resolve();
  private persistence = Promise.resolve();
  private closed = false;
  private closing?: Promise<void>;
  private manualCompaction?: AbortController;

  constructor(private readonly options: SessionHandleOptions, projection: SessionProjection) {
    this.id = options.id;
    this.modeController = new ModeController(projection.mode, projection.modeChanged);
    this.permissions = options.permissionEngine ?? new PermissionEngine();
    this.permissionRules = [options.globalPermissions ?? [], options.sessionPermissions ?? []];
    this.compaction = options.compaction ?? new CompactionEngine(options.storage, options.router);
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
        const transformed = this.modeController.transformContext(await hooks?.transformContext?.(messages, signal) ?? messages);
        if (!this.active?.finalizing) return transformed;
        // Providers can replay historical tool declarations from system messages.
        const withoutTools = transformed.map((message) => {
          if (message.role !== 'system') return message;
          const { toolsAdded, toolsRemoved, ...rest } = message;
          return rest;
        });
        return [...withoutTools, { role: 'user', content: SUBAGENT_FINAL, timestamp: Date.now() }];
      },
      prepareRequest: this.prepareRequest,
      finishTurn: () => this.active?.finalizing ? { action: 'end' } : undefined,
      prepareNextTurnWithContext: ({ context }) => ({
        context: { ...context, tools: this.refreshTools() },
      }),
      beforeToolCall: async (context, signal) => {
        const { toolCall, args } = context;
        const tool = this.toolDefinitions.get(toolCall.name) ?? { name: toolCall.name, readOnly: false };
        const run = this.active!;
        const reason = this.toolBlockReason(tool, args);
        if (reason) return { block: true, reason };
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
        return hooks?.beforeToolCall?.(context, signal);
      },
      afterToolCall: async (context, signal) => {
        const host = await hooks?.afterToolCall?.(context, signal);
        if (!options.truncator) return host;
        const result = {
          content: host?.content ?? context.result.content ?? [],
          details: host?.details ?? context.result.details,
        };
        const truncated = await options.truncator.truncate(this.id, context.toolCall.id, result);
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
      followUp: (input) => this.followUp(input),
      emit: (runId, event) => this.emit(this.mapper.create(runId, event)),
    };
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

  async prompt(input: string, options: PromptOptions = {}): Promise<TurnSummary> {
    if (this.closed) throw new Error(`Session "${this.id}" is closed`);
    if (this.active) throw new Error('Session is already processing a prompt. Use steer() or followUp().');

    const summary: TurnSummary = { sessionId: this.id, runId: randomUUID(), state: 'completed', text: '' };
    const controller = new AbortController();
    const run: ActiveRun = {
      summary, controller, acceptingMessages: true, steps: 0, finalizing: false, retryOverflow: this.compaction.enabled,
      promise: this.runPrompt(input, options, summary, controller.signal),
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
    try {
      await this.changes;
      const alias = options.modelAlias ?? this.modelAlias;
      const route = this.options.router.resolve(alias);
      if (signal.aborted) {
        summary.state = 'cancelled';
        this.emit(this.mapper.create(summary.runId, { type: 'turn_start', mode: this.mode, modelId: route.model.id }));
        this.emit(this.mapper.create(summary.runId, { type: 'turn_end', state: summary.state }));
        return summary;
      }

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
          this.agent.state.messages.pop();
          run.acceptingMessages = true;
          let result: CompactionResult | undefined;
          try {
            result = await this.compactContext('overflow', run.requestModel ?? route.model, summary.runId, signal);
          } finally {
            if (result?.status !== 'compacted') this.agent.state.messages.push(failed);
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
    } finally {
      if (this.active) this.active.overflowMessage = undefined;
      await this.persistMessages();
    }
  }

  private toolBlockReason(tool: Pick<CoreToolDefinition, 'name' | 'readOnly'>, args: unknown): string | undefined {
    if (this.modeController.blocks(tool)) return `Plan mode blocked "${tool.name}". This tool is not read-only.`;
    if ((this.mode === 'plan' || this.options.subagentProfile?.readOnly) && tool === this.options.subagents?.tool) {
      return this.options.subagents.planBlockReason(args);
    }
    return undefined;
  }

  private refreshTools(): AgentTool[] {
    const savedRules = this.permissions.savedRules;
    const registry = this.options.toolRegistry ?? new ToolRegistry(this.options.getTools?.());
    const { definitions, visible } = registry.snapshot((tool) =>
      (this.options.subagents?.allowsTool(tool, this.options.depth ?? 0, this.options.subagentProfile) ?? true) &&
      !this.permissions.isToolGloballyDenied(tool, [...this.permissionRules, savedRules, this.modeController.rulesFor(tool)]));
    this.toolDefinitions = definitions;
    const run = this.active!;
    const snapshot = visible.map((tool): AgentTool => ({
      name: tool.name,
      label: tool.label,
      description: tool.description,
      parameters: tool.parameters,
      executionMode: tool.executionMode,
      execute: async (toolCallId, params, signal = run.controller.signal, onUpdate) => {
        // Pi can prepare multiple calls before executing them. The mode can change in between.
        const reason = this.toolBlockReason(tool, params);
        if (reason) throw new Error(reason);
        const result = await tool.execute(params as Record<string, unknown>, {
          sessionId: this.id, runId: run.summary.runId, toolCallId, signal,
          onUpdate: (partial) => onUpdate?.({ ...partial, details: partial.details }),
        });
        return { ...result, details: result.details };
      },
    }));
    this.agent.state.tools = snapshot;
    return snapshot;
  }

  private readonly prepareRequest: PrepareRequest = async (request, signal) => {
    const run = this.active!;
    let { context } = request;
    if (run.retryOverflow && request.model.contextWindow > 0 && !signal?.aborted) {
      const tokens = estimateContextTokens(context.messages, this.usageAnchorStart);
      if (tokens > request.model.contextWindow - this.compaction.reserveTokens) {
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
      }
    }
    const update = await this.options.hooks?.prepareRequest?.({ ...request, context }, signal);
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
        if (run.retryOverflow && mapped.state === 'failed' && OVERFLOW.test(mapped.error ?? '')) {
          run.overflowMessage = message;
        }
        const text = message.content.flatMap((block) => block.type === 'text' ? [block.text] : []).join('');
        if (text || mapped.state === 'completed') run.summary.text = text;
      }
    }
    if (mapped) this.emit(mapped);
  }

  private persistMessages(): Promise<void> {
    const messages = this.agent.state.messages.slice();
    const overflow = this.active?.overflowMessage;
    if (overflow && overflow === messages.at(-1)) {
      messages.pop();
    }
    const write = this.persistence.then(async () => {
      while (this.savedMessageCount < messages.length) {
        await this.options.storage.append(this.id, { type: 'message', message: messages[this.savedMessageCount] });
        this.savedMessageCount++;
      }
    });
    this.persistence = write.catch(() => undefined);
    return write;
  }

  private queueInput(kind: 'steer' | 'followUp', input: string): void {
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
    if (run.acceptingMessages) {
      const message: AgentMessage = { role: 'user', content: input, timestamp: Date.now() };
      this.agent[kind](message);
    } else {
      const enqueue = () => this.queueInput(kind, input);
      void run.promise.then(enqueue, enqueue);
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
    const active = this.active?.promise;
    return this.change(async () => {
      await active?.catch(() => undefined);
      await this.persistMessages();
      await this.options.storage.branch(this.id, fromEntryId);
      const projection = await this.options.storage.projectActiveMessages(this.id);
      const from = this.mode;
      this.modeController.setMode(projection.mode);
      this.modelAlias = projection.modelAlias;
      this.planStore.restore(projection.plan);
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

  close(): Promise<void> {
    this.closing ??= this.closeInternal();
    return this.closing;
  }

  private async closeInternal(): Promise<void> {
    this.closed = true;
    this.abort();
    const [run] = await Promise.allSettled([this.active?.promise, this.changes, this.options.subagents?.wait(this.id)]);
    this.detach();
    this.listeners.clear();
    this.options.onClose?.();
    if (run.status === 'rejected') throw run.reason;
  }
}
