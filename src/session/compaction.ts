import type { AgentMessage } from '@earendil-works/pi-agent-core';
import { createInitialSystemMessage, normalizeContext, type Api, type Model } from '@earendil-works/pi-ai';
import { raceWithAbortSignal } from '@earendil-works/pi-ai/utils/abort';
import { isContextOverflow } from '@earendil-works/pi-ai/utils/overflow';
import { DEFAULT_PROMPT_TEMPLATES } from '../harness/prompt.js';
import { isRuleUpdateMessage } from '../harness/rules.js';
import type { ModelRouter } from '../model/router.js';
import type { CompactionOptions, PromptTemplates } from '../types/config.js';
import type { CompactionResult, SessionEntry, SessionStorageAdapter } from '../types/session.js';
import { getCompactionIndex, getCompactionStart, projectBranchMessages } from './store.js';

const SHRINK_RATIOS = [0.85, 0.70, 0.50, 0.35];

const estimateText = (text: string) => Math.ceil(text.length / 4);

export function estimateMessageTokens(message: AgentMessage): number {
  switch (message.role) {
    case 'system':
    case 'user':
    case 'assistant':
    case 'toolResult':
      if (typeof message.content === 'string') return estimateText(message.content);
      return message.content.reduce((total, block) => {
        switch (block.type) {
          case 'text': return total + estimateText(block.text);
          case 'thinking': return total + estimateText(block.thinking);
          case 'image': return total + 1200;
          case 'toolCall': return total + estimateText(JSON.stringify(block.arguments));
        }
      }, 0);
    default:
      return 0;
  }
}

export function estimateContextTokens(messages: readonly AgentMessage[], usageAnchorStart = 0): number {
  let estimated = 0;
  for (let index = messages.length - 1; index >= 0; index--) {
    const message = messages[index];
    if (index >= usageAnchorStart && message.role === 'assistant' &&
      message.stopReason !== 'error' && message.stopReason !== 'aborted' &&
      Number.isFinite(message.usage?.totalTokens) && message.usage.totalTokens > 0) {
      return estimated + message.usage.totalTokens;
    }
    estimated += estimateMessageTokens(message);
  }
  return estimated;
}

function isConversationEntry(entry: SessionEntry): entry is Extract<SessionEntry, { type: 'message' }> {
  return entry.type === 'message' &&
    (entry.message.role === 'user' || entry.message.role === 'assistant' || entry.message.role === 'toolResult');
}

export function findCutPoint(entries: readonly SessionEntry[], keepRecentTokens: number): number | undefined {
  const checkpoint = getCompactionIndex(entries);
  const start = getCompactionStart(entries, checkpoint);
  let tokens = 0;
  let boundary: number | undefined;

  for (let index = entries.length - 1; index >= start; index--) {
    const entry = entries[index];
    if (!isConversationEntry(entry)) continue;
    const { message } = entry;
    if (message.role !== 'toolResult' && !isRuleUpdateMessage(message)) boundary = index;
    tokens += estimateMessageTokens(message);
    if (tokens >= keepRecentTokens && boundary !== undefined) {
      // Keep an entire assistant/tool-result group if the budget was crossed inside it.
      return entries.slice(start, boundary).some(isConversationEntry) ? boundary : undefined;
    }
  }
  return undefined;
}

function clip(text: string, limit: number): string {
  if (text.length <= limit) return text;
  if (limit <= 0) return '';
  if (limit === 1) return '…';
  return text.slice(0, limit - 1) + '…';
}

function serializeMessage(message: AgentMessage, stage: number, older: boolean): string {
  switch (message.role) {
    case 'user':
    case 'assistant':
    case 'toolResult': {
      if (message.role === 'toolResult' && stage >= 2 && older) return '[toolResult omitted]';
      const content = typeof message.content === 'string' ? message.content : message.content.flatMap((block) => {
        switch (block.type) {
          case 'text': return [block.text];
          case 'image': return ['[image]'];
          case 'toolCall': {
            const args = JSON.stringify(block.arguments);
            return [`[tool call ${block.name} ${stage >= 2 ? clip(args, 300) : args}]`];
          }
          default: return [];
        }
      }).join('\n');
      if (message.role === 'toolResult') {
        const limit = stage >= 1 ? 800 : 2000;
        return `toolResult (${message.toolName}): ${content.length > limit ? content.slice(0, limit) + '…' : content}`;
      }
      return content ? `${message.role}: ${content}` : '';
    }
    default:
      return '';
  }
}

/** Share the character budget across long lines and leave short lines intact. */
function fitLines(lines: string[], budget: number): string {
  const joined = lines.join('\n\n');
  if (joined.length <= budget) return joined;
  let available = budget - 2 * (lines.length - 1);
  if (available < lines.length) {
    // Extremely small budgets may not even fit the separators. Keep the ends.
    if (lines.length <= 2 || budget < 4) return clip(joined, budget);
    return fitLines([lines[0], lines.at(-1)!], budget);
  }
  let remaining = lines.length;
  let cap = available;
  for (const length of lines.map((line) => line.length).sort((left, right) => left - right)) {
    cap = Math.floor(available / remaining);
    if (length > cap) break;
    remaining--;
    available -= length;
  }
  return lines.map((line) => clip(line, cap)).join('\n\n');
}

interface SummarySource {
  messages: AgentMessage[];
  previousSummary?: string;
  previousId?: string;
  firstKeptEntryId: string;
}

function summarySource(branch: readonly SessionEntry[], cut: number): SummarySource {
  const checkpoint = getCompactionIndex(branch);
  const previous = branch[checkpoint];
  return {
    messages: branch.slice(getCompactionStart(branch, checkpoint), cut)
      .filter(isConversationEntry).map(({ message }) => message),
    previousSummary: previous?.type === 'compaction' ? previous.summary : undefined,
    previousId: previous?.id,
    firstKeptEntryId: branch[cut].id,
  };
}

function summaryHistory(source: SummarySource, instructions: string | undefined, stage: number, budget: number): string {
  const { messages, previousSummary } = source;
  let latest = messages.length - 1;
  while (latest > 0 && (messages[latest].role !== 'user' || isRuleUpdateMessage(messages[latest]))) latest--;
  let lastToolResult = messages.length - 1;
  while (lastToolResult >= 0 && messages[lastToolResult].role !== 'toolResult') lastToolResult--;
  const olderThan = Math.max(latest, lastToolResult);
  const history = messages.map((message, index) => serializeMessage(message, stage, index < olderThan));
  const head = previousSummary === undefined ? [] : [`Previous checkpoint:\n${previousSummary}`];
  const tail = instructions ? [`Additional summary instructions:\n${instructions}`] : [];
  if (stage < 3) return fitLines([...head, ...history.filter(Boolean), ...tail], budget);

  const newest = history.slice(latest).filter(Boolean);
  if (stage === 4) {
    const recent = newest.length > 4 ? [newest[0], ...newest.slice(-3)] : newest;
    return fitLines([...head, ...recent.map((line) => clip(line, 320)), ...tail], budget);
  }

  let remaining = budget - [...head, ...newest, ...tail].join('\n\n').length;
  const kept: string[] = [];
  for (const line of history.slice(0, latest).filter(Boolean).reverse()) {
    remaining -= line.length + 2;
    if (remaining < 0) break;
    kept.push(line);
  }
  return fitLines([...head, ...kept.reverse(), ...newest, ...tail], budget);
}

export type CompactionTrigger = 'auto' | 'overflow' | 'manual';

interface CompactionInput {
  sessionId: string;
  model: Model<Api>;
  tokensBefore: number;
  trigger: CompactionTrigger;
  systemPrompt?: string;
  customInstructions?: string;
  signal?: AbortSignal;
  onStart?(): void;
}

type EngineResult = Exclude<CompactionResult, { status: 'compacted' }> | (
  Extract<CompactionResult, { status: 'compacted' }> & { messages: AgentMessage[] }
);

type SummaryResult = Exclude<CompactionResult, { status: 'compacted' }> | { status: 'summarized'; summary: string };

interface PrecomputedSummary {
  leafId: string;
  previousId?: string;
  firstKeptEntryId: string;
  model: Model<Api>;
  summary: string;
}

interface Precomputation {
  controller: AbortController;
  result?: PrecomputedSummary;
}

export class CompactionEngine {
  readonly enabled: boolean;
  readonly reserveTokens: number;
  readonly precomputeEnabled: boolean;
  private readonly keepRecentTokens: number;
  private readonly modelAlias?: string;
  private readonly precomputeLeadTokens: number;
  private readonly candidates = new Map<string, Precomputation>();

  constructor(
    private readonly storage: SessionStorageAdapter,
    private readonly router: ModelRouter,
    options: CompactionOptions = {},
    private readonly prompts: Pick<PromptTemplates, 'compactionSummary'> = DEFAULT_PROMPT_TEMPLATES,
  ) {
    this.enabled = options.enabled ?? true;
    this.reserveTokens = options.reserveTokens ?? 16_384;
    this.keepRecentTokens = options.keepRecentTokens ?? 20_000;
    this.modelAlias = options.modelAlias;
    this.precomputeEnabled = this.enabled && (options.precompute ?? false);
    this.precomputeLeadTokens = options.precomputeLeadTokens ?? 20_000;
  }

  maybePrecompute(sessionId: string, model: Model<Api>, tokens: number): void {
    if (!this.precomputeEnabled || model.contextWindow <= 0 || this.candidates.has(sessionId)) return;
    const threshold = model.contextWindow - this.reserveTokens;
    if (tokens < threshold - this.precomputeLeadTokens || tokens >= threshold) return;

    const pending: Precomputation = { controller: new AbortController() };
    this.candidates.set(sessionId, pending);
    void this.computeCandidate(sessionId, model, pending.controller.signal)
      .catch(() => undefined)
      .then((result) => {
        if (this.candidates.get(sessionId) !== pending) return;
        if (result) pending.result = result;
        else this.candidates.delete(sessionId);
      });
  }

  clearPrecompute(sessionId: string): void {
    const pending = this.candidates.get(sessionId);
    this.candidates.delete(sessionId);
    pending?.controller.abort();
  }

  private async computeCandidate(sessionId: string, model: Model<Api>, signal: AbortSignal): Promise<PrecomputedSummary | undefined> {
    const branch = await this.storage.getBranch(sessionId);
    const cut = findCutPoint(branch, this.keepRecentTokens);
    if (cut === undefined || signal.aborted) return undefined;
    const source = summarySource(branch, cut);
    const summaryModel = this.resolveModel(model);
    const result = await this.summarize(sessionId, summaryModel, source, signal);
    if (result.status !== 'summarized' || signal.aborted) return undefined;
    return {
      leafId: branch.at(-1)!.id, previousId: source.previousId, firstKeptEntryId: source.firstKeptEntryId,
      model: summaryModel, summary: result.summary,
    };
  }

  private resolveModel(model: Model<Api>): Model<Api> {
    return this.modelAlias ? this.router.resolve(this.modelAlias).model : model;
  }

  private async summarize(
    sessionId: string, model: Model<Api>, source: SummarySource, signal?: AbortSignal, instructions?: string,
  ): Promise<SummaryResult> {
    const fullHistory = summaryHistory(source, instructions, 0, Infinity);
    if (!fullHistory) return { status: 'skipped', reason: 'no-cut-point' };
    let content = fullHistory;
    for (let attempt = 0; ; attempt++) {
      if (signal?.aborted) return { status: 'skipped', reason: 'aborted' };
      try {
        const response = await this.router.streamFn(model, normalizeContext({
          systemPrompt: this.prompts.compactionSummary,
          tools: [],
          messages: [{ role: 'user', content, timestamp: Date.now() }],
        }), { signal, sessionId, toolChoice: 'none' });
        const result = response.result();
        const message = await (signal ? raceWithAbortSignal(result, signal) : result);
        if (signal?.aborted || message.stopReason === 'aborted') return { status: 'skipped', reason: 'aborted' };
        if (isContextOverflow(message, model.contextWindow)) {
          if (attempt === SHRINK_RATIOS.length) {
            return { status: 'failed', error: message.errorMessage ?? 'Summary request exceeded the context window' };
          }
          const budget = Math.max(0, Math.min(content.length - 1, Math.floor(fullHistory.length * SHRINK_RATIOS[attempt])));
          content = summaryHistory(source, instructions, attempt + 1, budget);
          continue;
        }
        if (message.stopReason === 'error') throw new Error(message.errorMessage ?? 'Summary request failed');
        const summary = message.content.flatMap((block) => block.type === 'text' ? [block.text] : []).join('').trim();
        if (!summary) throw new Error('Summary request returned no text');
        return { status: 'summarized', summary };
      } catch (error) {
        if (signal?.aborted) return { status: 'skipped', reason: 'aborted' };
        return { status: 'failed', error: error instanceof Error ? error.message : String(error) };
      }
    }
  }

  async compact(input: CompactionInput): Promise<EngineResult> {
    const { sessionId, tokensBefore, trigger, signal } = input;
    if (signal?.aborted) return { status: 'skipped', reason: 'aborted' };
    const branch = await this.storage.getBranch(sessionId);
    const candidate = trigger === 'auto' && !input.customInstructions ? this.candidates.get(sessionId)?.result : undefined;
    this.clearPrecompute(sessionId);
    if (signal?.aborted) return { status: 'skipped', reason: 'aborted' };

    const cut = findCutPoint(branch, this.keepRecentTokens) ??
      (trigger === 'overflow' ? findCutPoint(branch, 0) : undefined);
    if (cut === undefined) return { status: 'skipped', reason: 'no-cut-point' };
    const source = summarySource(branch, cut);
    if (!source.messages.some((message) => serializeMessage(message, 0, false))) {
      return { status: 'skipped', reason: 'no-cut-point' };
    }
    input.onStart?.();

    const entryFor = (summary: string, firstKeptEntryId: string) =>
      ({ type: 'compaction' as const, summary, firstKeptEntryId, tokensBefore });
    const estimateAfter = (entry: ReturnType<typeof entryFor>) => {
      const messages = projectBranchMessages([...branch, {
        ...entry, id: '', parentId: branch.at(-1)!.id, timestamp: Date.now(),
      }]).messages;
      const baseline = messages[0]?.role === 'system' ? undefined : createInitialSystemMessage(input.systemPrompt, undefined);
      return estimateContextTokens(messages, messages.length) + (baseline ? estimateMessageTokens(baseline) : 0);
    };

    let model: Model<Api>;
    try {
      model = this.resolveModel(input.model);
    } catch (error) {
      return { status: 'failed', error: error instanceof Error ? error.message : String(error) };
    }

    let entry: ReturnType<typeof entryFor> | undefined;
    if (candidate && candidate.model === model && candidate.previousId === source.previousId) {
      const anchor = branch.findIndex(({ id }) => id === candidate.leafId);
      const cachedCut = branch.findIndex(({ id }) => id === candidate.firstKeptEntryId);
      if (cachedCut >= 0 && cachedCut <= cut && anchor >= cachedCut) {
        const cached = entryFor(candidate.summary, candidate.firstKeptEntryId);
        // On an extending branch, the earlier cut can retain extra turns if they still fit.
        if (cachedCut === cut || estimateAfter(cached) < input.model.contextWindow - this.reserveTokens) entry = cached;
      }
    }
    if (!entry) {
      const result = await this.summarize(sessionId, model, source, signal, input.customInstructions);
      if (result.status !== 'summarized') return result;
      entry = entryFor(result.summary, source.firstKeptEntryId);
    }
    if (signal?.aborted) return { status: 'skipped', reason: 'aborted' };
    const tokensAfter = estimateAfter(entry);
    const saved = await this.storage.append(sessionId, { ...entry, tokensAfter });
    return {
      status: 'compacted', summary: entry.summary, firstKeptEntryId: entry.firstKeptEntryId, tokensBefore, tokensAfter,
      messages: projectBranchMessages([...branch, saved]).messages,
    };
  }
}
