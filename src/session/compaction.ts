import type { AgentMessage } from '@earendil-works/pi-agent-core';
import { createInitialSystemMessage, normalizeContext, type Api, type Model } from '@earendil-works/pi-ai';
import { raceWithAbortSignal } from '@earendil-works/pi-ai/utils/abort';
import { isContextOverflow } from '@earendil-works/pi-ai/utils/overflow';
import { DEFAULT_PROMPT_TEMPLATES } from '../harness/prompt.js';
import { isRuleUpdateMessage } from '../harness/rules.js';
import type { ModelRouter } from '../model/router.js';
import type { CompactionOptions, PromptTemplates } from '../types/config.js';
import type { CompactionResult, SessionCompactionWindow, SessionEntry, SessionStorageAdapter } from '../types/session.js';
import { getCompactionIndex, getCompactionStart, projectBranchMessages } from './store.js';

const SHRINK_RATIOS = [0.85, 0.70, 0.50, 0.35];
const RATE_LIMIT = /rate[ _-]*limit|too many requests|throttl/i;
const HTTP_429 = /(?:^\s*|\b(?:api|http(?:\/[\d.]+)?|(?:status|error)(?:[ _-]?code)?|code)["']?[\s:=(\[]+["']?)429\b/i;

export function isRateLimitError(message: string): boolean {
  return RATE_LIMIT.test(message) || HTTP_429.test(message);
}

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

function clip(text: string, limit: number, force = false): string {
  if (!force && Buffer.byteLength(text) <= limit) return text;
  if (limit < 3) return '';
  // UTF-8 byte length is monotonic across complete code points.
  let low = 0;
  let high = Math.min(text.length, limit - 3);
  while (low < high) {
    const mid = Math.ceil((low + high) / 2);
    if (Buffer.byteLength(text.slice(0, mid)) <= limit - 3) low = mid;
    else high = mid - 1;
  }
  const previous = text.charCodeAt(low - 1);
  const next = text.charCodeAt(low);
  if (previous >= 0xd800 && previous <= 0xdbff && next >= 0xdc00 && next <= 0xdfff) low--;
  return text.slice(0, low) + '…';
}

function serializeMessage(message: AgentMessage, stage: number, older: boolean, budget: number): string {
  switch (message.role) {
    case 'user':
    case 'assistant':
    case 'toolResult': {
      if (message.role === 'toolResult' && stage >= 2 && older) return '[toolResult omitted]';
      const prefix = message.role === 'toolResult' ? `toolResult (${message.toolName}): ` : `${message.role}: `;
      const limit = budget - Buffer.byteLength(prefix);
      if (limit <= 0) return clip(prefix, budget);
      const toolLimit = stage >= 1 ? 800 : 2000;
      const bodyLimit = message.role === 'toolResult' ? Math.min(limit, toolLimit + 3) : limit;
      const content = message.content;
      const singleText = typeof content === 'string' ? content
        : content.length === 1 && content[0].type === 'text' ? content[0].text : undefined;
      if (typeof content === 'string' || singleText !== undefined) {
        const value = singleText ?? '';
        if (!value && message.role !== 'toolResult') return '';
        if (message.role === 'toolResult' && Buffer.byteLength(value) > toolLimit && limit >= toolLimit + 3) {
          return prefix + clip(value, toolLimit + 3, true);
        }
        return prefix + clip(value, bodyLimit);
      }
      const parts: string[] = [];
      let remaining = bodyLimit;
      for (const block of content) {
        let part: string;
        switch (block.type) {
          case 'text': part = block.text; break;
          case 'image': part = '[image]'; break;
          case 'toolCall': {
            const args = JSON.stringify(block.arguments);
            part = `[tool call ${block.name} ${clip(args, stage >= 2 ? 300 : remaining)}]`;
            break;
          }
          default: continue;
        }
        if (remaining <= 0) break;
        const kept = clip(part, remaining);
        if (kept) parts.push(kept);
        remaining -= Buffer.byteLength(kept) + 1;
        if (kept !== part) break;
      }
      const body = parts.join('\n');
      return body || message.role === 'toolResult' ? prefix + body : '';
    }
    default:
      return '';
  }
}

function linesSize(lines: readonly string[]): number {
  return lines.reduce((size, line) => size + Buffer.byteLength(line), Math.max(0, (lines.length - 1) * 2));
}

/** Share the remaining budget across priority lines, keeping shorter lines intact. */
function fitLines(lines: string[], budget: number): string {
  if (linesSize(lines) <= budget) return lines.join('\n\n');
  let available = budget - 2 * (lines.length - 1);
  if (available < lines.length * 3) {
    if (lines.length <= 2 || budget < 8) return clip(lines[0] ?? '', budget);
    return fitLines([lines[0], lines.at(-1)!], budget);
  }
  let remaining = lines.length;
  let cap = available;
  for (const length of lines.map((line) => Buffer.byteLength(line)).sort((left, right) => left - right)) {
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
  const isRequest = (message: AgentMessage) => message.role === 'user' && !isRuleUpdateMessage(message);
  let latest = 0;
  let lastTool = -1;
  for (let index = messages.length - 1; index >= 0; index--) {
    if (lastTool < 0 && messages[index].role === 'toolResult') lastTool = index;
    // If the cut falls immediately after a user message, also keep the preceding exchange.
    if (index < messages.length - 1 && isRequest(messages[index])) {
      latest = index;
      break;
    }
  }
  const olderThan = Math.max(latest, lastTool);
  const serialize = (message: AgentMessage, index: number) => serializeMessage(message, stage, index < olderThan, budget);
  const head = previousSummary === undefined ? [] : [`Previous checkpoint:\n${clip(previousSummary, budget)}`];
  const tail = instructions ? [`Additional summary instructions:\n${clip(instructions, budget)}`] : [];
  const newest = messages.slice(latest).map((message, index) => serialize(message, latest + index)).filter(Boolean);
  if (stage === 4) {
    const recent = newest.length > 4 ? [newest[0], ...newest.slice(-3)] : newest;
    return fitLines([...head, ...recent.map((line) => clip(line, 320)), ...tail], budget);
  }

  const priority = [...head, ...newest, ...tail];
  let remaining = budget - linesSize(priority);
  if (remaining <= 0) return fitLines(priority, budget);
  const older: string[] = [];
  let exchange: string[] = [];
  let exchangeSize = 0;
  for (let index = latest - 1; index >= 0; index--) {
    const line = serialize(messages[index], index);
    if (line) {
      exchangeSize += Buffer.byteLength(line) + 2;
      if (stage === 0 && exchangeSize > remaining) break;
      exchange.push(line);
    }
    if (index === 0 || isRequest(messages[index])) {
      if (exchangeSize > remaining) {
        if (remaining > 2) {
          const reduced = fitLines(exchange.reverse(), remaining - 2);
          if (reduced) older.push(reduced);
        }
        break;
      }
      for (const line of exchange) older.push(line);
      remaining -= exchangeSize;
      exchange = [];
      exchangeSize = 0;
    }
  }
  return [...head, ...older.reverse(), ...newest, ...tail].join('\n\n');
}

async function drainSummary(response: Awaited<ReturnType<ModelRouter['streamFn']>>) {
  for await (const _event of response) {
    // Consume intermediate events as they arrive. Only the final message is used.
  }
  return response.result();
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
    const { entries: branch, leafId } = await this.getWindow(sessionId);
    const cut = findCutPoint(branch, this.keepRecentTokens);
    if (cut === undefined || leafId === null || signal.aborted) return undefined;
    const source = summarySource(branch, cut);
    const summaryModel = this.resolveModel(model);
    const result = await this.summarize(sessionId, summaryModel, source, signal);
    if (result.status !== 'summarized' || signal.aborted) return undefined;
    return {
      leafId, previousId: source.previousId, firstKeptEntryId: source.firstKeptEntryId,
      model: summaryModel, summary: result.summary,
    };
  }

  private async getWindow(sessionId: string): Promise<SessionCompactionWindow> {
    if (this.storage.getCompactionWindow) return this.storage.getCompactionWindow(sessionId);
    const entries = await this.storage.getBranch(sessionId);
    const first = entries[0];
    return {
      entries, leafId: entries.at(-1)?.id ?? null,
      systemBaseline: first?.type === 'message' && first.message.role === 'system' ? first.message : undefined,
    };
  }

  private resolveModel(model: Model<Api>): Model<Api> {
    return this.modelAlias ? this.router.resolve(this.modelAlias).model : model;
  }

  private async summarize(
    sessionId: string, model: Model<Api>, source: SummarySource, signal?: AbortSignal, instructions?: string,
  ): Promise<SummaryResult> {
    const system = this.prompts.compactionSummary;
    const protocolReserve = Math.max(64, Math.min(512, Math.ceil(model.contextWindow * 0.05)));
    const available = model.contextWindow - protocolReserve - Math.ceil(Buffer.byteLength(system) / 4);
    if (available < 2) return { status: 'failed', error: 'Summary model context has no room for summary input and output' };
    // Normally reserve the configured limit. Leave room for input on models whose output ceiling spans the context.
    const maxTokens = model.maxTokens < available ? model.maxTokens : Math.floor(available / 2);
    const budget = (available - maxTokens) * 4;
    const fullHistory = summaryHistory(source, instructions, 0, budget);
    if (!fullHistory) return { status: 'skipped', reason: 'no-cut-point' };
    let content = fullHistory;
    for (let attempt = 0; ; attempt++) {
      if (signal?.aborted) return { status: 'skipped', reason: 'aborted' };
      try {
        const response = await this.router.streamFn(model, normalizeContext({
          systemPrompt: system,
          tools: [],
          messages: [{ role: 'user', content, timestamp: Date.now() }],
        }), { signal, sessionId, toolChoice: 'none', maxTokens });
        const result = drainSummary(response);
        const message = await (signal ? raceWithAbortSignal(result, signal) : result);
        if (signal?.aborted || message.stopReason === 'aborted') return { status: 'skipped', reason: 'aborted' };
        if (!isRateLimitError(message.errorMessage ?? '') && isContextOverflow(message, model.contextWindow)) {
          if (attempt === SHRINK_RATIOS.length) {
            return { status: 'failed', error: message.errorMessage ?? 'Summary request exceeded the context window' };
          }
          const budget = Math.max(0, Math.min(Buffer.byteLength(content) - 1, Math.floor(Buffer.byteLength(fullHistory) * SHRINK_RATIOS[attempt])));
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
    const { entries: branch, leafId, systemBaseline } = await this.getWindow(sessionId);
    const candidate = trigger === 'auto' && !input.customInstructions ? this.candidates.get(sessionId)?.result : undefined;
    this.clearPrecompute(sessionId);
    if (signal?.aborted) return { status: 'skipped', reason: 'aborted' };

    const cut = findCutPoint(branch, this.keepRecentTokens) ??
      (trigger === 'overflow' ? findCutPoint(branch, 0) : undefined);
    if (cut === undefined || leafId === null) return { status: 'skipped', reason: 'no-cut-point' };
    const source = summarySource(branch, cut);
    if (!source.messages.length) {
      return { status: 'skipped', reason: 'no-cut-point' };
    }
    input.onStart?.();

    const entryFor = (summary: string, firstKeptEntryId: string) =>
      ({ type: 'compaction' as const, summary, firstKeptEntryId, tokensBefore });
    const estimateAfter = (entry: ReturnType<typeof entryFor>) => {
      const messages = projectBranchMessages([...branch, {
        ...entry, id: '', parentId: leafId, timestamp: Date.now(),
      }], systemBaseline ?? null).messages;
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
    const tokensAfter = estimateAfter(entry);
    if (signal?.aborted) return { status: 'skipped', reason: 'aborted' };
    let saved: SessionEntry | undefined;
    if (this.storage.appendCompaction) {
      saved = await this.storage.appendCompaction(sessionId, leafId, { ...entry, tokensAfter }, signal);
    } else {
      const current = await this.getWindow(sessionId);
      if (current.leafId === leafId && current.entries.some(({ id }) => id === entry.firstKeptEntryId) && !signal?.aborted) {
        saved = await this.storage.append(sessionId, { ...entry, tokensAfter });
      }
    }
    if (!saved) return signal?.aborted ? { status: 'skipped', reason: 'aborted' }
      : { status: 'failed', error: 'Session branch changed during compaction' };
    return {
      status: 'compacted', summary: entry.summary, firstKeptEntryId: entry.firstKeptEntryId, tokensBefore, tokensAfter,
      messages: projectBranchMessages([...branch, saved], systemBaseline ?? null).messages,
    };
  }
}
