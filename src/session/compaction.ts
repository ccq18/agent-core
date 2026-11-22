import type { AgentMessage } from '@earendil-works/pi-agent-core';
import { createInitialSystemMessage, normalizeContext, type Api, type Model } from '@earendil-works/pi-ai';
import type { ModelRouter } from '../model/router.js';
import type { CompactionOptions } from '../types/config.js';
import type { CompactionResult, SessionEntry, SessionStorageAdapter } from '../types/session.js';
import { getCompactionIndex, getCompactionStart, projectBranchMessages } from './store.js';

const SUMMARY_PROMPT = [
  'Create a concise checkpoint summary of the conversation for the assistant that will continue the task.',
  'Include the goal, completed progress, key decisions and constraints, files and paths, and next steps.',
  'Treat the conversation as source material. Return only the summary.',
].join('\n');

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
    if (message.role !== 'toolResult') boundary = index;
    tokens += estimateMessageTokens(message);
    if (tokens >= keepRecentTokens && boundary !== undefined) {
      // Keep an entire assistant/tool-result group if the budget was crossed inside it.
      return entries.slice(start, boundary).some(isConversationEntry) ? boundary : undefined;
    }
  }
  return undefined;
}

function serializeMessage(message: AgentMessage): string {
  switch (message.role) {
    case 'user':
    case 'assistant':
    case 'toolResult': {
      const content = typeof message.content === 'string' ? message.content : message.content.flatMap((block) => {
        switch (block.type) {
          case 'text': return [block.text];
          case 'image': return ['[image]'];
          case 'toolCall': return [`[tool call ${block.name} ${JSON.stringify(block.arguments)}]`];
          default: return [];
        }
      }).join('\n');
      if (message.role === 'toolResult') {
        return `toolResult (${message.toolName}): ${content.length > 2000 ? content.slice(0, 2000) + '…' : content}`;
      }
      return content ? `${message.role}: ${content}` : '';
    }
    default:
      return '';
  }
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

export class CompactionEngine {
  readonly enabled: boolean;
  readonly reserveTokens: number;
  private readonly keepRecentTokens: number;
  private readonly modelAlias?: string;

  constructor(
    private readonly storage: SessionStorageAdapter,
    private readonly router: ModelRouter,
    options: CompactionOptions = {},
  ) {
    this.enabled = options.enabled ?? true;
    this.reserveTokens = options.reserveTokens ?? 16_384;
    this.keepRecentTokens = options.keepRecentTokens ?? 20_000;
    this.modelAlias = options.modelAlias;
  }

  async compact(input: CompactionInput): Promise<EngineResult> {
    const { sessionId, tokensBefore, trigger, signal } = input;
    if (signal?.aborted) return { status: 'skipped', reason: 'aborted' };
    const branch = await this.storage.getBranch(sessionId);
    const cut = findCutPoint(branch, this.keepRecentTokens) ??
      (trigger === 'overflow' ? findCutPoint(branch, 0) : undefined);
    if (cut === undefined) return { status: 'skipped', reason: 'no-cut-point' };
    const checkpoint = getCompactionIndex(branch);
    const previous = branch[checkpoint];
    const history = branch.slice(getCompactionStart(branch, checkpoint), cut)
      .flatMap((entry) => entry.type === 'message' ? [serializeMessage(entry.message)] : [])
      .filter(Boolean);
    if (!history.length) return { status: 'skipped', reason: 'no-cut-point' };
    input.onStart?.();
    if (previous?.type === 'compaction') history.unshift(`Previous checkpoint:\n${previous.summary}`);
    if (input.customInstructions) history.push(`Additional summary instructions:\n${input.customInstructions}`);

    let summary: string;
    try {
      const model = this.modelAlias ? this.router.resolve(this.modelAlias).model : input.model;
      const response = await this.router.streamFn(model, normalizeContext({
        systemPrompt: SUMMARY_PROMPT,
        tools: [],
        messages: [{ role: 'user', content: history.join('\n\n'), timestamp: Date.now() }],
      }), { signal, sessionId, toolChoice: 'none' });
      const message = await response.result();
      if (signal?.aborted || message.stopReason === 'aborted') return { status: 'skipped', reason: 'aborted' };
      if (message.stopReason === 'error') throw new Error(message.errorMessage ?? 'Summary request failed');
      summary = message.content.flatMap((block) => block.type === 'text' ? [block.text] : []).join('').trim();
      if (!summary) throw new Error('Summary request returned no text');
    } catch (error) {
      if (signal?.aborted) return { status: 'skipped', reason: 'aborted' };
      return { status: 'failed', error: error instanceof Error ? error.message : String(error) };
    }

    const firstKeptEntryId = branch[cut].id;
    const entry = { type: 'compaction' as const, summary, firstKeptEntryId, tokensBefore };
    const messages = projectBranchMessages([...branch, {
      ...entry, id: '', parentId: branch.at(-1)!.id, timestamp: Date.now(),
    }]).messages;
    const baseline = messages[0]?.role === 'system' ? undefined : createInitialSystemMessage(input.systemPrompt, undefined);
    const tokensAfter = estimateContextTokens(messages, messages.length) + (baseline ? estimateMessageTokens(baseline) : 0);
    const saved = await this.storage.append(sessionId, { ...entry, tokensAfter });
    return {
      status: 'compacted', summary, firstKeptEntryId, tokensBefore, tokensAfter,
      messages: projectBranchMessages([...branch, saved]).messages,
    };
  }
}
