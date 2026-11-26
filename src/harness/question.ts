import { randomUUID } from 'node:crypto';
import type { AgentCoreEventData } from '../types/events.js';
import type { PendingQuestionRequest, QuestionAnswer, QuestionItem } from '../types/question.js';
import type { CoreToolDefinition, ToolExecutionContext } from '../types/tool.js';
import { DEFAULT_PROMPT_TEMPLATES } from './prompt.js';

const DEFAULT_DESCRIPTION = DEFAULT_PROMPT_TEMPLATES.questionDescription +
  ' Call question on its own when you need user input. Mixing it with slow tools in the same batch makes the whole batch run sequentially.';

type QuestionReply = QuestionAnswer | { cancelled: true };
type QuestionEvent = Extract<AgentCoreEventData, { type: 'question_requested' | 'question_resolved' }>;
type QuestionResult = { cancelled: true } | { cancelled: false; answers: QuestionAnswer };

interface PendingQuestion {
  request: PendingQuestionRequest;
  finish(result: QuestionResult): void;
}

function isCancellation(reply: QuestionReply): reply is { cancelled: true } {
  return reply.cancelled === true;
}

export class QuestionManager {
  private readonly pending = new Map<string, PendingQuestion>();
  readonly tool: CoreToolDefinition = {
    name: 'question',
    label: 'Ask the user',
    description: DEFAULT_DESCRIPTION,
    permissionAction: 'question',
    readOnly: true,
    executionMode: 'sequential',
    parameters: {
      type: 'object',
      required: ['questions'],
      properties: {
        questions: {
          type: 'array',
          items: {
            type: 'object',
            required: ['id', 'question'],
            properties: {
              id: { type: 'string' },
              question: { type: 'string' },
              multiSelect: { type: 'boolean' },
              options: {
                type: 'array',
                items: {
                  type: 'object',
                  required: ['label'],
                  properties: { label: { type: 'string' }, description: { type: 'string' } },
                },
              },
            },
          },
        },
      },
    },
    execute: async (params, context) => {
      const result = await this.ask(params.questions as QuestionItem[], context);
      return { content: [{ type: 'text', text: JSON.stringify(result) }], structuredContent: result };
    },
  };

  constructor(
    private readonly emit: (request: PendingQuestionRequest, event: QuestionEvent) => void,
    description = DEFAULT_DESCRIPTION,
  ) {
    this.tool.description = description === DEFAULT_PROMPT_TEMPLATES.questionDescription ? DEFAULT_DESCRIPTION : description;
  }

  getPendingQuestions(sessionId: string): PendingQuestionRequest[] {
    return Array.from(this.pending.values())
      .filter(({ request }) => request.sessionId === sessionId)
      .map(({ request }) => structuredClone(request));
  }

  replyQuestion(sessionId: string, requestId: string, reply: QuestionReply): void {
    const pending = this.pending.get(requestId);
    if (pending?.request.sessionId !== sessionId) return;
    pending.finish(isCancellation(reply) ? { cancelled: true } : {
      cancelled: false, answers: structuredClone(reply),
    });
  }

  private ask(questions: QuestionItem[], { sessionId, runId, toolCallId, signal }: ToolExecutionContext): Promise<QuestionResult> {
    if (signal.aborted) return Promise.resolve({ cancelled: true });
    return new Promise((resolve) => {
      const request = { id: randomUUID(), sessionId, runId, toolCallId, questions: structuredClone(questions), createdAt: Date.now() };
      const finish = (result: QuestionResult) => {
        if (!this.pending.delete(request.id)) return;
        signal.removeEventListener('abort', abort);
        resolve(result);
        this.emit(request, { type: 'question_resolved', requestId: request.id, ...structuredClone(result) });
      };
      const abort = () => finish({ cancelled: true });
      this.pending.set(request.id, { request, finish });
      signal.addEventListener('abort', abort, { once: true });
      this.emit(request, { type: 'question_requested', request: structuredClone(request) });
    });
  }
}
