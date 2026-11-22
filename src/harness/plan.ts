import type { PlanItem, SessionStorageAdapter } from '../types/session.js';
import type { CoreToolDefinition } from '../types/tool.js';
import { DEFAULT_PROMPT_TEMPLATES } from './prompt.js';

export class PlanStore {
  private items: PlanItem[];

  constructor(
    private readonly sessionId: string,
    private readonly storage: SessionStorageAdapter,
    items: PlanItem[],
    private readonly onUpdate?: (items: PlanItem[], runId?: string) => void,
    private readonly persistTranscript?: () => Promise<void>,
  ) {
    this.items = structuredClone(items);
  }

  getItems(): PlanItem[] {
    return structuredClone(this.items);
  }

  restore(items: PlanItem[]): void {
    this.items = structuredClone(items);
  }

  async update(items: PlanItem[], runId?: string): Promise<void> {
    const snapshot = structuredClone(items);
    await this.persistTranscript?.();
    await this.storage.append(this.sessionId, { type: 'plan', items: snapshot });
    this.items = snapshot;
    this.onUpdate?.(this.getItems(), runId);
  }
}

export function createPlanTools(
  getStore: (sessionId: string) => PlanStore | Promise<PlanStore>,
  description = DEFAULT_PROMPT_TEMPLATES.todoWriteDescription,
): CoreToolDefinition[] {
  return [
    {
      name: 'todowrite',
      label: 'Update plan',
      description,
      readOnly: true,
      executionMode: 'sequential',
      parameters: {
        type: 'object',
        properties: {
          items: {
            type: 'array',
            items: {
              type: 'object',
              properties: {
                id: { type: 'string' }, content: { type: 'string' },
                status: { type: 'string', enum: ['pending', 'in_progress', 'completed', 'cancelled'] },
                priority: { type: 'string', enum: ['low', 'medium', 'high'] },
              },
              required: ['id', 'content', 'status'],
            },
          },
        },
        required: ['items'],
      },
      async execute(params, context) {
        const items = params.items as PlanItem[];
        const store = await getStore(context.sessionId);
        await store.update(items, context.runId);
        const completed = items.filter(({ status }) => status === 'completed').length;
        const pending = items.filter(({ status }) => status === 'pending').length;
        const inProgress = items.filter(({ status }) => status === 'in_progress').map(({ content }) => content);
        const text = [
          `Plan updated. Total: ${items.length}, completed: ${completed}, in progress: ${inProgress.length}, pending: ${pending}.`,
          ...inProgress.map((content) => `- ${content}`),
        ].join('\n');
        return { content: [{ type: 'text', text }] };
      },
    },
    {
      name: 'todoread',
      label: 'Read plan',
      description: 'Read the current session plan and task statuses.',
      readOnly: true,
      parameters: { type: 'object', properties: {} },
      async execute(_params, context) {
        const store = await getStore(context.sessionId);
        return { content: [{ type: 'text', text: JSON.stringify(store.getItems(), null, 2) }] };
      },
    },
  ];
}
