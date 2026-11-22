import type { AgentMessage } from '@earendil-works/pi-agent-core';
import type { PromptTemplates } from '../types/config.js';
import type { PermissionRule } from '../types/permission.js';
import type { AgentMode } from '../types/session.js';
import type { CoreToolDefinition } from '../types/tool.js';
import { DEFAULT_PROMPT_TEMPLATES, wrapSystemUpdate } from './prompt.js';

const PLAN_READ: readonly PermissionRule[] = [{ action: '*', resource: '*', effect: 'allow' }];
const PLAN_WRITE: readonly PermissionRule[] = [{ action: '*', resource: '*', effect: 'deny' }];
export class ModeController {
  private shouldRemind: boolean;

  constructor(
    private currentMode: AgentMode,
    changed = false,
    private readonly prompts: Pick<PromptTemplates, 'planModeEnter' | 'buildModeEnter'> = DEFAULT_PROMPT_TEMPLATES,
  ) {
    this.shouldRemind = changed || currentMode === 'plan';
  }

  get mode(): AgentMode {
    return this.currentMode;
  }

  setMode(mode: AgentMode): void {
    if (this.mode !== mode) this.shouldRemind = true;
    this.currentMode = mode;
  }

  blocks(tool: Pick<CoreToolDefinition, 'readOnly'>): boolean {
    return this.mode === 'plan' && tool.readOnly !== true;
  }

  rulesFor(tool: Pick<CoreToolDefinition, 'readOnly'>): readonly PermissionRule[] {
    if (this.mode === 'build') return [];
    return this.blocks(tool) ? PLAN_WRITE : PLAN_READ;
  }

  transformContext(messages: AgentMessage[]): AgentMessage[] {
    if (!this.shouldRemind) return messages;
    const template = this.mode === 'plan' ? this.prompts.planModeEnter : this.prompts.buildModeEnter;
    if (!template.trim()) return messages;
    const content = wrapSystemUpdate(template);
    const tail = messages.at(-1);
    if (tail?.role === 'user' && tail.content === content) return messages;
    const index = tail?.role === 'user' ? messages.length - 1 : messages.length;
    const preceding = messages[index - 1];
    if (preceding?.role === 'user' && preceding.content === content) return messages;
    const updated = messages.slice();
    updated.splice(index, 0, { role: 'user', content, timestamp: Date.now() });
    return updated;
  }
}
