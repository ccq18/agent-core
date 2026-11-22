import type { AgentMessage } from '@earendil-works/pi-agent-core';
import type { PermissionRule } from '../types/permission.js';
import type { AgentMode } from '../types/session.js';
import type { CoreToolDefinition } from '../types/tool.js';

const PLAN_READ: readonly PermissionRule[] = [{ action: '*', resource: '*', effect: 'allow' }];
const PLAN_WRITE: readonly PermissionRule[] = [{ action: '*', resource: '*', effect: 'deny' }];
const modeUpdates: Record<AgentMode, string> = {
  plan: '<system-update>\nPlan mode is active. Inspect facts and make a plan without modifying files or performing side effects. Use todowrite to maintain the plan.\n</system-update>',
  build: '<system-update>\nBuild mode is active. You can proceed with the plan using the available tools.\n</system-update>',
};

export class ModeController {
  private shouldRemind: boolean;

  constructor(private currentMode: AgentMode, changed = false) {
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
    const content = modeUpdates[this.mode];
    const tail = messages.at(-1);
    if (tail?.role === 'user' && tail.content === content) return messages;
    return [...messages, { role: 'user', content, timestamp: Date.now() }];
  }
}
