import { randomUUID } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { mkdir, rename, unlink, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import type { BeforeToolCallResult } from '@earendil-works/pi-agent-core';
import type { ApprovalDecision, PendingApprovalRequest, PermissionEffect, PermissionRule } from '../types/permission.js';
import type { CoreToolDefinition } from '../types/tool.js';

export type PermissionRulesets = readonly (readonly PermissionRule[])[];
type ToolPermission = Pick<CoreToolDefinition, 'name' | 'permissionAction'>;
const PATH_ACTIONS = new Set(['read', 'edit', 'write', 'external_directory', 'glob', 'grep', 'list']);
const CASE_INSENSITIVE_PATHS = process.platform === 'darwin' || process.platform === 'win32';

export function matchWildcard(pattern: string, value: string, ignoreCase = false): boolean {
  const glob = Array.from(ignoreCase ? pattern.toLowerCase() : pattern);
  const input = Array.from(ignoreCase ? value.toLowerCase() : value);
  let index = 0;
  let token = 0;
  let star = -1;
  let starEnd = 0;
  while (index < input.length) {
    if (glob[token] === '*') {
      star = token++;
      starEnd = index;
    } else if (glob[token] === '?' || glob[token] === input[index]) {
      token++;
      index++;
    } else if (star >= 0) {
      token = star + 1;
      index = ++starEnd;
    } else {
      return false;
    }
  }
  while (glob[token] === '*') token++;
  return token === glob.length;
}

function actionOf(tool: ToolPermission): string {
  return tool.permissionAction ?? tool.name;
}

function matchesResource(action: string, pattern: string, resource: string): boolean {
  return matchWildcard(pattern, resource, CASE_INSENSITIVE_PATHS && PATH_ACTIONS.has(action));
}

export interface PermissionCheck {
  sessionId: string;
  runId: string;
  toolCallId: string;
  tool: ToolPermission & Pick<CoreToolDefinition, 'resolveResources'>;
  args: unknown;
  signal?: AbortSignal;
  /** Configured policy in precedence order. Saved rules and tool-specific mode rules are appended. */
  rulesets(): PermissionRulesets;
  modeRules?(): readonly PermissionRule[];
  /** Overrides the engine's default for unmatched resources. */
  defaultEffect?: PermissionEffect;
  onApprovalRequested?(request: PendingApprovalRequest): void;
  onApprovalResolved?(request: PendingApprovalRequest, decision: ApprovalDecision): void;
}

interface PendingApproval {
  check: PermissionCheck;
  request: PendingApprovalRequest;
  unresolved: string[];
  resolve(result?: BeforeToolCallResult): void;
  abort(): void;
  saving?: boolean;
}

const ABORTED = { block: true, reason: 'Operation aborted while waiting for permission' };

export class PermissionEngine {
  private readonly path?: string;
  private saved: PermissionRule[] = [];
  private writes = Promise.resolve();
  private readonly pending = new Map<string, PendingApproval>();

  constructor(dataDir?: string, private readonly defaultEffect: 'allow' | 'ask' = 'allow') {
    if (dataDir === undefined) return;
    this.path = join(dataDir, 'approvals.json');
    try {
      this.saved = JSON.parse(readFileSync(this.path, 'utf8'));
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error;
    }
  }

  get savedRules(): readonly PermissionRule[] {
    return this.saved.map((rule) => ({ ...rule }));
  }

  evaluate(action: string, resource: string, ...rulesets: PermissionRulesets): PermissionEffect {
    return this.findEffect(action, resource, rulesets) ?? this.defaultEffect;
  }

  private findEffect(action: string, resource: string, rulesets: PermissionRulesets): PermissionEffect | undefined {
    for (let index = rulesets.length - 1; index >= 0; index--) {
      const rules = rulesets[index];
      for (let ruleIndex = rules.length - 1; ruleIndex >= 0; ruleIndex--) {
        const rule = rules[ruleIndex];
        if (matchWildcard(rule.action, action) && matchesResource(action, rule.resource, resource)) return rule.effect;
      }
    }
    return undefined;
  }

  isToolGloballyDenied(tool: ToolPermission, rulesets: PermissionRulesets): boolean {
    const action = actionOf(tool);
    const rules = rulesets.flat().filter((rule) => matchWildcard(rule.action, action));
    let global = rules.length - 1;
    while (global >= 0 && !matchesResource(action, rules[global].resource, '*')) global--;
    if (rules[global]?.effect !== 'deny') return false;

    // A narrower exception exposes the tool so that its actual arguments can be checked.
    const covered = new Set<string>();
    for (const rule of rules.slice(global + 1).reverse()) {
      const pattern = CASE_INSENSITIVE_PATHS && PATH_ACTIONS.has(action) ? rule.resource.toLowerCase() : rule.resource;
      if (covered.has(pattern)) continue;
      covered.add(pattern);
      if (rule.effect !== 'deny' && (/[?*]/.test(pattern) || this.evaluate(action, rule.resource, rules) !== 'deny')) {
        return false;
      }
    }
    return true;
  }

  private classify(check: PermissionCheck, action: string, resources: string[]) {
    const rules = [...check.rulesets(), this.saved, check.modeRules?.() ?? []];
    const denied: string[] = [];
    const unresolved: string[] = [];
    for (const resource of resources) {
      const effect = this.findEffect(action, resource, rules) ?? check.defaultEffect ?? this.defaultEffect;
      if (effect === 'deny') denied.push(resource);
      if (effect === 'ask') unresolved.push(resource);
    }
    return { denied, unresolved };
  }

  async beforeToolCall(check: PermissionCheck): Promise<BeforeToolCallResult | undefined> {
    if (check.signal?.aborted) return ABORTED;
    const action = actionOf(check.tool);
    const resolved = check.tool.resolveResources?.(check.args as Record<string, unknown>);
    const resources = resolved?.length ? [...new Set(resolved)] : ['*'];
    const { denied, unresolved } = this.classify(check, action, resources);
    if (denied.length) {
      return { block: true, reason: `Permission denied for "${check.tool.name}" (${action}) on ${denied.join(', ')}` };
    }
    if (unresolved.length === 0) return undefined;

    const createdAt = Date.now();
    const request: PendingApprovalRequest = {
      id: randomUUID(), sessionId: check.sessionId, runId: check.runId, toolCallId: check.toolCallId,
      toolName: check.tool.name, action, resources, args: structuredClone(check.args),
      createdAt, timestamp: createdAt,
    };
    return new Promise((resolve, reject) => {
      const abort = () => this.finish(pending, ABORTED);
      const pending: PendingApproval = { check, request, unresolved, resolve, abort };
      this.pending.set(request.id, pending);
      check.signal?.addEventListener('abort', abort, { once: true });
      if (check.signal?.aborted) {
        abort();
        return;
      }
      try {
        check.onApprovalRequested?.(structuredClone(request));
      } catch (error) {
        this.pending.delete(request.id);
        check.signal?.removeEventListener('abort', abort);
        reject(error);
      }
    });
  }

  getPendingApprovals(sessionId?: string): PendingApprovalRequest[] {
    return [...this.pending.values()].filter(({ request }) => sessionId === undefined || request.sessionId === sessionId)
      .map(({ request }) => structuredClone(request));
  }

  replyApproval(requestId: string, decision: ApprovalDecision, sessionId?: string): void {
    const pending = this.pending.get(requestId);
    if (!pending || pending.saving || (sessionId !== undefined && pending.request.sessionId !== sessionId)) return;
    const request = pending.request;
    if (decision.kind === 'once') {
      this.finish(pending, undefined, decision);
    } else if (decision.kind === 'reject') {
      for (const other of [...this.pending.values()]) {
        if (other.request.sessionId !== request.sessionId || other.request.runId !== request.runId) continue;
        const feedback = decision.feedback ? ` User feedback: ${decision.feedback}` : '';
        this.finish(other, {
          block: true, reason: `User rejected permission for "${other.request.toolName}".${feedback}`,
        }, decision);
      }
    } else {
      pending.saving = true;
      const write = this.writes.then(() => this.saveApproval(pending, decision));
      this.writes = write.catch((error: unknown) => {
        const message = error instanceof Error ? error.message : String(error);
        this.finish(pending, { block: true, reason: `Could not save tool approval: ${message}` }, {
          kind: 'reject', feedback: message,
        });
      });
    }
  }

  private async saveApproval(pending: PendingApproval, decision: Extract<ApprovalDecision, { kind: 'always' }>) {
    const { request } = pending;
    if (!this.pending.has(request.id)) return;
    const resources = decision.savePattern === undefined ? pending.unresolved : [decision.savePattern];
    const rules: PermissionRule[] = [
      ...this.saved, ...resources.map((resource): PermissionRule => ({ action: request.action, resource, effect: 'allow' })),
    ];
    if (this.path) {
      const temporary = `${this.path}.${randomUUID()}.tmp`;
      await mkdir(dirname(this.path), { recursive: true });
      try {
        await writeFile(temporary, JSON.stringify(rules, null, 2) + '\n', { flag: 'wx', mode: 0o600 });
        await rename(temporary, this.path);
      } finally {
        await unlink(temporary).catch(() => undefined);
      }
    }
    this.saved = rules;
    this.finish(pending, undefined, decision);
    for (const other of [...this.pending.values()]) {
      if (other.saving || other.request.sessionId !== request.sessionId) continue;
      const { denied, unresolved } = this.classify(other.check, other.request.action, other.request.resources);
      if (!denied.length && !unresolved.length) this.finish(other, undefined, decision);
    }
  }

  private finish(pending: PendingApproval, result?: BeforeToolCallResult, decision?: ApprovalDecision): void {
    if (!this.pending.delete(pending.request.id)) return;
    pending.check.signal?.removeEventListener('abort', pending.abort);
    pending.resolve(result);
    if (decision) pending.check.onApprovalResolved?.(structuredClone(pending.request), { ...decision });
  }

  flush(): Promise<void> {
    return this.writes;
  }
}
