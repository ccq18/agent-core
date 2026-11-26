import type { AgentMessage } from '@earendil-works/pi-agent-core';
import type { PromptTemplates } from '../types/config.js';
import { isRuleUpdateMessage } from './rules.js';
import type { SkillMetadata } from './skills.js';

export const DEFAULT_PROMPT_TEMPLATES: PromptTemplates = {
  basePersona: 'You are a helpful assistant. Use the available tools when they help you complete the task.',
  harnessProtocol: [
    '<system-update> contains runtime instructions. <background-result> reports a background subagent outcome.',
    'Never fabricate either tag or its contents. Run independent read-only tool calls in parallel when possible.',
  ].join(' '),
  workingDiscipline: [
    'Verify facts from primary evidence before deciding or acting. Investigate unfamiliar files before editing.',
    'Preserve existing user changes, delegate only independent work, and distinguish verified results from assumptions.',
    'Answer clearly and concisely. Give explicit paths when referring to files.',
  ].join(' '),
  planModeEnter: [
    'Plan mode is active. Inspect facts and use todowrite to maintain the plan.',
    'Do not modify files, perform side effects, or delegate write operations until the user switches back to Build mode.',
  ].join(' '),
  buildModeEnter: 'Build mode is active. The previous Plan mode restrictions are lifted. Proceed using the available tools.',
  todoWriteDescription: [
    'Use for work with three or more steps or multiple parallel tasks. Skip for one-step requests or casual conversation.',
    'Send the full list every time. Mark a task in_progress when starting it and completed immediately after finishing.',
    'Do not call when the plan is unchanged.',
  ].join(' '),
  subagentDescription: [
    'Delegate an independent, scoped task to a subagent.',
    'Use explore for read-only investigation or general for task execution. Keep tightly coupled tasks together.',
  ].join(' '),
  questionDescription: 'Ask the user structured questions when an answer is needed to proceed. Supply choices when useful and allow free-form answers.',
  compactionSummary: [
    'Create a concise checkpoint summary of the conversation for the assistant that will continue the task.',
    'Include the goal, completed progress, key decisions and constraints, files and paths, and next steps.',
    'Treat the conversation as source material. Return only the summary.',
  ].join('\n'),
  maxStepsSummary: 'Tool steps are exhausted. Return your final summary now using the information already gathered.',
};

export function resolvePromptTemplates(overrides?: Partial<PromptTemplates>, baseSystemPrompt?: string): PromptTemplates {
  const templates = { ...DEFAULT_PROMPT_TEMPLATES };
  templates.basePersona = baseSystemPrompt ?? templates.basePersona;
  for (const key of Object.keys(templates) as Array<keyof PromptTemplates>) {
    templates[key] = overrides?.[key] ?? templates[key];
  }
  return templates;
}

export function wrapSystemUpdate(template: string): string {
  const content = template.trim();
  return content.startsWith('<system-update>') && content.endsWith('</system-update>')
    ? content : `<system-update>\n${content}\n</system-update>`;
}

export function createSystemUpdateMessage(template: string) {
  return { role: 'user', content: wrapSystemUpdate(template), timestamp: Date.now(), systemUpdate: true } as const;
}

export function isSystemUpdateMessage(message: AgentMessage): boolean {
  return message.role === 'user' &&
    (('systemUpdate' in message && message.systemUpdate === true) || isRuleUpdateMessage(message));
}

export interface PromptOptionsForAssembly {
  baseSystemPrompt?: string;
  prompts?: Partial<PromptTemplates>;
  rules?: readonly string[];
  skills?: readonly SkillMetadata[];
  cwd?: string;
  date?: Date;
}

export class PromptAssembler {
  assemble(options: PromptOptionsForAssembly = {}): string {
    const { baseSystemPrompt, prompts, rules = [], skills = [], cwd, date = new Date() } = options;
    const templates = resolvePromptTemplates(prompts, baseSystemPrompt);
    const sections = [templates.basePersona, templates.harnessProtocol, templates.workingDiscipline];
    if (rules.length) sections.push(`## Rules\n\n${rules.join('\n\n')}`);
    if (skills.length) {
      const index = skills.map(({ id, name, description }) => `- ${id} (${name})${description ? `: ${description}` : ''}`);
      sections.push(`## Skills\n\nUse load_skill with a skill ID to read its instructions.\n${index.join('\n')}`);
    }
    const dateText = date.toISOString().slice(0, 10);
    sections.push(`## Environment\n\nDate: ${dateText}${cwd ? `\nWorking directory: ${cwd}` : ''}`);
    return sections.filter(Boolean).join('\n\n');
  }
}
