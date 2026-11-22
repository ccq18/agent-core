import type { SkillMetadata } from './skills.js';

const DEFAULT_PROMPT = 'You are a helpful assistant. Use the available tools when they help you complete the task.';

export interface PromptOptionsForAssembly {
  baseSystemPrompt?: string;
  rules?: readonly string[];
  skills?: readonly SkillMetadata[];
  cwd?: string;
  date?: Date;
}

export class PromptAssembler {
  assemble(options: PromptOptionsForAssembly = {}): string {
    const { baseSystemPrompt = DEFAULT_PROMPT, rules = [], skills = [], cwd, date = new Date() } = options;
    const sections = [baseSystemPrompt];
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
