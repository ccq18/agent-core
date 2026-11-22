import { readdirSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { basename, dirname, join, resolve } from 'node:path';
import { parseDocument } from 'yaml';
import type { CoreToolDefinition } from '../types/tool.js';
import { readTextPrefix } from './rules.js';

export interface SkillMetadata {
  id: string;
  name: string;
  description?: string;
}

interface Skill extends SkillMetadata {
  path: string;
  disableModelInvocation: boolean;
}

const FRONTMATTER = /^\uFEFF?---[^\S\r\n]*\r?\n([\s\S]*?)^---[^\S\r\n]*(?:\r?\n|$)/m;

function skillPaths(directory: string, root = true): string[] {
  let entries;
  try {
    entries = readdirSync(directory, { withFileTypes: true });
  } catch (error) {
    if (['ENOENT', 'ENOTDIR'].includes((error as NodeJS.ErrnoException).code ?? '')) return [];
    throw error;
  }
  return entries.sort((a, b) => a.name.localeCompare(b.name)).flatMap((entry) => {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) return skillPaths(path, false);
    return (entry.isFile() || entry.isSymbolicLink()) &&
      (entry.name === 'SKILL.md' || (root && entry.name.endsWith('.md'))) ? [path] : [];
  });
}

function metadata({ id, name, description }: Skill): SkillMetadata {
  return { id, name, ...(description === undefined ? {} : { description }) };
}

function escapeAttribute(value: string): string {
  return value.replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

export class SkillsRegistry {
  private readonly skills = new Map<string, Skill>();

  constructor(directories: readonly string[] = []) {
    for (const directory of directories) {
      for (const path of skillPaths(resolve(directory))) {
        const prefix = readTextPrefix(path, 64 * 1024);
        if (prefix === undefined) continue;
        const frontmatter = FRONTMATTER.exec(prefix);
        const document = frontmatter?.index === 0 ? parseDocument(frontmatter[1]) : undefined;
        const metadata = document?.errors.length ? undefined : document;
        const configuredName = metadata?.get('name');
        const description = metadata?.get('description');
        const fallback = basename(path) === 'SKILL.md' ? basename(dirname(path)) : basename(path, '.md');
        const name = typeof configuredName === 'string' && configuredName.trim() ? configuredName.trim() : fallback;
        if (this.skills.has(name)) continue;
        this.skills.set(name, {
          id: name, name, description: typeof description === 'string' ? description.trim() : undefined,
          disableModelInvocation: metadata?.get('disable-model-invocation') === true, path,
        });
      }
    }
  }

  list(): SkillMetadata[] {
    return [...this.skills.values()].map(metadata);
  }

  modelIndex(): SkillMetadata[] {
    return [...this.skills.values()].filter((skill) => !skill.disableModelInvocation).map(metadata);
  }

  createTool(): CoreToolDefinition {
    return {
      name: 'load_skill',
      label: 'Load skill',
      description: 'Load the instructions for a skill by its ID.',
      readOnly: true,
      permissionAction: 'skill',
      resolveResources: (params) => [String(params.id)],
      parameters: { type: 'object', properties: { id: { type: 'string' } }, required: ['id'] },
      execute: async (params) => {
        const id = String(params.id);
        const skill = this.skills.get(id);
        if (!skill) throw new Error(`Unknown skill "${id}"`);
        const raw = await readFile(skill.path, 'utf8');
        const frontmatter = FRONTMATTER.exec(raw);
        const body = frontmatter?.index === 0 ? raw.slice(frontmatter[0].length) : raw;
        const end = body.endsWith('\n') ? '' : '\n';
        return { content: [{ type: 'text', text: `<skill_content id="${escapeAttribute(id)}">\n${body}${end}</skill_content>` }] };
      },
    };
  }
}
