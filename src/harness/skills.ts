import { readdirSync } from 'node:fs';
import { readdir, readFile } from 'node:fs/promises';
import { basename, dirname, join, resolve } from 'node:path';
import { parseDocument } from 'yaml';
import type { CoreToolDefinition } from '../types/tool.js';
import { readTextPrefix, readTextPrefixAsync } from './rules.js';

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
const SKILL_PREFIX_BYTES = 64 * 1024;
const SCAN_CONCURRENCY = 8;

type Schedule = <T>(operation: () => Promise<T>) => Promise<T>;

function boundedScan(): Schedule {
  let active = 0;
  const waiting: Array<() => void> = [];
  return async (operation) => {
    if (active >= SCAN_CONCURRENCY) await new Promise<void>((start) => waiting.push(start));
    else active++;
    try {
      return await operation();
    } finally {
      const next = waiting.shift();
      if (next) next();
      else active--;
    }
  };
}

async function orderedResults<T>(pending: Promise<T>[]): Promise<T[]> {
  return (await Promise.allSettled(pending)).map((result) => {
    if (result.status === 'rejected') throw result.reason;
    return result.value;
  });
}

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

async function skillPathsAsync(directory: string, schedule: Schedule, root = true): Promise<string[]> {
  const entries = await schedule(() => readdir(directory, { withFileTypes: true })).catch((error) => {
    if (['ENOENT', 'ENOTDIR'].includes((error as NodeJS.ErrnoException).code ?? '')) return [];
    throw error;
  });
  const paths = await orderedResults(entries.sort((a, b) => a.name.localeCompare(b.name)).map(async (entry) => {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) return skillPathsAsync(path, schedule, false);
    return (entry.isFile() || entry.isSymbolicLink()) &&
      (entry.name === 'SKILL.md' || (root && entry.name.endsWith('.md'))) ? [path] : [];
  }));
  return paths.flat();
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
        this.register(path, readTextPrefix(path, SKILL_PREFIX_BYTES));
      }
    }
  }

  static async loadAsync(directories: readonly string[] = []): Promise<SkillsRegistry> {
    const registry = new SkillsRegistry();
    const schedule = boundedScan();
    const paths = (await orderedResults(directories.map((directory) => skillPathsAsync(resolve(directory), schedule)))).flat();
    for (let start = 0; start < paths.length; start += SCAN_CONCURRENCY) {
      const batch = paths.slice(start, start + SCAN_CONCURRENCY);
      const prefixes = await orderedResults(batch.map((path) => readTextPrefixAsync(path, SKILL_PREFIX_BYTES)));
      for (let i = 0; i < batch.length; i++) registry.register(batch[i], prefixes[i]);
    }
    return registry;
  }

  private register(path: string, prefix: string | undefined): void {
    if (prefix === undefined) return;
    const frontmatter = FRONTMATTER.exec(prefix);
    const document = frontmatter?.index === 0 ? parseDocument(frontmatter[1]) : undefined;
    const metadata = document?.errors.length ? undefined : document;
    const configuredName = metadata?.get('name');
    const description = metadata?.get('description');
    const fallback = basename(path) === 'SKILL.md' ? basename(dirname(path)) : basename(path, '.md');
    const name = typeof configuredName === 'string' && configuredName.trim() ? configuredName.trim() : fallback;
    if (this.skills.has(name)) return;
    this.skills.set(name, {
      id: name, name, description: typeof description === 'string' ? description.trim() : undefined,
      disableModelInvocation: metadata?.get('disable-model-invocation') === true, path,
    });
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
