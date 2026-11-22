import { closeSync, constants, fstatSync, openSync, readSync, realpathSync, statSync } from 'node:fs';
import { basename, dirname, isAbsolute, join, relative, resolve, sep } from 'node:path';
import { StringDecoder } from 'node:string_decoder';
import type { AgentMessage } from '@earendil-works/pi-agent-core';

const MAX_RULE_BYTES = 256 * 1024;
const RULE_FILES = ['AGENTS.md', 'CLAUDE.md'];

export type RuleUpdateMessage = Extract<AgentMessage, { role: 'user' }> & { rulePaths: string[] };

export function isRuleUpdateMessage(message: AgentMessage): message is RuleUpdateMessage {
  return message.role === 'user' && 'rulePaths' in message && Array.isArray(message.rulePaths);
}

function isInside(directory: string, path: string): boolean {
  const fromRoot = relative(directory, path);
  return fromRoot !== '..' && !fromRoot.startsWith(`..${sep}`) && !isAbsolute(fromRoot);
}

function realPathIfPresent(path: string): string | undefined {
  try {
    return realpathSync(path);
  } catch (error) {
    if (absent(error)) return undefined;
    throw error;
  }
}

// A write or lookup may target a file that does not exist yet.
function resolveExistingParents(path: string): string {
  const actual = realPathIfPresent(path);
  if (actual !== undefined) return actual;
  const parent = dirname(path);
  return parent === path ? path : join(resolveExistingParents(parent), basename(path));
}

function absent(error: unknown): boolean {
  return ['ENOENT', 'ENOTDIR', 'ENAMETOOLONG', 'ERR_INVALID_ARG_VALUE']
    .includes((error as NodeJS.ErrnoException).code ?? '');
}

/** Read a bounded UTF-8 prefix from a regular file. */
export function readTextPrefix(path: string, maxBytes: number): string | undefined {
  let fd: number;
  try {
    fd = openSync(path, constants.O_RDONLY | constants.O_NONBLOCK);
  } catch (error) {
    if (absent(error)) return undefined;
    throw error;
  }
  try {
    const stat = fstatSync(fd);
    if (!stat.isFile()) return undefined;
    const buffer = Buffer.allocUnsafe(Math.min(stat.size, maxBytes));
    let size = 0;
    while (size < buffer.length) {
      const read = readSync(fd, buffer, size, buffer.length - size, null);
      if (read === 0) break;
      size += read;
    }
    const bytes = buffer.subarray(0, size);
    return stat.size > maxBytes ? new StringDecoder('utf8').write(bytes) : bytes.toString('utf8');
  } finally {
    closeSync(fd);
  }
}

export class RulesLoader {
  private readonly globalText: string[] = [];
  private readonly globalPaths = new Set<string>();

  constructor(globalRules: readonly string[] = []) {
    for (const rule of globalRules) {
      const text = readTextPrefix(resolve(rule), MAX_RULE_BYTES);
      if (text === undefined) {
        this.globalText.push(rule);
      } else {
        const path = realpathSync(resolve(rule));
        if (!this.globalPaths.has(path)) this.globalText.push(text);
        this.globalPaths.add(path);
      }
    }
  }

  load(cwd?: string): string[] {
    return this.loadWithPaths(cwd).rules;
  }

  loadWithPaths(cwd?: string): { rules: string[]; paths: Set<string> } {
    const project = cwd ? this.discoverDirectories(resolve(cwd), this.globalPaths) : [];
    return {
      rules: [...this.globalText, ...project.map(({ rule }) => rule)],
      paths: new Set([...this.globalPaths, ...project.map(({ path }) => path)]),
    };
  }

  discoverForPath(targetPath: string, loadedPaths: ReadonlySet<string>, stopDir?: string): Array<{ path: string; rule: string }> {
    if (!targetPath.trim() || targetPath.includes('\0') || /^[a-z][a-z0-9+.-]*:\/\//i.test(targetPath)) return [];
    const root = stopDir === undefined ? undefined : resolve(stopDir);
    const target = root === undefined ? resolve(targetPath) : resolve(root, targetPath);
    if (root !== undefined && !isInside(root, target)) return [];
    const stop = root === undefined ? undefined : resolveExistingParents(root);
    const actual = resolveExistingParents(target);
    if (stop !== undefined && !isInside(stop, actual)) return [];
    const directory = statSync(actual, { throwIfNoEntry: false })?.isDirectory() ? actual : dirname(actual);
    return this.discoverDirectories(directory, loadedPaths, stop);
  }

  private discoverDirectories(start: string, loadedPaths: ReadonlySet<string>, stop?: string): Array<{ path: string; rule: string }> {
    if (stop !== undefined && !isInside(stop, start)) return [];
    const project: Array<{ path: string; rule: string }> = [];
    let directory = start;
    for (let level = 0; level < 8; level++) {
      for (const name of RULE_FILES) {
        const path = realPathIfPresent(join(directory, name));
        if (path === undefined || (stop !== undefined && !isInside(stop, path))) continue;
        if (loadedPaths.has(path)) break;
        const rule = readTextPrefix(path, MAX_RULE_BYTES);
        if (rule === undefined) continue;
        project.push({ path, rule });
        break;
      }
      const parent = dirname(directory);
      if (parent === directory || directory === stop || statSync(join(directory, '.git'), { throwIfNoEntry: false })) break;
      directory = parent;
    }
    const seen = new Set<string>();
    return project.reverse().filter(({ path }) => {
      if (seen.has(path)) return false;
      seen.add(path);
      return true;
    });
  }
}
