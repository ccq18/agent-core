import { closeSync, constants, fstatSync, openSync, readSync, realpathSync, statSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { StringDecoder } from 'node:string_decoder';

const MAX_RULE_BYTES = 256 * 1024;

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
    const paths = new Set(this.globalPaths);
    const project: Array<{ path: string; rule: string }> = [];
    if (cwd) {
      let directory = resolve(cwd);
      for (let level = 0; level < 8; level++) {
        for (const name of ['AGENTS.md', 'CLAUDE.md']) {
          const candidate = join(directory, name);
          const rule = readTextPrefix(candidate, MAX_RULE_BYTES);
          if (rule === undefined) continue;
          project.push({ path: realpathSync(candidate), rule });
          break;
        }
        const parent = dirname(directory);
        if (parent === directory || statSync(join(directory, '.git'), { throwIfNoEntry: false })) break;
        directory = parent;
      }
    }
    const rules = [...this.globalText];
    for (const { path, rule } of project.reverse()) {
      if (paths.has(path)) continue;
      paths.add(path);
      rules.push(rule);
    }
    return rules;
  }
}
