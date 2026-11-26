import { closeSync, constants, fstatSync, openSync, readSync, realpathSync, statSync, type Stats } from 'node:fs';
import { open, realpath, stat } from 'node:fs/promises';
import { basename, dirname, isAbsolute, join, relative, resolve, sep } from 'node:path';
import { StringDecoder } from 'node:string_decoder';
import type { AgentMessage } from '@earendil-works/pi-agent-core';

const MAX_RULE_BYTES = 256 * 1024;
const RULE_FILES = ['AGENTS.md', 'CLAUDE.md'];
const MAX_CACHED_DIRECTORIES = 256;
const NEGATIVE_TTL_MS = 1000;

interface RuleFile {
  path: string;
  rule?: string;
  stats: Stats;
}

interface DirectoryRules {
  files: RuleFile[];
  git: boolean;
  directory?: Stats;
  checkedAt: number;
  skippedLoaded?: boolean;
}

type RuleDiscovery = Array<{ path: string; rule: string }>;

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

function optionalStat(path: string): Promise<Stats | undefined> {
  return stat(path).catch((error) => {
    if (!absent(error)) throw error;
    return undefined;
  });
}

function optionalRealpath(path: string): Promise<string | undefined> {
  return realpath(path).catch((error) => {
    if (!absent(error)) throw error;
    return undefined;
  });
}

async function resolveExistingParentsAsync(path: string): Promise<string> {
  const actual = await optionalRealpath(path);
  if (actual !== undefined) return actual;
  const parent = dirname(path);
  return parent === path ? path : join(await resolveExistingParentsAsync(parent), basename(path));
}

export async function readTextPrefixAsync(path: string, maxBytes: number, noFollow = false): Promise<string | undefined> {
  const file = await open(path, constants.O_RDONLY | constants.O_NONBLOCK | (noFollow ? constants.O_NOFOLLOW : 0))
    .catch((error) => {
      if (!absent(error)) throw error;
      return undefined;
    });
  if (!file) return undefined;
  try {
    const stats = await file.stat();
    if (!stats.isFile()) return undefined;
    const buffer = Buffer.allocUnsafe(Math.min(stats.size, maxBytes));
    let size = 0;
    while (size < buffer.length) {
      const { bytesRead } = await file.read(buffer, size, buffer.length - size, null);
      if (bytesRead === 0) break;
      size += bytesRead;
    }
    const bytes = buffer.subarray(0, size);
    return stats.size > maxBytes ? new StringDecoder('utf8').write(bytes) : bytes.toString('utf8');
  } finally {
    await file.close();
  }
}

function sameStats(a: Stats | undefined, b: Stats): boolean {
  return a?.ino === b.ino && a.dev === b.dev && a.size === b.size && a.mtimeMs === b.mtimeMs && a.ctimeMs === b.ctimeMs;
}

export class RulesLoader {
  private readonly globalText: string[] = [];
  private readonly globalPaths = new Set<string>();
  private readonly cache = new Map<string, DirectoryRules>();
  private readonly inFlight = new Map<string, Promise<DirectoryRules>>();

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

  async loadWithPathsAsync(cwd?: string): Promise<{ rules: string[]; paths: Set<string> }> {
    const project = cwd ? await this.discoverForProjectAsync(cwd) : [];
    return {
      rules: [...this.globalText, ...project.map(({ rule }) => rule)],
      paths: new Set([...this.globalPaths, ...project.map(({ path }) => path)]),
    };
  }

  async discoverForProjectAsync(cwd: string): Promise<RuleDiscovery> {
    return this.discoverDirectoriesAsync(resolve(cwd), this.globalPaths, undefined, new Map());
  }

  async discoverForPathAsync(
    targetPath: string, loadedPaths: ReadonlySet<string>, stopDir?: string,
  ): Promise<RuleDiscovery> {
    const root = stopDir === undefined ? undefined : resolve(stopDir);
    const stop = root === undefined ? undefined : await resolveExistingParentsAsync(root);
    return this.discoverTargetAsync(targetPath, loadedPaths, root, stop, new Map());
  }

  /** Best-effort discovery across tool arguments; an unreadable target leaves the others usable. */
  async discoverForPathsAsync(
    targetPaths: readonly string[], loadedPaths: ReadonlySet<string>, stopDir?: string,
  ): Promise<RuleDiscovery> {
    if (!targetPaths.length) return [];
    const root = stopDir === undefined ? undefined : resolve(stopDir);
    const stop = root === undefined ? undefined : await resolveExistingParentsAsync(root).catch(() => undefined);
    if (root !== undefined && stop === undefined) return [];
    const batch = new Map<string, Promise<DirectoryRules>>();
    const results = await Promise.allSettled([...new Set(targetPaths)].map((target) =>
      this.discoverTargetAsync(target, loadedPaths, root, stop, batch)));
    const found = new Map<string, string>();
    for (const result of results) {
      if (result.status === 'fulfilled') for (const { path, rule } of result.value) found.set(path, rule);
    }
    return Array.from(found, ([path, rule]) => ({ path, rule }));
  }

  private async discoverTargetAsync(
    targetPath: string, loadedPaths: ReadonlySet<string>, root: string | undefined, stop: string | undefined,
    batch: Map<string, Promise<DirectoryRules>>,
  ): Promise<RuleDiscovery> {
    if (!targetPath.trim() || targetPath.includes('\0') || /^[a-z][a-z0-9+.-]*:\/\//i.test(targetPath)) return [];
    const target = root === undefined ? resolve(targetPath) : resolve(root, targetPath);
    if (root !== undefined && !isInside(root, target)) return [];
    const actual = await resolveExistingParentsAsync(target);
    if (stop !== undefined && !isInside(stop, actual)) return [];
    const directory = (await optionalStat(actual))?.isDirectory() ? actual : dirname(actual);
    return this.discoverDirectoriesAsync(directory, loadedPaths, stop, batch);
  }

  private async discoverDirectoriesAsync(
    start: string, loadedPaths: ReadonlySet<string>, stop: string | undefined, batch: Map<string, Promise<DirectoryRules>>,
  ): Promise<RuleDiscovery> {
    if (stop !== undefined && !isInside(stop, start)) return [];
    const project: RuleDiscovery = [];
    const knownPaths = new Set(loadedPaths);
    const loadedKey = knownPaths.size ? JSON.stringify([...knownPaths].sort()) : '';
    let directory = start;
    for (let level = 0; level < 8; level++) {
      const key = (stop ?? '') + '\0' + directory;
      const requestKey = key + '\0' + loadedKey;
      let pending = batch.get(requestKey);
      if (!pending) {
        pending = this.directoryRules(key, requestKey, directory, knownPaths, stop);
        batch.set(requestKey, pending);
      }
      const { files, git } = await pending;
      for (const { path, rule } of files) {
        if (loadedPaths.has(path)) break;
        if (rule === undefined) continue;
        project.push({ path, rule });
        break;
      }
      const parent = dirname(directory);
      if (parent === directory || directory === stop || git) break;
      directory = parent;
    }
    const seen = new Set<string>();
    return project.reverse().filter(({ path }) => {
      if (loadedPaths.has(path) || seen.has(path)) return false;
      seen.add(path);
      return true;
    });
  }

  private directoryRules(
    key: string, requestKey: string, directory: string, loadedPaths: ReadonlySet<string>, stop?: string,
  ): Promise<DirectoryRules> {
    const existing = this.inFlight.get(requestKey);
    if (existing) return existing;
    const pending = this.probeDirectory(directory, this.cache.get(key), loadedPaths, stop).then((result) => {
      // A skipped file is still a candidate for other sessions.
      if (result.skippedLoaded) return result;
      this.cache.delete(key);
      this.cache.set(key, result);
      if (this.cache.size > MAX_CACHED_DIRECTORIES) this.cache.delete(this.cache.keys().next().value!);
      return result;
    }).finally(() => { this.inFlight.delete(requestKey); });
    this.inFlight.set(requestKey, pending);
    return pending;
  }

  private async probeDirectory(
    directory: string, cached: DirectoryRules | undefined, loadedPaths: ReadonlySet<string>, stop?: string,
  ): Promise<DirectoryRules> {
    const [directoryStats, git] = await Promise.all([
      optionalStat(directory), directory === stop || dirname(directory) === directory
        ? undefined : optionalStat(join(directory, '.git')),
    ]);
    const checkedAt = Date.now();
    if (cached && cached.files.length === 0 && checkedAt - cached.checkedAt < NEGATIVE_TTL_MS &&
        (directoryStats ? sameStats(cached.directory, directoryStats) : cached.directory === undefined)) {
      return { ...cached, git: !!git };
    }
    const files: RuleFile[] = [];
    for (const name of RULE_FILES) {
      const path = await optionalRealpath(join(directory, name));
      if (path === undefined || (stop !== undefined && !isInside(stop, path))) continue;
      if (loadedPaths.has(path)) return { files, git: !!git, directory: directoryStats, checkedAt, skippedLoaded: true };
      const stats = await optionalStat(path);
      if (!stats) continue;
      const previous = cached?.files.find((file) => file.path === path && sameStats(file.stats, stats));
      const rule = previous ? previous.rule : await readTextPrefixAsync(path, MAX_RULE_BYTES, true);
      files.push({ path, stats, rule });
      if (rule !== undefined) break;
    }
    return { files, git: !!git, directory: directoryStats, checkedAt };
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
