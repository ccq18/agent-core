import { randomUUID } from 'node:crypto';
import { constants } from 'node:fs';
import { mkdir, open as openFile, readdir, unlink } from 'node:fs/promises';
import { basename, join } from 'node:path';
import type { CreateSessionOptions } from '../types/config.js';
import type {
  SessionEntry, SessionEntryData, SessionHeader, SessionProjection, SessionSnapshot,
  SessionStorageAdapter, SessionSummary,
} from '../types/session.js';

interface SessionTree extends SessionSnapshot {
  entries: Map<string, SessionEntry>;
  updatedAt: number;
  needsNewline: boolean;
}

function isValidId(id: string): boolean {
  return typeof id === 'string' && id.length >= 1 && id.length <= 128 && !/[^a-zA-Z0-9_-]/.test(id);
}

function isMissing(error: unknown): boolean {
  return (error as NodeJS.ErrnoException).code === 'ENOENT';
}

function traceBranch(tree: SessionTree, leafId: string | null): SessionEntry[] {
  const branch: SessionEntry[] = [];
  for (let id = leafId; id !== null;) {
    const entry = tree.entries.get(id);
    if (!entry) throw new Error(`Unknown session entry "${id}"`);
    branch.push(entry);
    id = entry.parentId;
  }
  return branch.reverse();
}

function projectSettings(header: SessionHeader, branch: SessionEntry[]) {
  let mode = header.mode;
  let modeChanged = false;
  let modelAlias = header.modelAlias;
  let plan: SessionProjection['plan'] = [];
  for (const entry of branch) {
    if (entry.type === 'mode') {
      mode = entry.mode;
      modeChanged = true;
    }
    if (entry.type === 'model') modelAlias = entry.modelAlias;
    if (entry.type === 'plan') plan = entry.items;
  }
  return { mode, modelAlias, plan, ...(modeChanged ? { modeChanged } : {}) };
}

export function getCompactionIndex(branch: readonly SessionEntry[]): number {
  for (let index = branch.length - 1; index >= 0; index--) {
    if (branch[index].type === 'compaction') return index;
  }
  return -1;
}

export function getCompactionStart(branch: readonly SessionEntry[], checkpoint: number): number {
  const entry = branch[checkpoint];
  if (entry?.type !== 'compaction') return 0;
  const firstKept = branch.findIndex(({ id }) => id === entry.firstKeptEntryId);
  return firstKept >= 0 && firstKept < checkpoint ? firstKept : checkpoint;
}

export function projectBranchMessages(branch: readonly SessionEntry[]): Pick<
  SessionProjection, 'messages' | 'compactedMessageCount'
> {
  const checkpoint = getCompactionIndex(branch);
  const compaction = branch[checkpoint];
  const messages: SessionProjection['messages'] = [];
  let compactedMessageCount: number | undefined;
  if (compaction?.type === 'compaction') {
    const baseline = branch[0];
    if (baseline?.type === 'message' && baseline.message.role === 'system') messages.push(baseline.message);
    messages.push({
      role: 'user',
      content: `<compaction-summary>\n${compaction.summary}\n</compaction-summary>`,
      timestamp: compaction.timestamp,
    });
    for (const entry of branch.slice(getCompactionStart(branch, checkpoint), checkpoint)) {
      if (entry.type === 'message' && entry.message.role !== 'system') messages.push(entry.message);
    }
    compactedMessageCount = messages.length;
  }
  for (const entry of branch.slice(checkpoint + 1)) {
    if (entry.type === 'message') messages.push(entry.message);
  }
  return compactedMessageCount === undefined ? { messages } : { messages, compactedMessageCount };
}

export class JsonlTreeSessionStore implements SessionStorageAdapter {
  private readonly directory: string;
  private readonly trees = new Map<string, SessionTree>();
  private readonly operations = new Map<string, Promise<unknown>>();

  constructor(dataDir: string) {
    this.directory = join(dataDir, 'sessions');
  }

  private path(id: string): string {
    if (!isValidId(id)) throw new Error(`Invalid session ID: ${JSON.stringify(id)}`);
    return join(this.directory, `${id}.jsonl`);
  }

  private async withSession<T>(id: string, operation: (path: string) => Promise<T>): Promise<T> {
    const path = this.path(id);
    const previous = this.operations.get(id) ?? Promise.resolve();
    const next = previous.catch(() => undefined).then(() => operation(path));
    this.operations.set(id, next);
    try {
      return await next;
    } finally {
      if (this.operations.get(id) === next) this.operations.delete(id);
    }
  }

  create(options: CreateSessionOptions = {}): Promise<SessionHeader> {
    const id = options.id ?? randomUUID();
    return this.withSession(id, async (path) => {
      const header: SessionHeader = {
        type: 'session', version: 1, id, createdAt: Date.now(),
        mode: options.mode ?? 'build', modelAlias: options.modelAlias,
        title: options.title, cwd: options.cwd, permissions: options.permissions, metadata: options.metadata,
      };
      const line = JSON.stringify(header);
      await mkdir(this.directory, { recursive: true });
      const file = await openFile(path, constants.O_WRONLY | constants.O_CREAT | constants.O_EXCL, 0o600);
      try {
        await file.writeFile(line + '\n');
      } catch (error) {
        await unlink(path);
        throw error;
      } finally {
        await file.close();
      }
      const saved: SessionHeader = JSON.parse(line);
      this.trees.set(id, {
        header: saved, leafId: null, entries: new Map(), updatedAt: saved.createdAt, needsNewline: false,
      });
      return structuredClone(saved);
    });
  }

  private async load(id: string, path: string): Promise<SessionTree> {
    const cached = this.trees.get(id);
    if (cached) return cached;
    const file = await openFile(path, constants.O_RDONLY | constants.O_NOFOLLOW);
    let raw: string;
    try {
      raw = await file.readFile('utf8');
    } finally {
      await file.close();
    }
    const [first, ...lines] = raw.split('\n');
    const header: SessionHeader = JSON.parse(first);
    if (header.type !== 'session' || header.id !== id) throw new Error(`Invalid session header for "${id}"`);
    const entries = new Map<string, SessionEntry>();
    let leafId: string | null = null;
    let updatedAt = header.createdAt;
    for (const line of lines) {
      if (!line.trim()) continue;
      const entry: SessionEntry = JSON.parse(line);
      if (entries.has(entry.id) || (entry.parentId !== null && !entries.has(entry.parentId))) {
        throw new Error(`Invalid session tree entry "${entry.id}" in "${id}"`);
      }
      entries.set(entry.id, entry);
      leafId = entry.id;
      updatedAt = Math.max(updatedAt, entry.timestamp);
    }
    const tree = { header, entries, leafId, updatedAt, needsNewline: !raw.endsWith('\n') };
    this.trees.set(id, tree);
    return tree;
  }

  open(id: string): Promise<SessionSnapshot> {
    return this.withSession(id, async (path) => {
      const { header, leafId } = await this.load(id, path);
      return { header: structuredClone(header), leafId };
    });
  }

  async list(): Promise<SessionSummary[]> {
    const files = await readdir(this.directory, { withFileTypes: true }).catch((error: unknown) => {
      if (isMissing(error)) return [];
      throw error;
    });
    const summaries = await Promise.all(files
      .filter((file) => file.isFile() && file.name.endsWith('.jsonl') && isValidId(file.name.slice(0, -6)))
      .map((file) => {
        const id = file.name.slice(0, -6);
        return this.withSession(id, async (path) => {
          const tree = await this.load(id, path);
          const { header, updatedAt } = tree;
          const branch = traceBranch(tree, tree.leafId);
          const { mode, modelAlias } = projectSettings(header, branch);
          let metadata = header.metadata;
          for (const entry of branch) {
            if (entry.type === 'metadata') metadata = { ...metadata, ...entry.metadata };
          }
          const title = typeof metadata?.title === 'string' ? metadata.title : header.title;
          return { id, title, mode, modelAlias, createdAt: header.createdAt, updatedAt, metadata: structuredClone(metadata) };
        }).catch((error: unknown) => {
          if (isMissing(error)) return undefined;
          throw error;
        });
      }));
    return summaries.filter((summary) => summary !== undefined)
      .sort((a, b) => b.updatedAt - a.updatedAt || a.id.localeCompare(b.id));
  }

  delete(id: string): Promise<void> {
    return this.withSession(id, async (path) => {
      try {
        // A case-insensitive filesystem can resolve a different spelling of an ID.
        if ((await readdir(this.directory)).includes(basename(path))) await unlink(path);
      } catch (error) {
        if (!isMissing(error)) throw error;
      }
      this.trees.delete(id);
    });
  }

  append(id: string, entry: SessionEntryData): Promise<SessionEntry> {
    return this.withSession(id, async (path) => {
      const tree = await this.load(id, path);
      const line = JSON.stringify({ ...entry, id: randomUUID(), parentId: tree.leafId, timestamp: Date.now() });
      const saved: SessionEntry = JSON.parse(line);
      const file = await openFile(path, constants.O_WRONLY | constants.O_APPEND | constants.O_NOFOLLOW);
      try {
        const { size } = await file.stat();
        try {
          await file.writeFile((tree.needsNewline ? '\n' : '') + line + '\n');
        } catch (error) {
          await file.truncate(size);
          throw error;
        }
      } finally {
        await file.close();
      }
      tree.entries.set(saved.id, saved);
      tree.leafId = saved.id;
      tree.updatedAt = Math.max(tree.updatedAt, saved.timestamp);
      tree.needsNewline = false;
      return structuredClone(saved);
    });
  }

  branch(id: string, fromEntryId: string): Promise<void> {
    return this.withSession(id, async (path) => {
      const tree = await this.load(id, path);
      if (!tree.entries.has(fromEntryId)) throw new Error(`Unknown session entry "${fromEntryId}"`);
      tree.leafId = fromEntryId;
    });
  }

  getBranch(id: string, leafId?: string): Promise<SessionEntry[]> {
    return this.withSession(id, async (path) => {
      const tree = await this.load(id, path);
      return structuredClone(traceBranch(tree, leafId ?? tree.leafId));
    });
  }

  projectActiveMessages(id: string): Promise<SessionProjection> {
    return this.withSession(id, async (path) => {
      const tree = await this.load(id, path);
      const branch = traceBranch(tree, tree.leafId);
      return structuredClone({ ...projectSettings(tree.header, branch), ...projectBranchMessages(branch) });
    });
  }
}
