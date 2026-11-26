import { randomUUID } from 'node:crypto';
import { constants, type Stats } from 'node:fs';
import { lstat, mkdir, open as openFile, readFile, readdir, rename, unlink, writeFile, type FileHandle } from 'node:fs/promises';
import { basename, join } from 'node:path';
import type { CreateSessionOptions } from '../types/config.js';
import { SessionBatchAppendError } from '../types/session.js';
import type {
  SessionCompactionInput, SessionCompactionWindow, SessionEntry, SessionEntryData, SessionEntryInput,
  SessionHeader, SessionProjection, SessionSnapshot,
  SessionStorageAdapter, SessionSummary,
} from '../types/session.js';

type SummarySettings = Pick<SessionSummary, 'title' | 'mode' | 'modelAlias' | 'metadata'>;
type SummaryEntry = Pick<SessionEntry, 'id' | 'parentId' | 'timestamp'> & (
  Extract<SessionEntryData, { type: 'mode' | 'model' | 'metadata' }> |
  { type: 'message' | 'plan' | 'compaction' }
);
type FileIdentity = Pick<Stats, 'dev' | 'ino' | 'size' | 'mtimeMs' | 'ctimeMs'>;

interface Journal<T> extends SessionSnapshot {
  entries: Map<string, T>;
  /** Undefined for an empty journal, null for a journal with multiple roots. */
  uniqueRootId?: string | null;
  updatedAt: number;
  needsNewline: boolean;
}

interface SessionTree extends Journal<SessionEntry> {
  identity: FileIdentity;
  settings: SummarySettings;
}

const SUMMARY_VERSION = 1;
const LIST_CONCURRENCY = 16;
const MAX_BATCH_ENTRIES = 64;
const MAX_BATCH_BYTES = 1024 * 1024;

interface BatchRecord {
  entry: SessionEntry;
  line: string;
  bytes: number;
}

function createRecord(entry: SessionEntryInput, parentId: string | null): BatchRecord {
  const line = JSON.stringify({ ...entry, id: randomUUID(), parentId, timestamp: Date.now() }) + '\n';
  return { entry: JSON.parse(line), line, bytes: Buffer.byteLength(line) };
}

interface SummaryIndex {
  schemaVersion: number;
  identity: FileIdentity;
  leafId?: string | null;
  summary: SessionSummary;
}

export interface JsonlTreeSessionStoreOptions {
  maxIdleTrees?: number;
  /** Idle tree weight is the size of its JSONL file in bytes. */
  maxIdleBytes?: number;
}

function isValidId(id: string): boolean {
  return typeof id === 'string' && id.length >= 1 && id.length <= 128 && !/[^a-zA-Z0-9_-]/.test(id);
}

function isMissing(error: unknown): boolean {
  return (error as NodeJS.ErrnoException).code === 'ENOENT';
}

function traceBranch<T extends { parentId: string | null }>(tree: Journal<T>, leafId: string | null): T[] {
  const branch: T[] = [];
  for (let id = leafId; id !== null;) {
    const entry = tree.entries.get(id);
    if (!entry) throw new Error(`Unknown session entry "${id}"`);
    branch.push(entry);
    id = entry.parentId;
  }
  return branch.reverse();
}

function fileIdentity({ dev, ino, size, mtimeMs, ctimeMs }: Stats): FileIdentity {
  return { dev, ino, size, mtimeMs, ctimeMs };
}

function sameFile(a: FileIdentity | undefined, b: FileIdentity): boolean {
  return a?.dev === b.dev && a.ino === b.ino && a.size === b.size && a.mtimeMs === b.mtimeMs && a.ctimeMs === b.ctimeMs;
}

function summaryEntry(entry: SessionEntry): SummaryEntry {
  const { id, parentId, timestamp, type } = entry;
  switch (type) {
    case 'mode': return { id, parentId, timestamp, type, mode: entry.mode };
    case 'model': return { id, parentId, timestamp, type, modelAlias: entry.modelAlias };
    case 'metadata': return { id, parentId, timestamp, type, metadata: entry.metadata };
    default: return { id, parentId, timestamp, type };
  }
}

function applySummaryEntry(settings: SummarySettings, entry: SummaryEntry): SummarySettings {
  switch (entry.type) {
    case 'mode': return { ...settings, mode: entry.mode };
    case 'model': return { ...settings, modelAlias: entry.modelAlias };
    case 'metadata': return { ...settings, metadata: { ...settings.metadata, ...entry.metadata } };
    default: return settings;
  }
}

function projectSummarySettings<T extends SummaryEntry>(tree: Journal<T>, leafId = tree.leafId): SummarySettings {
  const { mode, modelAlias, title, metadata } = tree.header;
  return traceBranch(tree, leafId).reduce(applySummaryEntry, { mode, modelAlias, title, metadata });
}

function summarize(
  { id, createdAt }: Pick<SessionHeader, 'id' | 'createdAt'>, updatedAt: number, settings: SummarySettings,
): SessionSummary {
  const { mode, modelAlias, metadata } = settings;
  const title = typeof metadata?.title === 'string' ? metadata.title : settings.title;
  return { id, title, mode, modelAlias, createdAt, updatedAt, metadata: structuredClone(metadata) };
}

function isSummary(summary: SessionSummary | undefined, id: string): summary is SessionSummary {
  return summary?.id === id && (summary.mode === 'build' || summary.mode === 'plan')
    && Number.isFinite(summary.createdAt) && Number.isFinite(summary.updatedAt)
    && summary.updatedAt >= summary.createdAt
    && (summary.title === undefined || typeof summary.title === 'string')
    && (summary.modelAlias === undefined || typeof summary.modelAlias === 'string')
    && (summary.metadata === undefined || (summary.metadata !== null && typeof summary.metadata === 'object'
      && !Array.isArray(summary.metadata)));
}

async function readJournal<T extends SummaryEntry>(
  file: FileHandle, id: string, keep: (entry: SessionEntry) => T,
): Promise<Journal<T>> {
  const entries = new Map<string, T>();
  let header: SessionHeader | undefined;
  let leafId: string | null = null;
  let uniqueRootId: string | null | undefined;
  let updatedAt = 0;
  const pending: string[] = [];
  const readLine = (line: string) => {
    if (!header) {
      header = JSON.parse(line);
      if (header?.type !== 'session' || header.id !== id) throw new Error(`Invalid session header for "${id}"`);
      updatedAt = header.createdAt;
    } else if (line.trim()) {
      const entry: SessionEntry = JSON.parse(line);
      if (entries.has(entry.id) || (entry.parentId !== null && !entries.has(entry.parentId))) {
        throw new Error(`Invalid session tree entry "${entry.id}" in "${id}"`);
      }
      entries.set(entry.id, keep(entry));
      if (entry.parentId === null) uniqueRootId = uniqueRootId === undefined ? entry.id : null;
      leafId = entry.id;
      updatedAt = Math.max(updatedAt, entry.timestamp);
    }
  };
  const stream = file.createReadStream({ start: 0, encoding: 'utf8', autoClose: false });
  for await (const chunk of stream) {
    let start = 0;
    for (let end = chunk.indexOf('\n'); end !== -1; end = chunk.indexOf('\n', start)) {
      const line = chunk.slice(start, end);
      if (pending.length) {
        pending.push(line);
        readLine(pending.join(''));
        pending.length = 0;
      } else {
        readLine(line);
      }
      start = end + 1;
    }
    if (start < chunk.length) pending.push(chunk.slice(start));
  }
  if (pending.length || !header) readLine(pending.join(''));
  return { header: header!, entries, leafId, uniqueRootId, updatedAt, needsNewline: pending.length > 0 };
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

function systemBaseline(entry: SessionEntry | undefined): SessionCompactionWindow['systemBaseline'] {
  return entry?.type === 'message' && entry.message.role === 'system' ? entry.message : undefined;
}

export function projectBranchMessages(
  branch: readonly SessionEntry[], baseline = systemBaseline(branch[0]) ?? null,
): Pick<
  SessionProjection, 'messages' | 'compactedMessageCount'
> {
  const checkpoint = getCompactionIndex(branch);
  const compaction = branch[checkpoint];
  const messages: SessionProjection['messages'] = [];
  let compactedMessageCount: number | undefined;
  if (compaction?.type === 'compaction') {
    if (baseline) messages.push(baseline);
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
  private readonly maxIdleTrees: number;
  private readonly maxIdleBytes: number;
  private readonly trees = new Map<string, SessionTree>();
  private readonly operations = new Map<string, Promise<unknown>>();
  private readonly references = new Map<string, number>();
  private readonly selectedLeaves = new Map<string, string>();
  private readonly idle = new Map<string, number>();
  private idleBytes = 0;

  constructor(dataDir: string, options: JsonlTreeSessionStoreOptions = {}) {
    this.directory = join(dataDir, 'sessions');
    this.maxIdleTrees = options.maxIdleTrees ?? 32;
    this.maxIdleBytes = options.maxIdleBytes ?? 64 * 1024 * 1024;
  }

  retainSession(id: string): void {
    this.references.set(id, (this.references.get(id) ?? 0) + 1);
    this.removeIdle(id);
  }

  releaseSession(id: string): void {
    const count = this.references.get(id);
    if (count === undefined) return;
    if (count > 1) {
      this.references.set(id, count - 1);
    } else {
      this.references.delete(id);
      this.markIdle(id);
    }
  }

  private removeIdle(id: string): void {
    const bytes = this.idle.get(id);
    if (bytes !== undefined) {
      this.idleBytes -= bytes;
      this.idle.delete(id);
    }
  }

  private markIdle(id: string): void {
    const tree = this.trees.get(id);
    if (!tree || this.references.has(id) || this.operations.has(id)) return;
    const { size } = tree.identity;
    this.idle.set(id, size);
    this.idleBytes += size;
    for (const idleId of this.idle.keys()) {
      if (this.idle.size <= this.maxIdleTrees && this.idleBytes <= this.maxIdleBytes) break;
      this.removeIdle(idleId);
      this.trees.delete(idleId);
    }
  }

  private path(id: string): string {
    if (!isValidId(id)) throw new Error(`Invalid session ID: ${JSON.stringify(id)}`);
    return join(this.directory, `${id}.jsonl`);
  }

  private async withSession<T>(id: string, operation: (path: string) => Promise<T>): Promise<T> {
    const path = this.path(id);
    this.removeIdle(id);
    const previous = this.operations.get(id) ?? Promise.resolve();
    const next = previous.catch(() => undefined).then(() => operation(path));
    this.operations.set(id, next);
    try {
      return await next;
    } finally {
      if (this.operations.get(id) === next) {
        this.operations.delete(id);
        this.markIdle(id);
      }
    }
  }

  private summaryPath(path: string): string {
    return path.slice(0, -6) + '.summary.json';
  }

  private async writeSummary(
    path: string, identity: FileIdentity | undefined, leafId: string | null, summary: SessionSummary,
  ): Promise<void> {
    const indexPath = this.summaryPath(path);
    const temporary = indexPath + '.' + randomUUID() + '.tmp';
    try {
      if (!identity) {
        await unlink(indexPath);
        return;
      }
      const index: SummaryIndex = { schemaVersion: SUMMARY_VERSION, identity, leafId, summary };
      await writeFile(temporary, JSON.stringify(index), { flag: 'wx', mode: 0o600 });
      await rename(temporary, indexPath);
    } catch {
      // JSONL is authoritative. A leftover sidecar cannot validate against the changed log.
      await Promise.allSettled([unlink(temporary), unlink(indexPath)]);
    }
  }

  private async readSummary(
    path: string, id: string, identity: FileIdentity,
  ): Promise<Pick<SummaryIndex, 'leafId' | 'summary'> | undefined> {
    try {
      const raw = await readFile(this.summaryPath(path), {
        encoding: 'utf8', flag: constants.O_RDONLY | constants.O_NOFOLLOW | constants.O_NONBLOCK,
      });
      const index: Partial<SummaryIndex> | null = JSON.parse(raw);
      if (index?.schemaVersion === SUMMARY_VERSION && sameFile(index.identity, identity) && isSummary(index.summary, id)) {
        const { title, mode, modelAlias, createdAt, updatedAt, metadata } = index.summary;
        return { leafId: index.leafId, summary: { id, title, mode, modelAlias, createdAt, updatedAt, metadata } };
      }
    } catch {
      // Old, damaged or unreadable indexes can be rebuilt from the log.
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
      let identity: FileIdentity | undefined;
      try {
        await file.writeFile(line + '\n');
        identity = await file.stat().then(fileIdentity, () => undefined);
      } catch (error) {
        await unlink(path);
        throw error;
      } finally {
        await file.close();
      }
      const saved: SessionHeader = JSON.parse(line);
      const journal = { header: saved, leafId: null, entries: new Map<string, SessionEntry>(),
        updatedAt: saved.createdAt, needsNewline: false };
      const settings = projectSummarySettings(journal);
      this.trees.delete(id);
      this.selectedLeaves.delete(id);
      if (identity) this.trees.set(id, { ...journal, identity, settings });
      await this.writeSummary(path, identity, journal.leafId, summarize(saved, saved.createdAt, settings));
      return structuredClone(saved);
    });
  }

  private async load(id: string, path: string, checkIdentity = false): Promise<SessionTree> {
    const cached = this.trees.get(id);
    if (cached) {
      if (!checkIdentity || sameFile(cached.identity, fileIdentity(await lstat(path)))) return cached;
      this.trees.delete(id);
    }
    const file = await openFile(path, constants.O_RDONLY | constants.O_NOFOLLOW);
    try {
      return await this.loadJournal(id, file, fileIdentity(await file.stat()));
    } finally {
      await file.close();
    }
  }

  private async loadJournal(id: string, file: FileHandle, identity: FileIdentity): Promise<SessionTree> {
    const journal = await readJournal(file, id, (entry) => entry);
    if (!sameFile(identity, fileIdentity(await file.stat()))) {
      throw new Error(`Session journal changed while reading "${id}"`);
    }
    const tree = { ...journal, identity, settings: projectSummarySettings(journal) };
    this.selectedLeaf(id, journal);
    this.trees.set(id, tree);
    return tree;
  }

  private selectedLeaf(id: string, journal: Journal<SummaryEntry>, stable = true): string | undefined {
    const selected = this.selectedLeaves.get(id);
    if (selected === undefined || journal.entries.has(selected)) return selected;
    if (stable) this.selectedLeaves.delete(id);
  }

  open(id: string): Promise<SessionSnapshot> {
    return this.withSession(id, async (path) => {
      const tree = await this.load(id, path, true);
      return { header: structuredClone(tree.header), leafId: this.selectedLeaves.get(id) ?? tree.leafId };
    });
  }

  private async listSession(id: string, path: string): Promise<SessionSummary> {
    const tree = this.trees.get(id);
    const selected = this.selectedLeaves.get(id);
    if (tree) {
      const settings = selected === undefined || selected === tree.leafId ? tree.settings : projectSummarySettings(tree, selected);
      return summarize(tree.header, tree.updatedAt, settings);
    }

    const stats = await lstat(path);
    if (!stats.isFile()) throw new Error(`Invalid session file for "${id}"`);
    const identity = fileIdentity(stats);
    const index = await this.readSummary(path, id, identity);
    if (index && (selected === undefined || selected === index.leafId)) return index.summary;

    const file = await openFile(path, constants.O_RDONLY | constants.O_NOFOLLOW);
    try {
      const before = fileIdentity(await file.stat());
      const journal = await readJournal(file, id, summaryEntry);
      const stable = sameFile(before, fileIdentity(await file.stat()));
      const currentIndex = stable && sameFile(identity, before) ? index : undefined;
      const summary = currentIndex?.summary ?? summarize(journal.header, journal.updatedAt, projectSummarySettings(journal));
      if (stable && currentIndex?.leafId === undefined) {
        await this.writeSummary(path, before, journal.leafId, summary);
      }
      const activeLeaf = this.selectedLeaf(id, journal, stable);
      if (activeLeaf === undefined || activeLeaf === journal.leafId) return summary;
      return summarize(journal.header, journal.updatedAt, projectSummarySettings(journal, activeLeaf));
    } finally {
      await file.close();
    }
  }

  async list(): Promise<SessionSummary[]> {
    const files = await readdir(this.directory, { withFileTypes: true }).catch((error: unknown) => {
      if (isMissing(error)) return [];
      throw error;
    });
    const ids = files
      .filter((file) => file.isFile() && file.name.endsWith('.jsonl') && isValidId(file.name.slice(0, -6)))
      .map((file) => file.name.slice(0, -6));
    const summaries: SessionSummary[] = [];
    let next = 0;
    const workers = await Promise.allSettled(Array.from({ length: Math.min(LIST_CONCURRENCY, ids.length) }, async () => {
      while (next < ids.length) {
        const id = ids[next++];
        try {
          summaries.push(await this.withSession(id, (path) => this.listSession(id, path)));
        } catch (error) {
          if (!isMissing(error)) throw error;
        }
      }
    }));
    for (const worker of workers) if (worker.status === 'rejected') throw worker.reason;
    return summaries.sort((a, b) => b.updatedAt - a.updatedAt || a.id.localeCompare(b.id));
  }

  delete(id: string): Promise<void> {
    return this.withSession(id, async (path) => {
      // A case-insensitive filesystem can resolve a different spelling of an ID.
      const files = await readdir(this.directory).catch((error): string[] => {
        if (isMissing(error)) return [];
        throw error;
      });
      const remove = (target: string) => unlink(target).catch((error) => {
        if (!isMissing(error)) throw error;
      });
      if (files.includes(basename(path))) await remove(path);
      this.trees.delete(id);
      this.selectedLeaves.delete(id);
      const indexPath = this.summaryPath(path);
      if (files.includes(basename(indexPath))) await unlink(indexPath).catch(() => undefined);
    });
  }

  append(id: string, entry: SessionEntryData): Promise<SessionEntry> {
    return this.appendEntry(id, entry);
  }

  appendCompaction(
    id: string, expectedLeafId: string, entry: SessionCompactionInput, signal?: AbortSignal,
  ): Promise<SessionEntry | undefined> {
    return this.appendEntry(id, entry, expectedLeafId, signal);
  }

  private appendEntry(id: string, entry: SessionEntryData): Promise<SessionEntry>;
  private appendEntry(
    id: string, entry: SessionCompactionInput, expectedLeafId: string, signal?: AbortSignal,
  ): Promise<SessionEntry | undefined>;
  private appendEntry(
    id: string, entry: SessionEntryData, expectedLeafId?: string, signal?: AbortSignal,
  ): Promise<SessionEntry | undefined> {
    return this.withSession(id, async (path) => {
      const [saved] = await this.appendBlock(id, path, (tree, leafId) => {
        if (expectedLeafId !== undefined && (leafId !== expectedLeafId || entry.type !== 'compaction' ||
            !this.hasAncestor(tree, leafId, entry.firstKeptEntryId))) return undefined;
        return [createRecord(entry, leafId)];
      }, { signal, single: true });
      return structuredClone(saved);
    });
  }

  private hasAncestor(tree: SessionTree, leafId: string, ancestor: string | null): boolean {
    for (let current: string | null = leafId; current !== null; current = tree.entries.get(current)!.parentId) {
      if (current === ancestor) return true;
    }
    return false;
  }

  appendBatch(id: string, entries: SessionEntryInput[]): Promise<SessionEntry[]> {
    return this.withSession(id, async (path) => {
      const saved: SessionEntry[] = [];
      let nextRecord: BatchRecord | undefined;
      while (saved.length < entries.length) {
        try {
          const block = await this.appendBlock(id, path, (tree, parentId) => {
            const records: BatchRecord[] = [];
            let bytes = tree.needsNewline ? 1 : 0;
            for (let i = saved.length; i < entries.length && records.length < MAX_BATCH_ENTRIES; i++) {
              if (!nextRecord || nextRecord.entry.parentId !== parentId) nextRecord = createRecord(entries[i], parentId);
              // A single large JSONL record must remain in one block.
              if (records.length && bytes + nextRecord.bytes > MAX_BATCH_BYTES) break;
              records.push(nextRecord);
              bytes += nextRecord.bytes;
              parentId = nextRecord.entry.id;
              nextRecord = undefined;
            }
            return records;
          });
          saved.push(...block);
        } catch (error) {
          const failure = error instanceof SessionBatchAppendError ? error : new SessionBatchAppendError(0, true, error);
          if (!saved.length) throw failure;
          throw new SessionBatchAppendError(failure.committedCount === undefined ? undefined : saved.length + failure.committedCount,
            failure.retrySafe, failure.cause);
        }
      }
      return structuredClone(saved);
    });
  }

  private async appendBlock(
    id: string, path: string, prepare: (tree: SessionTree, parentId: string | null) => BatchRecord[] | undefined,
    { signal, single = false }: { signal?: AbortSignal; single?: boolean } = {},
  ): Promise<SessionEntry[]> {
    if (signal?.aborted) return [];
    const file = await openFile(path, constants.O_RDWR | constants.O_APPEND | constants.O_NOFOLLOW);
    let batch: { tree: SessionTree; before: FileIdentity; records: BatchRecord[]; text: string } | undefined;
    let failedWrite: { error: unknown } | undefined;
    let uncertain = false;
    let identity: FileIdentity | undefined;
    try {
      const before = fileIdentity(await file.stat());
      const cached = this.trees.get(id);
      const tree = cached && sameFile(cached.identity, before) ? cached : await this.loadJournal(id, file, before);
      const records = prepare(tree, this.selectedLeaves.get(id) ?? tree.leafId);
      if (!records) return [];
      const text = (tree.needsNewline ? '\n' : '') + records.map(({ line }) => line).join('');
      if (signal?.aborted) return [];
      batch = { tree, before, records, text };
      try {
        await file.writeFile(text);
      } catch (error) {
        failedWrite = { error };
        try {
          await file.truncate(before.size);
        } catch (rollback) {
          uncertain = true;
          failedWrite.error = new AggregateError([error, rollback], 'Journal write and rollback failed');
        }
      }
      if (!failedWrite) {
        identity = await file.stat().then(fileIdentity, () => undefined);
        if (identity && identity.size !== before.size + Buffer.byteLength(text)) identity = undefined;
      }
    } finally {
      try {
        await file.close();
      } catch (error) {
        if (!batch) throw error;
        uncertain = true;
        failedWrite ??= { error };
      }
    }
    const { tree, records, before, text } = batch!;
    if (failedWrite) {
      this.trees.delete(id);
      await unlink(this.summaryPath(path)).catch(() => undefined);
      if (!uncertain) throw single ? failedWrite.error : new SessionBatchAppendError(0, true, failedWrite.error);
      const error = await this.recoverAppend(path, before, records, text, tree.needsNewline, failedWrite.error);
      if (!single || error.committedCount !== records.length) {
        if (error.committedCount) this.selectedLeaves.delete(id);
        throw single && error.retrySafe ? failedWrite.error : error;
      }
      // The verified suffix contains the entire single append, so the caller can use its original ID.
      identity = undefined;
    }
    const saved = records.map(({ entry }) => entry);
    const selected = this.selectedLeaves.get(id);
    const settings = selected === undefined || selected === tree.leafId ? tree.settings : projectSummarySettings(tree, selected);
    await this.publishAppend(id, path, tree, settings, saved, identity);
    return saved;
  }

  private async recoverAppend(
    path: string, before: FileIdentity, records: BatchRecord[], text: string, needsNewline: boolean, error: unknown,
  ): Promise<SessionBatchAppendError> {
    let committed: number | undefined;
    try {
      const file = await openFile(path, constants.O_RDWR | constants.O_NOFOLLOW);
      try {
        const current = await file.stat();
        const expected = Buffer.from(text);
        const length = current.size - before.size;
        if (current.ino !== before.ino || current.dev !== before.dev || length < 0 || length > expected.length) {
          throw new Error('Journal identity or size changed');
        }
        const actual = Buffer.allocUnsafe(length);
        for (let read = 0; read < length;) {
          const { bytesRead } = await file.read(actual, read, length - read, before.size + read);
          if (!bytesRead) throw new Error('Journal shortened during recovery');
          read += bytesRead;
        }
        if (!expected.subarray(0, length).equals(actual)) throw new Error('Journal suffix does not match attempted append');
        if (!sameFile(fileIdentity(current), fileIdentity(await file.stat()))) throw new Error('Journal changed during recovery');
        let end = needsNewline ? 1 : 0;
        committed = 0;
        for (const record of records) {
          if (end + record.bytes > length) break;
          end += record.bytes;
          committed++;
        }
        if (committed === 0) end = 0;
        if (end !== length) await file.truncate(before.size + end);
      } finally {
        await file.close();
      }
    } catch (recovery) {
      return new SessionBatchAppendError(committed, false, new AggregateError([error, recovery], 'Journal recovery failed'));
    }
    return new SessionBatchAppendError(committed, true, error);
  }

  private async publishAppend(
    id: string, path: string, tree: SessionTree, settings: SummarySettings, saved: SessionEntry[],
    identity: FileIdentity | undefined,
  ): Promise<void> {
    for (const entry of saved) {
      settings = applySummaryEntry(settings, entry);
      tree.entries.set(entry.id, entry);
      if (entry.parentId === null) tree.uniqueRootId = tree.uniqueRootId === undefined ? entry.id : null;
      tree.leafId = entry.id;
      tree.updatedAt = Math.max(tree.updatedAt, entry.timestamp);
    }
    tree.settings = settings;
    tree.needsNewline = false;
    this.selectedLeaves.delete(id);
    if (identity) tree.identity = identity;
    else this.trees.delete(id);
    await this.writeSummary(path, identity, tree.leafId, summarize(tree.header, tree.updatedAt, settings));
  }

  branch(id: string, fromEntryId: string): Promise<void> {
    return this.withSession(id, async (path) => {
      const tree = await this.load(id, path);
      if (!tree.entries.has(fromEntryId)) throw new Error(`Unknown session entry "${fromEntryId}"`);
      this.selectedLeaves.set(id, fromEntryId);
    });
  }

  getBranch(id: string, leafId?: string): Promise<SessionEntry[]> {
    return this.withSession(id, async (path) => {
      const tree = await this.load(id, path);
      return structuredClone(traceBranch(tree, leafId ?? this.selectedLeaves.get(id) ?? tree.leafId));
    });
  }

  getCompactionWindow(id: string): Promise<SessionCompactionWindow> {
    return this.withSession(id, async (path) => {
      const tree = await this.load(id, path);
      const leafId = this.selectedLeaves.get(id) ?? tree.leafId;
      const entries: SessionEntry[] = [];
      let checkpoint: SessionCompactionInput | undefined;
      let checkpointLength = 0;
      const uniqueRoot = tree.uniqueRootId ? tree.entries.get(tree.uniqueRootId) : undefined;
      let root = uniqueRoot;
      let collect = true;
      for (let current = leafId; current !== null && (collect || !uniqueRoot);) {
        const entry = tree.entries.get(current)!;
        if (collect) {
          entries.push(entry);
          if (!checkpoint && entry.type === 'compaction') {
            checkpoint = entry;
            checkpointLength = entries.length;
            if (checkpoint.firstKeptEntryId === null) collect = false;
          } else if (checkpoint?.firstKeptEntryId === current) {
            collect = false;
          }
        }
        if (!uniqueRoot) root = entry;
        current = entry.parentId;
      }
      // A missing or non-ancestor anchor has the same fallback as a null anchor.
      if (checkpoint && collect) entries.length = checkpointLength;
      return structuredClone({ entries: entries.reverse(), leafId, systemBaseline: systemBaseline(root) });
    });
  }

  projectActiveMessages(id: string): Promise<SessionProjection> {
    return this.withSession(id, async (path) => {
      const tree = await this.load(id, path);
      const branch = traceBranch(tree, this.selectedLeaves.get(id) ?? tree.leafId);
      return structuredClone({ ...projectSettings(tree.header, branch), ...projectBranchMessages(branch) });
    });
  }
}
