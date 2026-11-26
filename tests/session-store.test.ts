import * as fs from 'node:fs/promises';
import { appendFile, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { JsonlTreeSessionStore, SessionBatchAppendError } from '../src/index.js';
import { legacyStorage, reply, runtime, testDirectory } from './harness-fixtures.js';

vi.mock('node:fs/promises', { spy: true });

const user = (content: string) => ({ role: 'user' as const, content, timestamp: Date.now() });
let dataDir: string;
let store: JsonlTreeSessionStore;

beforeEach(async () => {
  dataDir = await testDirectory();
  store = new JsonlTreeSessionStore(dataDir);
});

afterEach(async () => {
  try {
    vi.clearAllMocks();
    vi.restoreAllMocks();
  } finally {
    await rm(dataDir, { recursive: true, force: true });
  }
});

const journal = (id: string) => join(dataDir, 'sessions', `${id}.jsonl`);
const indexFile = (id: string) => join(dataDir, 'sessions', `${id}.summary.json`);
const cached = (store: JsonlTreeSessionStore) => [...store['trees'].keys()];

describe('JsonlTreeSessionStore', () => {
  it('backfills unindexed JSONL without a final newline and lists with zero history opens on an index hit', async () => {
    await mkdir(join(dataDir, 'sessions'));
    await writeFile(journal('legacy'), [
      { type: 'session', version: 1, id: 'legacy', mode: 'build', createdAt: 1 },
      { type: 'message', id: 'root', parentId: null, timestamp: 2, message: user('Historical body') },
    ].map((record) => JSON.stringify(record)).join('\n'));
    expect(await store.list()).toMatchObject([{ id: 'legacy', updatedAt: 2 }]);
    expect(cached(store)).toEqual([]);
    const index = await readFile(indexFile('legacy'), 'utf8');
    expect(index).not.toContain('Historical body');
    const historyOpen = vi.mocked(fs.open);
    historyOpen.mockClear();
    const cold = new JsonlTreeSessionStore(dataDir);
    expect(await cold.list()).toEqual(await store.list());
    expect(historyOpen.mock.calls.map(([path]) => path)).not.toContain(journal('legacy'));
    expect(cached(cold)).toEqual([]);
    await cold.delete('legacy');
    await expect(fs.stat(indexFile('legacy'))).rejects.toMatchObject({ code: 'ENOENT' });
  });

  it('rebuilds malformed, obsolete and stale indexes and surfaces a damaged journal', async () => {
    const { id } = await store.create({ id: 'edited', title: 'Original' });
    const headerLine = await readFile(journal(id), 'utf8');
    await store.append(id, { type: 'metadata', metadata: { title: 'Updated' } });
    const cold = new JsonlTreeSessionStore(dataDir);
    await writeFile(indexFile(id), '{broken');
    expect(await cold.list()).toMatchObject([{ title: 'Updated' }]);
    const index = JSON.parse(await readFile(indexFile(id), 'utf8'));
    await writeFile(indexFile(id), JSON.stringify({ ...index, schemaVersion: -1 }));
    expect(await cold.list()).toMatchObject([{ title: 'Updated' }]);
    expect(JSON.parse(await readFile(indexFile(id), 'utf8')).schemaVersion).toBe(index.schemaVersion);

    // Same-length inode replacement followed by an external truncation.
    const replacement = journal(id) + '.replacement';
    await writeFile(replacement, (await readFile(journal(id), 'utf8')).replace('Updated', 'Changed'));
    await fs.rename(replacement, journal(id));
    expect(await cold.list()).toMatchObject([{ title: 'Changed' }]);
    await fs.truncate(journal(id), Buffer.byteLength(headerLine));
    expect(await cold.list()).toMatchObject([{ title: 'Original' }]);
    await appendFile(journal(id), '{"broken"');
    await expect(cold.list()).rejects.toThrow(SyntaxError);
    expect(cached(cold)).toEqual([]);
  });

  it('commits JSONL writes and removes stale indexes and temp files when summary publication fails', async () => {
    const { id } = await store.create({ id: 'rename-failure' });
    await fs.stat(indexFile(id));
    const rename = vi.mocked(fs.rename).mockRejectedValueOnce(new Error('index rename failed'));
    await expect(store.append(id, { type: 'metadata', metadata: { title: 'Committed' } })).resolves.toBeDefined();
    await expect(fs.stat(indexFile(id))).rejects.toMatchObject({ code: 'ENOENT' });
    expect(await fs.readdir(join(dataDir, 'sessions'))).toEqual([`${id}.jsonl`]);
    rename.mockRejectedValueOnce(new Error('index rename failed'));
    await expect(store.create({ id: 'create-failure' })).resolves.toMatchObject({ id: 'create-failure' });
    expect((await fs.readdir(join(dataDir, 'sessions'))).sort()).toEqual(['create-failure.jsonl', `${id}.jsonl`]);
    expect(await new JsonlTreeSessionStore(dataDir).list()).toEqual(expect.arrayContaining([
      expect.objectContaining({ id, title: 'Committed' }),
      expect.objectContaining({ id: 'create-failure' }),
    ]));
  });

  it('keeps physical update time, selected settings and sort order through eviction and append', async () => {
    store = new JsonlTreeSessionStore(dataDir, { maxIdleTrees: 0 });
    const clock = vi.spyOn(Date, 'now').mockReturnValue(100);
    const { id } = await store.create({ id: 'a', title: 'Header', metadata: { source: 'header' } });
    const root = await store.append(id, { type: 'mode', mode: 'plan' });
    await store.append(id, { type: 'model', modelAlias: 'fast' });
    const selected = await store.append(id, { type: 'metadata', metadata: { title: 'Selected' } });
    await store.branch(id, root.id);
    clock.mockReturnValue(500);
    await store.append(id, { type: 'metadata', metadata: { title: 'Physical' } });
    await store.create({ id: 'b' });
    clock.mockReturnValue(300);
    await store.append(id, { type: 'mode', mode: 'build' });
    await store.branch(id, selected.id);
    expect(cached(store)).toEqual([]);
    const historyOpen = vi.mocked(fs.open);
    historyOpen.mockClear();
    expect(await store.list()).toMatchObject([
      { id, title: 'Selected', mode: 'plan', modelAlias: 'fast', updatedAt: 500, metadata: { source: 'header', title: 'Selected' } },
      { id: 'b', updatedAt: 500 },
    ]);
    expect(historyOpen.mock.calls.map(([path]) => path)).toContain(journal(id));
    expect(cached(store)).toEqual([]);
    expect(await new JsonlTreeSessionStore(dataDir).list()).toMatchObject([{ id, title: 'Physical', mode: 'build' }, { id: 'b' }]);
    expect(await store.open(id)).toMatchObject({ leafId: selected.id });
    expect((await store.getBranch(id)).at(-1)).toEqual(selected);
    const appended = await store.append(id, { type: 'metadata', metadata: { title: 0 } });
    expect(appended.parentId).toBe(selected.id);
    expect(await new JsonlTreeSessionStore(dataDir).list()).toMatchObject([
      { id, title: 'Header', mode: 'plan', modelAlias: 'fast', updatedAt: 500 }, { id: 'b' },
    ]);
  });

  it('evicts least recently used idle trees by count and bytes while retained trees stay loaded', async () => {
    store = new JsonlTreeSessionStore(dataDir, { maxIdleTrees: 2, maxIdleBytes: 1024 });
    store.retainSession('held');
    store.retainSession('held');
    await store.create({ id: 'held' });
    await store.append('held', { type: 'message', message: user('x'.repeat(2048)) });
    await store.create({ id: 'a' });
    await store.create({ id: 'b' });
    await store.open('a');
    await store.create({ id: 'c' });
    expect(cached(store).sort()).toEqual(['a', 'c', 'held']);
    store.releaseSession('held');
    expect(cached(store)).toContain('held');
    store.releaseSession('held');
    expect(cached(store)).toEqual([]);
    await store.open('a');
    expect(cached(store)).toEqual(['a']);
    await store.append('a', { type: 'message', message: user('x'.repeat(2048)) });
    expect(cached(store)).toEqual([]);
  });

  it.each(['list', 'open', 'getBranch', 'append'] as const)(
    'falls back to the replacement journal after eviction when %s first sees a missing selected leaf',
    async (operation) => {
      store = new JsonlTreeSessionStore(dataDir, { maxIdleTrees: 0 });
      const { id } = await store.create({ id: 'replaced' });
      const kept = await store.append(id, { type: 'metadata', metadata: { title: 'Replacement leaf' } });
      const replacement = await readFile(journal(id), 'utf8');
      const removed = await store.append(id, { type: 'mode', mode: 'plan' });
      await store.append(id, { type: 'message', message: user('Old physical leaf') });
      await store.branch(id, removed.id);
      expect(cached(store)).toEqual([]);
      await writeFile(journal(id) + '.replacement', replacement);
      await fs.rename(journal(id) + '.replacement', journal(id));

      if (operation === 'list') {
        expect(await store.list()).toMatchObject([{ id, title: 'Replacement leaf', mode: 'build' }]);
      } else if (operation === 'open') {
        expect(await store.open(id)).toMatchObject({ leafId: kept.id });
      } else if (operation === 'getBranch') {
        expect(await store.getBranch(id)).toEqual([kept]);
      } else {
        expect(await store.append(id, { type: 'message', message: user('New entry') }))
          .toMatchObject({ parentId: kept.id });
      }
      expect(store['selectedLeaves'].has(id)).toBe(false);
      await expect(store.branch(id, removed.id)).rejects.toThrow(/Unknown session entry/);
      expect(await store.list()).toMatchObject([{ id, title: 'Replacement leaf', mode: 'build' }]);
      expect(cached(store)).toEqual([]);
    },
  );

  it.each([{ maxIdleTrees: 0 }, { maxIdleBytes: 1024 }])(
    'only retains the selected leaf ID when large metadata exceeds the idle budget %j',
    async (options) => {
      store = new JsonlTreeSessionStore(dataDir, options);
      const { id } = await store.create({ id: 'large-branch', title: 'Header' });
      const payload = 'x'.repeat(256 * 1024);
      const selected = await store.append(id, { type: 'metadata', metadata: { title: 'Selected', payload } });
      await store.branch(id, selected.id);
      const historyOpen = vi.mocked(fs.open);
      historyOpen.mockClear();
      expect(await store.list()).toMatchObject([{ id, title: 'Selected' }]);
      expect(historyOpen.mock.calls.map(([path]) => path)).not.toContain(journal(id));
      await store.append(id, { type: 'metadata', metadata: { title: 'Physical', payload: '' } });
      await store.branch(id, selected.id);
      expect(cached(store)).toEqual([]);
      expect(store['selectedLeaves'].get(id)).toBe(selected.id);
      expect(await store.list()).toMatchObject([{ id, title: 'Selected', metadata: { payload } }]);
      expect(store['selectedLeaves'].get(id)).toBe(selected.id);
      expect(cached(store)).toEqual([]);
    },
  );

  it('protects the tree during an in-flight append after its last handle reference is released', async () => {
    store = new JsonlTreeSessionStore(dataDir, { maxIdleTrees: 0 });
    store.retainSession('busy');
    await store.create({ id: 'busy' });
    const publishing = Promise.withResolvers<void>();
    const release = Promise.withResolvers<void>();
    const actual = await vi.importActual<typeof fs>('node:fs/promises');
    vi.mocked(fs.rename).mockImplementationOnce(async (...args) => {
      publishing.resolve();
      await release.promise;
      await actual.rename(...args);
    });
    const appending = store.append('busy', { type: 'mode', mode: 'plan' });
    try {
      await publishing.promise;
      store.releaseSession('busy');
      await store.create({ id: 'pressure' });
      expect(cached(store)).toEqual(['busy']);
    } finally {
      release.resolve();
      await appending;
    }
    expect(cached(store)).toEqual([]);
    expect(await store.list()).toContainEqual(expect.objectContaining({ id: 'busy', mode: 'plan' }));
  });

  it('lists created sessions immediately, orders by latest update and deletes by ID', async () => {
    const clock = vi.spyOn(Date, 'now').mockReturnValue(1000);
    const first = await store.create({ title: 'First' });
    clock.mockReturnValue(2000);
    const second = await store.create();
    expect((await store.list()).map(({ id }) => id)).toEqual([second.id, first.id]);
    clock.mockReturnValue(3000);
    await store.append(first.id, { type: 'metadata', metadata: { title: 'Renamed' } });
    expect(await store.list()).toMatchObject([{ id: first.id, title: 'Renamed', updatedAt: 3000 }, { id: second.id }]);
    await store.delete(first.id);
    expect(await store.list()).toMatchObject([{ id: second.id }]);
    await expect(store.open(first.id)).rejects.toThrow();
  });

  it('reports log deletion as successful when the summary sidecar cannot be unlinked', async () => {
    const { id } = await store.create({ id: 'blocked-sidecar' });
    const { id: survivor } = await store.create({ id: 'survivor' });
    const leaf = await store.append(id, { type: 'mode', mode: 'plan' });
    await store.branch(id, leaf.id);
    await fs.unlink(indexFile(id));
    await mkdir(indexFile(id));

    await expect(store.delete(id)).resolves.toBeUndefined();
    await expect(fs.stat(journal(id))).rejects.toMatchObject({ code: 'ENOENT' });
    expect((await fs.stat(indexFile(id))).isDirectory()).toBe(true);
    expect(await store.list()).toMatchObject([{ id: survivor }]);
    expect(cached(store)).not.toContain(id);
    expect(store['selectedLeaves'].has(id)).toBe(false);
  });

  it.each(['../other', 'a/b', '/tmp/absolute', 'a\\b', 'a.jsonl', 'valid\n', '', 'x'.repeat(129)])(
    'rejects invalid session ID %j across the storage API', async (id) => {
      await expect(store.create({ id })).rejects.toThrow(/invalid session id/i);
      await expect(store.open(id)).rejects.toThrow(/invalid session id/i);
      await expect(store.delete(id)).rejects.toThrow(/invalid session id/i);
      await expect(store.append(id, { type: 'mode', mode: 'plan' })).rejects.toThrow(/invalid session id/i);
      await expect(store.branch(id, 'entry')).rejects.toThrow(/invalid session id/i);
      await expect(store.getBranch(id)).rejects.toThrow(/invalid session id/i);
      await expect(store.projectActiveMessages(id)).rejects.toThrow(/invalid session id/i);
    },
  );

  it('rejects duplicate creation and ignores files with invalid IDs during listing', async () => {
    const { id } = await store.create({ id: 'custom_OK-1' });
    await expect(store.create({ id })).rejects.toThrow();
    await store.delete(id.toLowerCase());
    expect(await store.open(id)).toMatchObject({ header: { id } });
    await writeFile(join(dataDir, 'sessions', 'invalid.id.jsonl'), 'invalid');
    expect(await store.list()).toMatchObject([{ id }]);
  });

  it('branches without deleting siblings and reopens with the last appended leaf', async () => {
    const { id } = await store.create({ mode: 'build', modelAlias: 'base' });
    const rootMessage = user('root');
    const root = await store.append(id, { type: 'message', message: rootMessage });
    const original = await store.append(id, { type: 'mode', mode: 'plan' });
    await expect(store.branch(id, 'missing')).rejects.toThrow(/missing/);
    await store.branch(id, root.id);
    const alternative = await store.append(id, { type: 'model', modelAlias: 'other' });
    expect(alternative.parentId).toBe(root.id);
    expect(await store.getBranch(id)).toEqual([root, alternative]);
    expect(await store.getBranch(id, original.id)).toEqual([root, original]);
    await appendFile(join(dataDir, 'sessions', `${id}.jsonl`), '\n\n');
    const reopened = new JsonlTreeSessionStore(dataDir);
    expect(await reopened.open(id)).toMatchObject({ leafId: alternative.id });
    expect(await reopened.projectActiveMessages(id)).toMatchObject({
      messages: [rootMessage], mode: 'build', modelAlias: 'other',
    });
    await reopened.branch(id, original.id);
    expect(await reopened.projectActiveMessages(id)).toMatchObject({ mode: 'plan', modelAlias: 'base' });
  });

  it('appends to a JSONL file whose final record has no newline', async () => {
    const { id } = await store.create({ id: 'x'.repeat(128) });
    const path = join(dataDir, 'sessions', `${id}.jsonl`);
    await writeFile(path, (await readFile(path, 'utf8')).trimEnd());
    const reopened = new JsonlTreeSessionStore(dataDir);
    const entry = await reopened.append(id, { type: 'mode', mode: 'plan' });
    expect(await new JsonlTreeSessionStore(dataDir).getBranch(id)).toEqual([entry]);
  });

  it('serializes overlapping appends into a single recoverable chain', async () => {
    const { id } = await store.create();
    const entries = await Promise.all(['one', 'two', 'three'].map((value) =>
      store.append(id, { type: 'message', message: user(value) })));
    expect(entries.map((entry) => entry.parentId)).toEqual([null, entries[0].id, entries[1].id]);
    expect(await new JsonlTreeSessionStore(dataDir).getBranch(id)).toEqual(entries);
  });

  it('uses the latest checkpoint and retains messages and settings from the active branch', async () => {
    const { id } = await store.create();
    await store.append(id, { type: 'message', message: user('discard me') });
    const oldKept = await store.append(id, { type: 'message', message: user('also discard me') });
    await store.append(id, { type: 'compaction', summary: 'old summary', firstKeptEntryId: oldKept.id, tokensBefore: 50 });
    await store.append(id, { type: 'mode', mode: 'plan' });
    await store.append(id, { type: 'model', modelAlias: 'fast' });
    const plan = [{ id: 'p1', content: 'Check output', status: 'completed' as const }];
    await store.append(id, { type: 'plan', items: plan });
    const keptMessage = user('keep me');
    const kept = await store.append(id, { type: 'message', message: keptMessage });
    await store.append(id, { type: 'compaction', summary: 'latest summary', firstKeptEntryId: kept.id, tokensBefore: 100 });
    const latestMessage = user('after checkpoint');
    await store.append(id, { type: 'message', message: latestMessage });
    const projection = await new JsonlTreeSessionStore(dataDir).projectActiveMessages(id);
    expect(projection).toMatchObject({ mode: 'plan', modelAlias: 'fast', plan });
    expect(projection.messages).toEqual([
      expect.objectContaining({ role: 'user', content: expect.stringContaining('latest summary') }),
      keptMessage,
      latestMessage,
    ]);
  });
});

it('batches entries into contiguous journal chains and updates the index once per block', async () => {
  const { id } = await store.create();
  vi.mocked(fs.open).mockClear();
  vi.mocked(fs.rename).mockClear();
  const entries = Array.from({ length: 65 }, (_, i) => ({ type: 'message' as const, message: user(String(i)) }));
  const saved = await store.appendBatch(id, entries);
  expect(vi.mocked(fs.open)).toHaveBeenCalledTimes(2);
  expect(vi.mocked(fs.rename)).toHaveBeenCalledTimes(2);
  expect(saved.map(({ parentId }) => parentId)).toEqual([null, ...saved.slice(0, -1).map(({ id }) => id)]);
  entries[0].message.content = 'Input mutation';
  const first = saved[0];
  if (first.type === 'message' && first.message.role === 'user') first.message.content = 'Output mutation';
  expect((await store.getBranch(id))[0]).toMatchObject({ message: { content: '0' } });
});

it('rolls a failed batch write back to its previous offset and lets the suffix be retried', async () => {
  const { id } = await store.create();
  const before = await readFile(journal(id));
  const open = fs.open.bind(await vi.importActual<typeof fs>('node:fs/promises'));
  vi.mocked(fs.open).mockImplementationOnce(async (...args) => {
    const file = await open(...args);
    const write = file.writeFile.bind(file);
    vi.spyOn(file, 'writeFile').mockImplementationOnce(async (data) => {
      await write(Buffer.from(data as string).subarray(0, 40));
      throw new Error('partial write');
    });
    return file;
  });
  const inputs = [1, 2].map((i) => ({ type: 'message' as const, message: user(String(i)) }));
  await expect(store.appendBatch(id, inputs)).rejects.toMatchObject({ committedCount: 0, retrySafe: true });
  expect(await readFile(journal(id))).toEqual(before);
  await expect(store.appendBatch(id, inputs)).resolves.toHaveLength(2);
});

it('uses the latest checkpoint window and keeps the branch-root system message', async () => {
  const { id } = await store.create();
  const base = await store.append(id, { type: 'message', message: { role: 'system', content: 'base', timestamp: 1 } });
  const old = await store.append(id, { type: 'message', message: user('old') });
  await store.append(id, { type: 'compaction', firstKeptEntryId: old.id, summary: 'old summary', tokensBefore: 10 });
  const kept = await store.append(id, { type: 'message', message: user('kept') });
  const checkpoint = await store.append(id, {
    type: 'compaction', firstKeptEntryId: kept.id, summary: 'latest summary', tokensBefore: 10,
  });
  const leaf = await store.append(id, { type: 'message', message: user('new') });
  expect(await store.getCompactionWindow(id)).toEqual({
    entries: [kept, checkpoint, leaf], leafId: leaf.id, systemBaseline: base.type === 'message' ? base.message : undefined,
  });
});

it('splits by UTF-8 bytes without interleaving concurrent appends', async () => {
  const { id } = await store.create();
  const sizes: number[] = [];
  const { open } = await vi.importActual<typeof fs>('node:fs/promises');
  const openSpy = vi.mocked(fs.open).mockImplementation(async (...args) => {
    const file = await open(...args);
    const write = file.writeFile.bind(file);
    vi.spyOn(file, 'writeFile').mockImplementation(async (data) => {
      sizes.push(Buffer.byteLength(data as string));
      await write(data);
    });
    return file;
  });
  try {
    const contents = ['猫'.repeat(220_000), '舟'.repeat(220_000)];
    const [batch, single, next] = await Promise.all([
      store.appendBatch(id, contents.map((content) => ({ type: 'message', message: user(content) }))),
      store.append(id, { type: 'mode', mode: 'plan' }),
      store.appendBatch(id, [{ type: 'metadata', metadata: { title: 'Last' } }, { type: 'plan', items: [] }]),
    ]);
    expect(sizes).toHaveLength(4);
    expect(sizes.slice(0, 2).every((size) => size > 650_000 && size < 1024 ** 2)).toBe(true);
    expect(await store.getBranch(id)).toEqual([...batch, single, ...next]);
    expect(single.parentId).toBe(batch[1].id);
    expect(next[0].parentId).toBe(single.id);
  } finally {
    openSpy.mockRestore();
  }
});

it('recovers a known prefix after rollback failure on a selected branch and trims its partial tail', async () => {
  const { id } = await store.create();
  const root = await store.append(id, { type: 'mode', mode: 'plan' });
  const abandoned = await store.append(id, { type: 'metadata', metadata: { title: 'Abandoned' } });
  await store.branch(id, root.id);
  const { open } = await vi.importActual<typeof fs>('node:fs/promises');
  vi.mocked(fs.open).mockImplementationOnce(async (...args) => {
    const file = await open(...args);
    const write = file.writeFile.bind(file);
    vi.spyOn(file, 'writeFile').mockImplementationOnce(async (data) => {
      const text = data as string;
      await write(text.slice(0, text.indexOf('\n') + 12));
      throw new Error('short write');
    });
    vi.spyOn(file, 'truncate').mockRejectedValueOnce(new Error('rollback unavailable'));
    return file;
  });
  const entries = ['committed', 'retry me'].map((content) => ({ type: 'message' as const, message: user(content) }));
  await expect(store.appendBatch(id, entries)).rejects.toMatchObject({ committedCount: 1, retrySafe: true });
  const [retried] = await store.appendBatch(id, entries.slice(1));
  const branch = await new JsonlTreeSessionStore(dataDir).getBranch(id);
  expect(branch.map(({ id }) => id)).not.toContain(abandoned.id);
  expect(branch).toMatchObject([root, { parentId: root.id, message: { content: 'committed' } }, retried]);
  expect(retried.parentId).toBe(branch[1].id);
});

it('reports all records of a later block as committed when close fails after the write', async () => {
  const { id } = await store.create();
  const { open } = await vi.importActual<typeof fs>('node:fs/promises');
  vi.mocked(fs.open).mockImplementationOnce(open).mockImplementationOnce(async (...args) => {
    const file = await open(...args);
    const close = file.close.bind(file);
    vi.spyOn(file, 'close').mockImplementationOnce(async () => {
      await close();
      throw new Error('close result lost');
    });
    return file;
  });
  const inputs = Array.from({ length: 65 }, (_, i) => ({ type: 'message' as const, message: user(String(i)) }));
  await expect(store.appendBatch(id, inputs)).rejects.toMatchObject({ committedCount: 65, retrySafe: true });
  expect(await store.getBranch(id)).toHaveLength(65);
  await expect(fs.stat(indexFile(id))).rejects.toMatchObject({ code: 'ENOENT' });
});

it('returns the verified single append after the journal descriptor close reports a failure', async () => {
  const { id } = await store.create();
  const root = await store.append(id, { type: 'message', message: user('Original branch') });
  await store.append(id, { type: 'message', message: user('Abandoned branch') });
  await store.branch(id, root.id);
  const { open } = await vi.importActual<typeof fs>('node:fs/promises');
  vi.mocked(fs.open).mockImplementationOnce(async (...args) => {
    const file = await open(...args);
    const close = file.close.bind(file);
    vi.spyOn(file, 'close').mockImplementationOnce(async () => {
      await close();
      throw new Error('close result lost');
    });
    return file;
  });
  const saved = await store.append(id, { type: 'message', message: user('Committed once') });
  const next = await store.append(id, { type: 'message', message: user('After recovery') });
  expect(await new JsonlTreeSessionStore(dataDir).getBranch(id)).toEqual([root, saved, next]);
  expect((await readFile(journal(id), 'utf8')).match(/"content":"Committed once"/g)).toHaveLength(1);
});

it.each(['in the session queue', 'during journal open'] as const)('skips a compaction cancelled %s', async (stage) => {
  const core = runtime(dataDir, (model) => reply(model, { type: 'text', text: 'Response' }), {
    sessionStorage: store, compaction: { keepRecentTokens: 4, reserveTokens: 0 },
  });
  const entered = Promise.withResolvers<void>();
  const release = Promise.withResolvers<void>();
  let read: Promise<unknown> | undefined;
  try {
    const session = await core.createSession();
    await session.prompt('Old conversation');
    await session.prompt('Recent conversation');
    const { lstat, open } = await vi.importActual<typeof fs>('node:fs/promises');
    const append = store.appendCompaction.bind(store);
    vi.spyOn(store, 'appendCompaction').mockImplementationOnce((...args) => {
      if (stage === 'in the session queue') {
        vi.mocked(fs.lstat).mockImplementationOnce(async (...statArgs) => {
          entered.resolve();
          await release.promise;
          return lstat(...statArgs);
        });
        read = store.open(session.id);
      } else {
        vi.mocked(fs.open).mockImplementationOnce(async (...openArgs) => {
          const file = await open(...openArgs);
          entered.resolve();
          await release.promise;
          return file;
        });
      }
      return append(...args);
    });
    const pending = session.compact();
    try {
      await entered.promise;
      session.abort();
    } finally {
      release.resolve();
      await read;
    }
    await expect(pending).resolves.toEqual({ status: 'skipped', reason: 'aborted' });
    expect((await new JsonlTreeSessionStore(dataDir).getBranch(session.id))
      .some(({ type }) => type === 'compaction')).toBe(false);
  } finally {
    release.resolve();
    await core.dispose();
  }
});

it.each(['single', 'batch'] as const)('flushes only the retryable %s message suffix on close', async (kind) => {
  const error = new Error('Temporary journal failure');
  const append = store.append.bind(store);
  if (kind === 'single') {
    let fail = true;
    vi.spyOn(store, 'append').mockImplementation(async (id, entry) => {
      if (fail && entry.type === 'message' && entry.message.role === 'assistant') {
        fail = false;
        throw error;
      }
      return append(id, entry);
    });
  } else {
    const appendBatch = store.appendBatch.bind(store);
    vi.spyOn(store, 'appendBatch').mockImplementationOnce(async (id, entries) => {
      const committed = await appendBatch(id, entries.slice(0, -1));
      throw new SessionBatchAppendError(committed.length, true, error);
    });
  }
  const core = runtime(dataDir, (model) => reply(model, { type: 'text', text: 'Saved on close' }), {
    sessionStorage: kind === 'single' ? legacyStorage(store) : store, compaction: { enabled: false },
  });
  try {
    const session = await core.createSession();
    await expect(session.prompt('Keep this input')).rejects.toThrow(error.message);
    const messages = async () => (await new JsonlTreeSessionStore(dataDir).projectActiveMessages(session.id))
      .messages.filter(({ role }) => role !== 'system');
    const input = expect.objectContaining({ role: 'user', content: [{ type: 'text', text: 'Keep this input' }] });
    expect(await messages()).toEqual([input]);
    await session.close();
    expect(await messages()).toEqual([
      input,
      expect.objectContaining({ role: 'assistant', content: [{ type: 'text', text: 'Saved on close' }] }),
    ]);
  } finally {
    await core.dispose();
  }
});

it('reports an unknown commit without retrying the message suffix on close', async () => {
  const error = new SessionBatchAppendError(undefined, false, new Error('Journal recovery unavailable'));
  const append = vi.spyOn(store, 'appendBatch').mockRejectedValueOnce(error);
  const core = runtime(dataDir, (model) => reply(model, { type: 'text', text: 'Done' }), {
    sessionStorage: store, compaction: { enabled: false },
  });
  try {
    const session = await core.createSession();
    await expect(session.prompt('Pending message')).rejects.toBe(error);
    const branch = await store.getBranch(session.id);
    await expect(session.close()).rejects.toBe(error);
    expect(append).toHaveBeenCalledOnce();
    expect(await store.getBranch(session.id)).toEqual(branch);
  } finally {
    await core.dispose();
  }
});

it('keeps a batch committed even when its summary index cannot be published', async () => {
  const { id } = await store.create();
  vi.mocked(fs.rename).mockRejectedValueOnce(new Error('index rename failed'));
  const saved = await store.appendBatch(id, [
    { type: 'mode', mode: 'plan' }, { type: 'metadata', metadata: { title: 'Indexed later' } },
  ]);
  await expect(fs.stat(indexFile(id))).rejects.toMatchObject({ code: 'ENOENT' });
  const reopened = new JsonlTreeSessionStore(dataDir);
  expect(await reopened.getBranch(id)).toEqual(saved);
  expect(await reopened.list()).toEqual([expect.objectContaining({ mode: 'plan', title: 'Indexed later' })]);
});
