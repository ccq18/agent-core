import { appendFile, mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { JsonlTreeSessionStore } from '../src/index.js';

const user = (content: string) => ({ role: 'user' as const, content, timestamp: Date.now() });
let dataDir: string;
let store: JsonlTreeSessionStore;

beforeEach(async () => {
  await mkdir('node_modules/.test-tmp', { recursive: true });
  dataDir = await mkdtemp(join('node_modules/.test-tmp', 'store-'));
  store = new JsonlTreeSessionStore(dataDir);
});

afterEach(async () => {
  vi.restoreAllMocks();
  await rm(dataDir, { recursive: true, force: true });
});

describe('JsonlTreeSessionStore', () => {
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
