import { randomUUID } from 'node:crypto';
import * as fs from 'node:fs/promises';
import { readFile, rm } from 'node:fs/promises';
import { basename, dirname, join } from 'node:path';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { OutputTruncator, type AgentCoreEvent } from '../src/index.js';
import { call, reply, runtime, testDirectory, textResult, tool } from './harness-fixtures.js';

vi.mock('node:fs/promises', { spy: true });

let dir: string;
beforeEach(async () => { dir = await testDirectory(); });
afterEach(async () => {
  try {
    vi.restoreAllMocks();
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

it('passes content and details through unchanged within both limits', async () => {
  const result = { ...textResult('abc\ndef'), details: { progress: 1 } };
  const truncator = new OutputTruncator(dir, { maxLines: 2, maxBytes: 7 });
  expect(await truncator.truncate('session', 'call', result)).toBe(result);
});

it.each(['head', 'head-tail'] as const)('counts a trailing newline consistently with the %s strategy', async (strategy) => {
  const result = textResult('first\n');
  const withinBudget = new OutputTruncator(dir, { maxLines: 1, maxBytes: 6, strategy });
  expect(await withinBudget.truncate('session', 'fits', result)).toBe(result);
  const clipped = await new OutputTruncator(dir, { maxLines: 1, maxBytes: 4, strategy })
    .truncate('session', 'clipped', result);
  expect(clipped.details).toMatchObject({ totalLines: 1, keptLines: 0, omittedLines: 1, totalBytes: 6 });
  expect(await readFile((clipped.details as { outputPath: string }).outputPath, 'utf8')).toBe('first\n');
});

it('keeps the allowed lines and writes all text under a safe file name', async () => {
  const result = { content: [...textResult('a\nb').content, ...textResult('c\nd').content] };
  const truncator = new OutputTruncator(dir, { maxLines: 2, strategy: 'head' });
  const truncated = await truncator.truncate('../session', '..\\unsafe/call', result);
  expect(truncated.content[0]).toMatchObject({ text: expect.stringMatching(/^a\nb\n[^]*truncated/i) });
  expect(truncated.details).toMatchObject({ truncated: true, outputPath: expect.any(String) });
  const outputPath = (truncated.details as { outputPath: string }).outputPath;
  expect(dirname(outputPath)).toBe(join(dir, 'tool-output'));
  expect(await readFile(outputPath, 'utf8')).toBe('a\nb\nc\nd');
});

it('cuts UTF-8 text at complete code points and preserves host details', async () => {
  const result = { ...textResult('A猫😀Z'), details: { source: 'tool' } };
  const cut = await new OutputTruncator(dir, { maxBytes: 7, strategy: 'head' }).truncate('session', 'call', result);
  expect(cut.content[0]).toMatchObject({ text: expect.stringMatching(/^A猫\n[^]*truncated/i) });
  expect(cut.details).toMatchObject({ source: 'tool', truncated: true });
  expect(await readFile((cut.details as { outputPath: string }).outputPath, 'utf8')).toBe('A猫😀Z');
});

it('truncates the composed after-hook result and maps the saved path onto tool_end', async () => {
  const payload = 'first\nsecond\nthird';
  const core = runtime(dir, (model, context) => {
    const result = context.messages.at(-1);
    if (result?.role !== 'toolResult') return reply(model, call('large'));
    expect(result).toMatchObject({ content: [{ text: expect.stringContaining('first') }],
      details: { host: true, truncated: true, outputPath: expect.any(String) } });
    return reply(model, { type: 'text', text: 'done' });
  }, {
    hooks: { afterToolCall: async () => ({ ...textResult(payload), details: { host: true } }) },
    tools: [tool('large')], truncation: { perTool: { large: { maxLines: 1 } } },
  });
  try {
    const session = await core.createSession();
    const events: AgentCoreEvent[] = [];
    session.subscribe((event) => events.push(event));
    await expect(session.prompt('large')).resolves.toMatchObject({ state: 'completed', text: 'done' });
    const event = events.find((item) => item.type === 'tool_end');
    expect(event).toMatchObject({ truncated: true, outputPath: join(dir, 'tool-output', `${session.id}-call-large.txt`) });
    expect(await readFile(event!.outputPath!, 'utf8')).toBe(payload);
  } finally {
    await core.dispose();
  }
});

it('defaults to head-tail and guides the model to read the saved output without rerunning', async () => {
  const output = 'head1\nhead2\nhead3\nmiddle\nstatus\nERROR\ncursor=next';
  const cut = await new OutputTruncator(dir, { maxLines: 4 }).truncate('session', 'call', textResult(output));
  expect(cut.details).toMatchObject({ totalLines: 7, totalBytes: Buffer.byteLength(output),
    keptLines: 4, omittedLines: 3, strategy: 'head-tail' });
  expect(cut.content[0]).toMatchObject({ text: expect.stringMatching(/^head1\nhead2\n[^]*Do not rerun[^]*ERROR\ncursor=next$/) });
});

it('applies the tool-specific budget and strategy', async () => {
  const truncator = new OutputTruncator(dir, {
    perTool: { noisy: { maxLines: 1, strategy: 'head' } },
  });
  const result = textResult('first\nlast');
  expect(await truncator.truncate('session', 'quiet', result, 'quiet')).toBe(result);
  const cut = await truncator.truncate('session', 'noisy', result, 'noisy');
  expect(cut.details).toMatchObject({ keptLines: 1, strategy: 'head' });
  expect(cut.content[0]).toMatchObject({ text: expect.stringMatching(/^first\n\n\[Output truncated\./) });
});

it('writes bounded UTF-8 chunks and keeps both edges across existing text blocks', async () => {
  const { open } = await vi.importActual<typeof fs>('node:fs/promises');
  const sizes: number[] = [];
  const handle = vi.mocked(fs.open).mockImplementation(async (...args) => {
    const file = await open(...args);
    const write = file.writeFile.bind(file);
    vi.spyOn(file, 'writeFile').mockImplementation(async (data) => {
      sizes.push(Buffer.byteLength(data as string));
      await write(data);
    });
    return file;
  });
  const head = '标题\n' + 'a'.repeat(16_380) + '😀' + '猫'.repeat(100_000);
  const tail = '中间\nERROR 猫😀\ncursor=续\n';
  const image = { type: 'image' as const, data: 'AA==', mimeType: 'image/png' };
  try {
    const cut = await new OutputTruncator(dir, { maxBytes: 80 }).truncate('session', 'utf8', {
      content: [...textResult(head).content, image, ...textResult(tail).content],
    });
    const { outputPath } = cut.details as { outputPath: string };
    expect(sizes.length).toBeGreaterThan(2);
    expect(Math.max(...sizes)).toBeLessThanOrEqual(48 * 1024);
    expect(await fs.readFile(outputPath)).toEqual(Buffer.from(head + '\n' + tail));
    expect(cut.content).toContain(image);
    expect(cut.content[0]).toMatchObject({ text: expect.stringMatching(/^标题\n\n\[Output truncated\./) });
    expect(cut.content.at(-1)).toMatchObject({ text: 'ERROR 猫😀\ncursor=续' });
    expect(cut.content).not.toContainEqual(expect.objectContaining({ text: expect.stringContaining('\ufffd') }));
  } finally {
    handle.mockRestore();
  }
});

const managedName = () => 'agent-core-output.' + randomUUID() + '.txt';

it('leaves outputs alone by default and optionally prunes managed files by seven-day TTL and 1 GiB quota', async () => {
  const directory = join(dir, 'tool-output');
  await fs.mkdir(directory);
  const aged = join(directory, managedName());
  const overQuota = join(directory, managedName());
  const newest = join(directory, managedName());
  const unrelated = join(directory, 'notes-owned-by-user.txt');
  const link = join(directory, managedName());
  const now = Date.now();
  const day = 24 * 3600 * 1000;
  for (const path of [aged, overQuota, newest, unrelated]) await fs.writeFile(path, 'keep');
  await fs.truncate(overQuota, 1024 ** 3);
  await fs.utimes(aged, new Date(now - 8 * day), new Date(now - 8 * day));
  await fs.utimes(overQuota, new Date(now - day), new Date(now - day));
  await fs.utimes(unrelated, new Date(now - 20 * day), new Date(now - 20 * day));
  await fs.symlink(unrelated, link);
  const disabled = new OutputTruncator(dir, { maxLines: 1 });
  await disabled.truncate('session', 'disabled', textResult('one\ntwo'));
  await disabled.flushCleanup();
  expect(await fs.stat(aged)).toBeDefined();
  expect(await fs.stat(overQuota)).toBeDefined();

  const enabled = new OutputTruncator(dir, { maxLines: 1, cleanup: { enabled: true } });
  try {
    const cut = await enabled.truncate('session', 'enabled', textResult('one\ntwo'));
    await enabled.flushCleanup();
    expect((await fs.readdir(directory)).sort()).toEqual([
      basename((cut.details as { outputPath: string }).outputPath), basename(newest),
      basename(unrelated), basename(link), 'session-disabled.txt',
    ].sort());
  } finally {
    await enabled.flushCleanup();
  }
});

it('excludes unpublished files from cleanup while a concurrent write is in flight', async () => {
  const truncator = new OutputTruncator(dir, { maxLines: 1, cleanup: { enabled: true, maxTotalBytes: 0 } });
  const entered = Promise.withResolvers<string>();
  const release = Promise.withResolvers<void>();
  const { open } = await vi.importActual<typeof fs>('node:fs/promises');
  vi.mocked(fs.open).mockImplementationOnce(async (...args) => {
    const file = await open(...args);
    const write = file.writeFile.bind(file);
    vi.spyOn(file, 'writeFile').mockImplementationOnce(async (data) => {
      await write(data);
      entered.resolve(String(args[0]));
      await release.promise;
    });
    return file;
  });
  const first = truncator.truncate('session', 'slow', textResult('one\ntwo'));
  try {
    const stage = await entered.promise;
    expect(stage.endsWith('.tmp')).toBe(true);
    await truncator.truncate('session', 'fast', textResult('three\nfour'));
    await truncator.flushCleanup();
    expect(await readFile(stage, 'utf8')).toBe('one\ntwo');
  } finally {
    release.resolve();
    await first;
    await truncator.flushCleanup();
  }
  const result = await first;
  expect(await readFile((result.details as { outputPath: string }).outputPath, 'utf8')).toBe('one\ntwo');
  expect((await fs.readdir(join(dir, 'tool-output'))).some((file) => file.endsWith('.tmp'))).toBe(false);
});

it('protects renamed in-flight outputs and recent files while pruning older files over quota', async () => {
  const directory = join(dir, 'tool-output');
  await fs.mkdir(directory);
  const old = join(directory, managedName());
  const recent = join(directory, managedName());
  await fs.writeFile(old, 'old');
  await fs.writeFile(recent, 'recent');
  const older = new Date(Date.now() - 60_000);
  await fs.utimes(old, older, older);
  const truncator = new OutputTruncator(dir, { maxLines: 1, cleanup: { enabled: true, maxTotalBytes: 0 } });
  const renamed = Promise.withResolvers<string>();
  const release = Promise.withResolvers<void>();
  const { rename } = await vi.importActual<typeof fs>('node:fs/promises');
  vi.mocked(fs.rename).mockImplementationOnce(async (from, to) => {
    await rename(from, to);
    // Simulate an older mtime so the age grace period cannot hide a missing in-flight guard.
    await fs.utimes(to, older, older);
    renamed.resolve(String(to));
    await release.promise;
  });
  const first = truncator.truncate('session', 'slow', textResult('one\ntwo'));
  try {
    const path = await renamed.promise;
    await truncator.truncate('session', 'fast', textResult('three\nfour'));
    await truncator.flushCleanup();
    await expect(readFile(old)).rejects.toMatchObject({ code: 'ENOENT' });
    expect(await readFile(path, 'utf8')).toBe('one\ntwo');
    expect(await readFile(recent, 'utf8')).toBe('recent');
  } finally {
    release.resolve();
    await first;
    await truncator.flushCleanup();
  }
});
