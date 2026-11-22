import { readFile, rm } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { afterEach, beforeEach, expect, it } from 'vitest';
import { OutputTruncator, type AgentCoreEvent } from '../src/index.js';
import { call, reply, runtime, testDirectory, textResult, tool } from './harness-fixtures.js';

let dir: string;
beforeEach(async () => { dir = await testDirectory(); });
afterEach(async () => { await rm(dir, { recursive: true, force: true }); });

it('passes content and details through unchanged within both limits', async () => {
  const result = { ...textResult('abc\ndef'), details: { progress: 1 } };
  const truncator = new OutputTruncator(dir, { maxLines: 2, maxBytes: 7 });
  expect(await truncator.truncate('session', 'call', result)).toBe(result);
});

it('keeps the allowed lines and writes all text under a safe file name', async () => {
  const result = { content: [...textResult('a\nb').content, ...textResult('c\nd').content] };
  const truncator = new OutputTruncator(dir, { maxLines: 2 });
  const truncated = await truncator.truncate('../session', '..\\unsafe/call', result);
  expect(truncated.content[0]).toMatchObject({ text: expect.stringMatching(/^a\nb\n[^]*truncated/i) });
  expect(truncated.details).toMatchObject({ truncated: true, outputPath: expect.any(String) });
  const outputPath = (truncated.details as { outputPath: string }).outputPath;
  expect(dirname(outputPath)).toBe(join(dir, 'tool-output'));
  expect(await readFile(outputPath, 'utf8')).toBe('a\nb\nc\nd');
});

it('cuts UTF-8 text at complete code points and preserves host details', async () => {
  const result = { ...textResult('A猫😀Z'), details: { source: 'tool' } };
  const cut = await new OutputTruncator(dir, { maxBytes: 7 }).truncate('session', 'call', result);
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
    tools: [tool('large')], truncation: { maxLines: 1 },
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
