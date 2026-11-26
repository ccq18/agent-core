import { rm } from 'node:fs/promises';
import { getCurrentTools, type JsonObject } from '@earendil-works/pi-ai';
import { afterEach, beforeEach, expect, it } from 'vitest';
import { type AgentCoreEvent, type AgentRuntime, type PendingQuestionRequest } from '../src/index.js';
import { call, reply, runtime, testDirectory } from './harness-fixtures.js';

let dir: string;
const cores: AgentRuntime[] = [];
beforeEach(async () => { dir = await testDirectory(); });
afterEach(async () => {
  try {
    await Promise.all(cores.splice(0).map((core) => core.dispose()));
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

const questions: JsonObject[] = [
  { id: 'target', question: 'Which target?', options: [{ label: 'Build', description: 'Run the build' }] },
  { id: 'checks', question: 'Which checks?', multiSelect: true },
];

it.each(['answer', 'cancel', 'abort'] as const)('resolves a pending question via %s', async (action) => {
  const requested = Promise.withResolvers<PendingQuestionRequest>();
  const events: AgentCoreEvent[] = [];
  let result: unknown;
  const core = runtime(dir, (model, context) => {
    const last = context.messages.findLast(({ role }) => role === 'toolResult');
    if (last?.role !== 'toolResult') return reply(model, call('question', { questions }));
    const block = last.content[0];
    if (block.type === 'text') result = JSON.parse(block.text);
    return reply(model, { type: 'text', text: 'Continued' });
  }, { truncation: { maxBytes: 32 } });
  cores.push(core);
  const session = await core.createSession({ mode: 'plan' });
  session.subscribe((event) => {
    events.push(event);
    if (event.type === 'question_requested') requested.resolve(event.request);
  });
  const run = session.prompt('Ask before deciding');
  const request = await Promise.race([requested.promise, run.then((turn) => {
    throw new Error(`The run ended without requesting a question: ${turn.error ?? turn.state}`);
  })]);
  expect(request).toMatchObject({ sessionId: session.id, toolCallId: 'call-question', questions });
  expect(request.createdAt).toBeGreaterThan(0);
  expect(session.getPendingQuestions()).toEqual([request]);
  expect(events.some(({ type }) => type === 'tool_end')).toBe(false);
  const answers = { target: 'Build with a free-form answer beyond the tool output limit', checks: ['typecheck', 'test'] };
  if (action === 'abort') session.abort();
  else session.replyQuestion(request.id, action === 'cancel' ? { cancelled: true } : answers);
  const turn = await run;
  expect(request.runId).toBe(turn.runId);
  expect(turn.state).toBe(action === 'abort' ? 'cancelled' : 'completed');
  if (action !== 'abort') expect(result).toEqual(action === 'cancel' ? { cancelled: true } : { answers, cancelled: false });
  expect(session.getPendingQuestions()).toEqual([]);
  expect(events.filter(({ type }) => type === 'question_resolved')).toEqual([
    expect.objectContaining({
      requestId: request.id, cancelled: action !== 'answer',
      ...(action === 'answer' ? { answers } : {}),
    }),
  ]);
});

it('hides the question tool from both catalog and requests when disabled', async () => {
  const core = runtime(dir, (model, context) => {
    expect(getCurrentTools(context.messages).map(({ name }) => name)).not.toContain('question');
    return reply(model, { type: 'text', text: 'Disabled' });
  }, { enableQuestionTool: false });
  cores.push(core);
  expect(core.getCatalog().tools.map(({ name }) => name)).not.toContain('question');
  const session = await core.createSession();
  await expect(session.prompt('Do not ask')).resolves.toMatchObject({ text: 'Disabled' });
});
