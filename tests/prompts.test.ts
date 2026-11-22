import { rm } from 'node:fs/promises';
import { getCurrentTools, type TranscriptContext } from '@earendil-works/pi-ai';
import { expect, it } from 'vitest';
import { DEFAULT_PROMPT_TEMPLATES, resolvePromptTemplates } from '../src/index.js';
import { call, reply, runtime, testDirectory } from './harness-fixtures.js';

it('resolves every default segment and uses the legacy persona when an override is absent', () => {
  expect(Object.keys(DEFAULT_PROMPT_TEMPLATES).sort()).toEqual([
    'basePersona', 'buildModeEnter', 'compactionSummary', 'harnessProtocol', 'maxStepsSummary',
    'planModeEnter', 'questionDescription', 'subagentDescription', 'todoWriteDescription', 'workingDiscipline',
  ]);
  expect(Object.values(DEFAULT_PROMPT_TEMPLATES).every((value) => value.trim())).toBe(true);
  expect(resolvePromptTemplates(undefined, 'Legacy persona').basePersona).toBe('Legacy persona');
  expect(resolvePromptTemplates({ basePersona: 'New persona', workingDiscipline: '' }, 'Legacy persona'))
    .toMatchObject({ basePersona: 'New persona', workingDiscipline: '', harnessProtocol: DEFAULT_PROMPT_TEMPLATES.harnessProtocol });
});

it('uses host templates in the transcript, reminders, tool declarations and compaction request', async () => {
  const dir = await testDirectory();
  const contexts: TranscriptContext[] = [];
  const core = runtime(dir, (model, context) => {
    contexts.push(context);
    return reply(model, { type: 'text', text: 'Done' });
  }, {
    baseSystemPrompt: 'Legacy persona',
    prompts: {
      basePersona: 'Project leader',
      harnessProtocol: 'Custom protocol',
      workingDiscipline: 'Custom discipline',
      planModeEnter: '<system-update>Custom plan reminder</system-update>',
      buildModeEnter: 'Custom build reminder',
      todoWriteDescription: 'Custom todo',
      subagentDescription: 'Custom delegation',
      questionDescription: 'Custom question',
      compactionSummary: 'Custom checkpoint',
    },
    compaction: { keepRecentTokens: 4, reserveTokens: 0 },
  });
  try {
    expect(core.getCatalog().tools).toEqual(expect.arrayContaining([
      expect.objectContaining({ name: 'todowrite', description: 'Custom todo' }),
      expect.objectContaining({ name: 'subagent', description: 'Custom delegation' }),
      expect.objectContaining({ name: 'question', description: 'Custom question' }),
    ]));
    const session = await core.createSession();
    await session.prompt('Prior request');
    await session.setMode('plan');
    await session.prompt('Latest plan request');
    await session.setMode('build');
    await session.prompt('Latest build request');
    expect(contexts[0].messages[0]).toMatchObject({
      role: 'system',
      content: expect.stringMatching(/^Project leader[^]*Custom protocol[^]*Custom discipline/),
    });
    expect(getCurrentTools(contexts[1].messages)).toContainEqual(expect.objectContaining({
      name: 'todowrite', description: 'Custom todo',
    }));
    expect(contexts[1].messages.slice(-2)).toMatchObject([
      { role: 'user', content: '<system-update>Custom plan reminder</system-update>' },
      { role: 'user', content: [{ type: 'text', text: 'Latest plan request' }] },
    ]);
    expect(contexts[2].messages.slice(-2)).toMatchObject([
      { role: 'user', content: '<system-update>\nCustom build reminder\n</system-update>' },
      { role: 'user', content: [{ type: 'text', text: 'Latest build request' }] },
    ]);
    expect(contexts[1].messages[0]).toEqual(contexts[0].messages[0]);
    expect(contexts[2].messages[0]).toEqual(contexts[0].messages[0]);
    await expect(session.compact()).resolves.toMatchObject({ status: 'compacted', summary: 'Done' });
    expect(contexts[3].messages[0]).toMatchObject({ role: 'system', content: 'Custom checkpoint' });
  } finally {
    await core.dispose();
    await rm(dir, { recursive: true, force: true });
  }
});

it('applies the max-step template to the final tool-free subagent request', async () => {
  const dir = await testDirectory();
  const core = runtime(dir, (model, context, options) => {
    if (JSON.stringify(context.messages).includes('Delegation request')) {
      const last = context.messages.at(-1);
      if (last?.role !== 'toolResult') {
        return reply(model, call('subagent', { agent: 'brief', description: 'Gather facts', prompt: 'Child request' }));
      }
      expect(last.content).toContainEqual({ type: 'text', text: 'Child conclusion' });
      return reply(model, { type: 'text', text: 'Done' });
    }
    expect(getCurrentTools(context.messages)).toEqual([]);
    expect(options?.toolChoice).toBe('none');
    expect(context.messages.at(-1)).toMatchObject({
      role: 'user', content: '<system-update>\nHost final instructions\n</system-update>',
    });
    return reply(model, { type: 'text', text: 'Child conclusion' });
  }, { prompts: { maxStepsSummary: 'Host final instructions' }, subagents: { brief: { maxSteps: 1 } } });
  try {
    const session = await core.createSession();
    await expect(session.prompt('Delegation request')).resolves.toMatchObject({ state: 'completed', text: 'Done' });
  } finally {
    await core.dispose();
    await rm(dir, { recursive: true, force: true });
  }
});
