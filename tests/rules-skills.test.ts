import { mkdir, rm, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { afterEach, beforeEach, expect, it } from 'vitest';
import { RulesLoader, type AgentRuntime } from '../src/index.js';
import { call, reply, runtime, testDirectory } from './harness-fixtures.js';

let dir: string;
const cores: AgentRuntime[] = [];
beforeEach(async () => { dir = await testDirectory(); });
afterEach(async () => {
  await Promise.all(cores.splice(0).map((core) => core.dispose()));
  await rm(dir, { recursive: true, force: true });
});

it('loads global text and files, prefers AGENTS at each level, and stops at the Git root', async () => {
  const root = join(dir, 'project');
  const cwd = join(root, 'component', 'src');
  const global = join(dir, 'GLOBAL.md');
  await mkdir(cwd, { recursive: true });
  await writeFile(join(root, '.git'), 'gitdir: ../external');
  await writeFile(join(dir, 'AGENTS.md'), 'Outside repository');
  await writeFile(global, 'Global file');
  await writeFile(join(root, 'AGENTS.md'), 'Top rule');
  await writeFile(join(root, 'CLAUDE.md'), 'Ignored alternate');
  await writeFile(join(root, 'component', 'CLAUDE.md'), 'Component rule');
  await writeFile(join(cwd, 'AGENTS.md'), 'Leaf rule');
  const core = runtime(dir, (model, context) => {
    expect(context.messages[0]).toMatchObject({
      role: 'system',
      content: expect.stringMatching(/Base prompt[^]*Global text[^]*Global file[^]*Top rule[^]*Component rule[^]*Leaf rule[^]*Environment/),
    });
    expect(JSON.stringify(context.messages[0])).not.toMatch(/Outside repository|Ignored alternate/);
    return reply(model, { type: 'text', text: 'done' });
  }, { globalRules: ['Global text', global, join(dir, '.', 'GLOBAL.md')], baseSystemPrompt: 'Base prompt' });
  cores.push(core);
  const session = await core.createSession({ cwd });
  await expect(session.prompt('Read rules')).resolves.toMatchObject({ text: 'done' });
  expect(core.getCatalog().rules).toEqual(['Global text', 'Global file', 'Top rule', 'Component rule', 'Leaf rule']);
});

it('bounds rule file reads and examines at most eight directories', async () => {
  const levels = [dir];
  for (let index = 0; index < 9; index++) levels.push(join(levels.at(-1)!, String(index)));
  await mkdir(levels.at(-1)!, { recursive: true });
  await writeFile(join(levels[1], 'AGENTS.md'), 'outside of eight levels');
  await writeFile(join(levels[2], 'AGENTS.md'), 'inside eight levels');
  const big = join(dir, 'oversized.md');
  await writeFile(big, `${'a'.repeat(256 * 1024 - 1)}猫`);
  const rules = new RulesLoader([big]).load(levels.at(-1));
  expect(rules).toEqual(['a'.repeat(256 * 1024 - 1), 'inside eight levels']);
});

it('indexes permitted skill metadata, loads skill content on demand, and applies resource permissions', async () => {
  const skillDir = join(dir, 'skills');
  await mkdir(join(skillDir, 'nested', 'guide'), { recursive: true });
  await writeFile(join(skillDir, 'nested', 'guide', 'SKILL.md'),
    '---\nname: guide\ndescription: >-\n  Read the\n  project guide\n---\nSECRET GUIDE BODY\n');
  await writeFile(join(skillDir, 'private.md'),
    '---\nname: private\ndescription: "Private: workflow"\ndisable-model-invocation: true\n---\nPRIVATE BODY\n');
  await writeFile(join(skillDir, 'root.md'), '---\nname: root\ndescription: Root description\n---\nROOT BODY\n');

  let step = 0;
  const core = runtime(dir, (model, context) => {
    const baseline = context.messages[0];
    expect(baseline).toMatchObject({ role: 'system', content: expect.stringContaining('Read the project guide') });
    expect(JSON.stringify(baseline)).not.toMatch(/SECRET GUIDE BODY|PRIVATE BODY|Private: workflow|ROOT BODY/);
    if (step++ === 0) return reply(model, call('load_skill', { id: 'guide' }, 'guide'),
      call('load_skill', { id: 'private' }, 'private'));
    const results = context.messages.filter(({ role }) => role === 'toolResult');
    expect(results).toMatchObject([
      { isError: false, content: [{ text: expect.stringContaining('<skill_content id="guide">\nSECRET GUIDE BODY') }] },
      { isError: true, content: [{ text: expect.stringContaining('private') }] },
    ]);
    expect(JSON.stringify(results)).not.toContain('PRIVATE BODY');
    return reply(model, { type: 'text', text: 'done' });
  }, {
    skillDirs: [skillDir],
    permissions: [
      { action: 'skill', resource: '*', effect: 'deny' },
      { action: 'skill', resource: 'guide', effect: 'allow' },
    ],
  });
  cores.push(core);
  expect(core.getCatalog().skills).toEqual([
    { id: 'guide', name: 'guide', description: 'Read the project guide' },
    { id: 'private', name: 'private', description: 'Private: workflow' },
    { id: 'root', name: 'root', description: 'Root description' },
  ]);
  const session = await core.createSession();
  await expect(session.prompt('Load skills')).resolves.toMatchObject({ text: 'done' });
});
