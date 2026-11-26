import * as fs from 'node:fs/promises';
import { mkdir, rm, symlink, writeFile } from 'node:fs/promises';
import type { JsonObject, TranscriptContext } from '@earendil-works/pi-ai';
import { join } from 'node:path';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { RulesLoader, SkillsRegistry, type AgentRuntime } from '../src/index.js';
import { call, reply, runtime, testDirectory, tool } from './harness-fixtures.js';

vi.mock('node:fs/promises', { spy: true });

let dir: string;
const cores: AgentRuntime[] = [];
beforeEach(async () => { dir = await testDirectory(); });
afterEach(async () => {
  try {
    await Promise.all(cores.splice(0).map((core) => core.dispose()));
  } finally {
    vi.restoreAllMocks();
    await rm(dir, { recursive: true, force: true });
  }
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

it('injects newly discovered rules after a tool call, deduplicates aliases and stays inside cwd', async () => {
  const cwd = join(dir, 'project');
  const subdir = join(cwd, 'component');
  const outside = join(dir, 'outside');
  await mkdir(subdir, { recursive: true });
  await mkdir(outside);
  await writeFile(join(cwd, 'AGENTS.md'), 'Baseline rule');
  await writeFile(join(subdir, 'AGENTS.md'), 'Component instructions');
  await writeFile(join(outside, 'AGENTS.md'), 'External instructions');
  await symlink(outside, join(cwd, 'escape'), 'dir');
  const paths: JsonObject[] = [
    { file_path: 'component/file.ts' }, { filePath: join(subdir, 'file.ts') },
    { directory: outside }, { cwd: 'escape' }, { path: 'https://example.test/component/file.ts' },
  ];
  let step = 0;
  const core = runtime(dir, (model, { messages }) => {
    expect(JSON.stringify(messages[0])).toContain('Baseline rule');
    expect(JSON.stringify(messages)).not.toContain('External instructions');
    const updates = messages.filter((message) =>
      message.role === 'user' && typeof message.content === 'string' && message.content.includes('Component instructions'));
    expect(updates).toHaveLength(step ? 1 : 0);
    if (step > 0) expect(updates[0]).toMatchObject({
      content: expect.stringMatching(/<system-update>[^]*AGENTS\.md[^]*Component instructions/),
    });
    const args = paths[step++];
    return args ? reply(model, call('inspect', args)) : reply(model, { type: 'text', text: 'done' });
  }, { tools: [tool('inspect', { readOnly: true })] });
  cores.push(core);
  const session = await core.createSession({ cwd });
  await expect(session.prompt('Read a component file')).resolves.toMatchObject({ text: 'done' });
  expect(session.injectRulesForPath(subdir)).toEqual([]);
  expect(session.injectRulesForPath(outside)).toEqual([]);
});

it('lets a session rediscover a dynamic rule after compaction removes the original update', async () => {
  const cwd = join(dir, 'project');
  const subdir = join(cwd, 'component');
  await mkdir(subdir, { recursive: true });
  await writeFile(join(subdir, 'CLAUDE.md'), 'Component instructions');
  const contexts: TranscriptContext[] = [];
  const core = runtime(dir, (model, context) => {
    contexts.push(context);
    return reply(model, { type: 'text', text: 'Done' });
  }, { compaction: { keepRecentTokens: 4, reserveTokens: 0 } });
  cores.push(core);
  const session = await core.createSession({ cwd });
  expect(session.injectRulesForPath('component')).toEqual([join(subdir, 'CLAUDE.md')]);
  await session.prompt('Original request');
  await session.prompt('Keep this long recent request in context');
  await expect(session.compact()).resolves.toMatchObject({ status: 'compacted' });
  expect(session.injectRulesForPath('component')).toEqual([join(subdir, 'CLAUDE.md')]);
  await session.prompt('Return to the component');
  expect(contexts.at(-1)?.messages).toContainEqual(expect.objectContaining({
    role: 'user', content: expect.stringContaining('Component instructions'),
  }));
});

it('skips automatic discovery in a session without cwd', async () => {
  await writeFile(join(dir, 'AGENTS.md'), 'Discovered instructions');
  const core = runtime(dir, (model, { messages }) => {
    if (messages.at(-1)?.role !== 'toolResult') return reply(model, call('inspect', { directory: dir }));
    expect(JSON.stringify(messages)).not.toContain('Discovered instructions');
    return reply(model, { type: 'text', text: 'Done' });
  }, { tools: [tool('inspect', { readOnly: true })] });
  cores.push(core);
  const session = await core.createSession();
  await expect(session.prompt('Inspect the directory')).resolves.toMatchObject({ text: 'Done' });
});

it('deduplicates asynchronous rule lookups and refreshes positive and negative cache entries', async () => {
  const component = join(dir, 'component');
  await mkdir(component);
  const loader = new RulesLoader();
  const targets = [join(component, 'first.txt'), join(component, 'second.txt')];
  const discover = () => loader.discoverForPathsAsync(targets, new Set(), dir);
  const realpath = vi.mocked(fs.realpath);
  const open = vi.mocked(fs.open);
  realpath.mockClear();
  open.mockClear();
  expect(await discover()).toEqual([]);
  const missing = join(component, 'AGENTS.md');
  expect(realpath.mock.calls.filter(([file]) => file === missing)).toHaveLength(1);
  expect(await discover()).toEqual([]);
  expect(realpath.mock.calls.filter(([file]) => file === missing)).toHaveLength(1);
  const rule = join(component, 'CLAUDE.md');
  await writeFile(rule, 'new');
  const sameTime = await Promise.all([discover(), discover(), discover()]);
  const initial = [{ path: rule, rule: 'new' }];
  expect(sameTime).toEqual([initial, initial, initial]);
  expect(open.mock.calls.filter(([file]) => file === rule)).toHaveLength(1);
  expect(await discover()).toEqual(initial);
  expect(open.mock.calls.filter(([file]) => file === rule)).toHaveLength(1);
  await writeFile(rule, 'updated contents');
  expect(await discover()).toEqual([{ path: rule, rule: 'updated contents' }]);
  expect(open.mock.calls.filter(([file]) => file === rule)).toHaveLength(2);
  await writeFile(join(dir, 'AGENTS.md'), 'root');
  expect(await loader.discoverForProjectAsync(component)).toContainEqual({ path: rule, rule: 'updated contents' });
});

it('skips an unreadable loaded rule without losing discovery of its parent directory', async () => {
  const child = join(dir, 'child');
  const loaded = join(child, 'AGENTS.md');
  const rootRule = join(dir, 'AGENTS.md');
  await mkdir(child);
  await writeFile(rootRule, 'Parent rule');
  await writeFile(loaded, 'Loaded child rule');
  await writeFile(join(child, 'CLAUDE.md'), 'Lower priority rule');
  const { open } = await vi.importActual<typeof fs>('node:fs/promises');
  const accessError = Object.assign(new Error('Child rule is now unreadable'), { code: 'EACCES' });
  vi.mocked(fs.open).mockImplementation(async (...args) => {
    if (args[0] === loaded) throw accessError;
    return open(...args);
  });
  const loader = new RulesLoader();
  const [alreadyLoaded, unreadable] = await Promise.allSettled([
    loader.discoverForPathAsync(child, new Set([loaded]), dir),
    loader.discoverForPathAsync(child, new Set(), dir),
  ]);
  expect(alreadyLoaded).toEqual({ status: 'fulfilled', value: [{ path: rootRule, rule: 'Parent rule' }] });
  expect(unreadable).toEqual({ status: 'rejected', reason: accessError });
  expect(vi.mocked(fs.open).mock.calls.filter(([path]) => path === loaded)).toHaveLength(1);
});

it('can scan skills asynchronously with the same precedence as the synchronous registry', async () => {
  const a = join(dir, 'a');
  const b = join(dir, 'b');
  await mkdir(join(a, 'nested'), { recursive: true });
  await mkdir(b);
  await writeFile(join(a, 'nested', 'SKILL.md'), '---\nname: duplicate\ndescription: first\n---\nbody');
  await writeFile(join(b, 'duplicate.md'), '---\nname: duplicate\ndescription: second\n---\nbody');
  await writeFile(join(b, 'unique.md'), '---\ndisable-model-invocation: true\n---\nbody');
  const sync = new SkillsRegistry([a, b]);
  const asyncRegistry = await SkillsRegistry.loadAsync([a, b]);
  expect(asyncRegistry.list()).toEqual(sync.list());
  expect(asyncRegistry.modelIndex()).toEqual(sync.modelIndex());
});

it('preserves async path boundaries, symlink checks, .git stop and UTF-8 prefix limits', async () => {
  const root = join(dir, 'root');
  const outside = join(dir, 'outside');
  const repo = join(root, 'repo');
  const component = join(repo, 'component');
  await mkdir(outside);
  await mkdir(component, { recursive: true });
  await writeFile(join(root, 'AGENTS.md'), 'Above nested repo');
  await writeFile(join(repo, '.git'), 'gitdir: external');
  await writeFile(join(repo, 'AGENTS.md'), 'Repository rule');
  await writeFile(join(outside, 'AGENTS.md'), 'External rule');
  await symlink(outside, join(component, 'external'));
  await symlink(join(outside, 'AGENTS.md'), join(component, 'AGENTS.md'));
  const valid = join(component, 'CLAUDE.md');
  await writeFile(valid, 'a'.repeat(256 * 1024 - 1) + '猫');
  const loader = new RulesLoader();
  const found = await loader.discoverForPathsAsync([
    'repo/component/future.txt', 'repo/component/external/future.txt', '../outside/future.txt',
  ], new Set(), root);
  expect(found).toEqual([
    { path: join(repo, 'AGENTS.md'), rule: 'Repository rule' },
    { path: valid, rule: 'a'.repeat(256 * 1024 - 1) },
  ]);
  expect(await loader.discoverForPathAsync('repo/component/external/future.txt', new Set(), root)).toEqual([]);
  await fs.unlink(join(component, 'AGENTS.md'));
  await writeFile(join(component, 'AGENTS.md'), 'Preferred rule');
  expect(await loader.discoverForPathAsync('repo/component/future.txt', new Set(), root)).toEqual([
    { path: join(repo, 'AGENTS.md'), rule: 'Repository rule' },
    { path: join(component, 'AGENTS.md'), rule: 'Preferred rule' },
  ]);
  const levels = Array.from({ length: 8 }, (_, i) => 'level' + i);
  const deep = join(repo, ...levels);
  await mkdir(deep, { recursive: true });
  expect(await loader.discoverForPathAsync(deep, new Set(), root)).toEqual([]);
});

it('limits concurrent async skill directory reads while preserving registry order', async () => {
  const directories = Array.from({ length: 12 }, (_, index) => join(dir, String(index)));
  await Promise.all(directories.map(async (directory, index) => {
    await mkdir(directory);
    await writeFile(join(directory, index + '.md'), 'Body');
  }));
  const { readdir } = await vi.importActual<typeof fs>('node:fs/promises');
  let active = 0;
  let peak = 0;
  const read = vi.spyOn(fs, 'readdir').mockImplementation((async (...args: Parameters<typeof readdir>) => {
    peak = Math.max(peak, ++active);
    try {
      await new Promise<void>((resolve) => setImmediate(resolve));
      return await readdir(...args);
    } finally {
      active--;
    }
  }) as typeof readdir);
  try {
    const registry = await SkillsRegistry.loadAsync(directories);
    expect(registry.list().map(({ id }) => id)).toEqual(directories.map((_, index) => String(index)));
    expect(peak).toBe(8);
  } finally {
    read.mockRestore();
  }
});
