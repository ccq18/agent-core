import assert from 'node:assert/strict';
import { lstat, mkdir, mkdtemp, readFile, readdir, realpath, rm, stat, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { monitorEventLoopDelay, performance } from 'node:perf_hooks';
import { setTimeout as delay } from 'node:timers/promises';
import { createAgentRuntime, JsonlTreeSessionStore } from '../dist/index.js';

const SEED = 0x5eed2026;
const FIXTURE_TIME = Date.UTC(2026, 0, 1);
const MODEL_MS = 5;
const HOT_LIST_MS = 30_000;
const MiB = 1024 ** 2;
const mockModel = {
  alias: 'bench', providerId: 'bench', modelId: 'bench', protocol: 'openai-completions',
  baseUrl: 'http://127.0.0.1:1/v1', contextWindow: 262_144, maxTokens: 512,
};

const textBlock = (text) => ({ type: 'text', text });
const toolResult = (text) => ({ content: [textBlock(text)] });
const toolCall = (name, id, args = {}) => ({ type: 'toolCall', name, id, arguments: args });
const childCall = (id) => toolCall('subagent', id, { agent: 'explore', description: id, prompt: 'Benchmark child' });
const ms = (value) => Number(value.toFixed(3));
const mib = (bytes) => Number((bytes / MiB).toFixed(2));

function seededText(seed) {
  const alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
  let state = seed;
  return (length) => {
    let text = '';
    for (let i = 0; i < length; i++) {
      state ^= state << 13;
      state ^= state >>> 17;
      state ^= state << 5;
      text += alphabet[state & 63];
    }
    return text;
  };
}

function assistant(model, content, timestamp = Date.now(), stopReason = 'stop') {
  return {
    role: 'assistant', model: model.id, provider: model.provider, api: model.api,
    content, stopReason, timestamp,
    usage: {
      input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
    },
  };
}

// The override needs only the public stream contract: a final event and result().
function mockStream(respond, latencyMs = MODEL_MS, onResponse) {
  return (model, context, options) => {
    const blocks = respond(context, options);
    const result = delay(latencyMs).then(() => {
      const stop = blocks.some(({ type }) => type === 'toolCall') ? 'toolUse' : 'stop';
      const message = assistant(model, blocks, Date.now(), stop);
      onResponse?.();
      return message;
    });
    return {
      result: () => result,
      async *[Symbol.asyncIterator]() {
        const message = await result;
        yield { type: 'done', reason: message.stopReason, message };
      },
    };
  };
}

function runtime(dataDir, streamFnOverride, config = {}) {
  return createAgentRuntime({
    dataDir, models: { defaultAlias: mockModel.alias, profiles: [mockModel], streamFnOverride },
    defaultPermissionEffect: 'allow', compaction: { enabled: false }, ...config,
  });
}

function transcript(id, title) {
  const lines = [JSON.stringify({ type: 'session', version: 1, id, title, mode: 'build', createdAt: FIXTURE_TIME })];
  let leaf = null;
  return {
    add(entryId, entry, parentId = leaf) {
      lines.push(JSON.stringify({ ...entry, id: entryId, parentId, timestamp: FIXTURE_TIME + lines.length }));
      leaf = entryId;
    },
    message(entryId, message, parentId = leaf) {
      this.add(entryId, { type: 'message', message }, parentId);
    },
    checkpoint(firstKeptEntryId) {
      this.add('checkpoint', {
        type: 'compaction', summary: 'Previous benchmark checkpoint', firstKeptEntryId, tokensBefore: 40_000,
      });
    },
    write(directory) {
      return writeFile(join(directory, id + '.jsonl'), lines.join('\n') + '\n');
    },
  };
}

function percentiles(samples, ranks = [50, 95, 99]) {
  assert(samples.length > 0, 'No latency samples');
  const ordered = [...samples].sort((a, b) => a - b);
  return Object.fromEntries(ranks.map((p) => [`p${p}Ms`, ms(ordered[Math.ceil(p * ordered.length / 100) - 1])]));
}

function collectedMemory() {
  globalThis.gc?.();
  globalThis.gc?.();
  return process.memoryUsage();
}

function memoryMiB({ heapUsed, rss }) {
  return { heapUsedMiB: mib(heapUsed), rssMiB: mib(rss) };
}

async function listFixture(dataDir, signal) {
  const directory = join(dataDir, 'sessions');
  await mkdir(directory, { recursive: true });
  const nextText = seededText(SEED);
  const titles = new Map();
  for (let session = 0; session < 200; session++) {
    signal.throwIfAborted();
    const id = `list-${session}`;
    const title = `Fixture ${session}`;
    const tree = transcript(id, title);
    for (let i = 0; i < 500; i++) {
      const message = { role: 'user', content: nextText(1024), timestamp: FIXTURE_TIME + i };
      if (i === 440 && session % 2 === 0) {
        tree.message(`m${i}`, message, 'm409');
      } else {
        tree.message(`m${i}`, message);
      }
      if (i === 280 && session % 3 === 0) tree.checkpoint('m239');
      if (i === 420 && session % 2 === 0) {
        tree.add('abandoned-title', { type: 'metadata', metadata: { title: `Abandoned ${id}` } });
      }
    }
    const hasActiveTitle = session % 4 !== 0;
    if (hasActiveTitle) {
      tree.add('title', { type: 'metadata', metadata: { title: `Updated ${id}` } });
    }
    titles.set(id, hasActiveTitle ? `Updated ${id}` : title);
    await tree.write(directory);
  }
  return titles;
}

async function checkList(core, titles) {
  const start = performance.now();
  const summaries = await core.listSessions();
  const elapsed = performance.now() - start;
  assert.equal(summaries.length, titles.size);
  for (const { id, title } of summaries) assert.equal(title, titles.get(id), id);
  return elapsed;
}

async function listBench(dataDir, signal) {
  const titles = await listFixture(dataDir, signal);
  const countIndexes = async () => (await readdir(join(dataDir, 'sessions')))
    .filter((path) => path.endsWith('.summary.json')).length;
  const indexesBefore = await countIndexes();
  assert.equal(indexesBefore, 0, 'The first store must start without summary indexes');
  let first = runtime(dataDir);
  let firstColdMs;
  try {
    firstColdMs = ms(await checkList(first, titles));
  } finally {
    await first.dispose();
    first = undefined;
  }
  const indexesAfterFirstList = await countIndexes();
  // The first store is out of use; the second can consume any on-disk index it created.
  collectedMemory();
  const second = runtime(dataDir);
  try {
    const newStoreColdMs = ms(await checkList(second, titles));
    const afterColdGc = memoryMiB(collectedMemory());
    const samples = [];
    const start = performance.now();
    do {
      signal.throwIfAborted();
      samples.push(await checkList(second, titles));
    } while (performance.now() - start < HOT_LIST_MS);
    return {
      sessions: 200, messagesPerSession: 500, payloadBytes: 1024, firstColdMs, newStoreColdMs,
      indexesBefore, indexesAfterFirstList,
      firstColdMode: 'no-index', newStoreColdMode: indexesAfterFirstList === titles.size ? 'indexes-present' : 'full-or-partial-backfill',
      hotSamples: samples.length, hotWindowMs: ms(performance.now() - start),
      hot: percentiles(samples, [50, 95]), afterColdGc, afterHotGc: memoryMiB(collectedMemory()),
    };
  } finally {
    await second.dispose();
  }
}

async function subagentMemoryBench(dataDir, signal) {
  const parentId = 'memory-parent';
  let issued = 0;
  const stream = mockStream((context, options) => {
    if (options.sessionId === parentId && context.messages.at(-1)?.role === 'user') {
      return [childCall(`memory-child-${issued++}`)];
    }
    return [textBlock('Benchmark done')];
  });
  const core = runtime(dataDir, stream);
  try {
    const parent = await core.createSession({ id: parentId });
    const completed = new Set();
    let failed = 0;
    parent.subscribe((event) => {
      if (event.type !== 'subagent_completed') return;
      if (event.state === 'completed') completed.add(event.subagentId);
      else failed++;
    });
    const before = collectedMemory();
    const start = performance.now();
    for (let i = 0; i < 100; i++) {
      signal.throwIfAborted();
      const result = await parent.prompt(`Run child ${i}`);
      assert.equal(result.state, 'completed', result.error);
    }
    const elapsedMs = ms(performance.now() - start);
    // Keep the one runtime and its completed child sessions alive through GC.
    const after = collectedMemory();
    assert.equal(issued, 100);
    assert.equal(failed, 0);
    assert.equal(completed.size, 100);
    const sessions = await core.listSessions();
    assert.equal(sessions.filter(({ metadata }) => metadata?.parentSessionId === parentId).length, 100);
    return {
      children: completed.size, elapsedMs, beforeGc: memoryMiB(before), afterGc: memoryMiB(after),
      retainedHeapDeltaMiB: mib(after.heapUsed - before.heapUsed),
    };
  } finally {
    await core.dispose();
  }
}

class TimedMessageStore extends JsonlTreeSessionStore {
  openTurns = new Map();

  async append(id, entry) {
    const turn = this.openTurns.get(id);
    if (!turn || entry.type !== 'message') return super.append(id, entry);
    turn.firstAppend ??= performance.now();
    const saved = await super.append(id, entry);
    this.record(turn, [saved]);
    return saved;
  }

  async appendBatch(id, entries) {
    const turn = this.openTurns.get(id);
    if (turn) turn.firstAppend ??= performance.now();
    const saved = await super.appendBatch(id, entries);
    if (turn) this.record(turn, saved);
    return saved;
  }

  record(turn, records) {
    turn.lastAppend = performance.now();
    for (const saved of records) {
      if (saved.type !== 'message') continue;
      turn.messages++;
      turn.rulePaths += saved.message.rulePaths?.length ?? 0;
    }
  }
}

async function persistBench(dataDir, signal) {
  const concurrency = 8;
  const rounds = 24;
  const outputBytes = 128 * 1024;
  const largeText = seededText(SEED ^ 1)(outputBytes);
  const storage = new TimedMessageStore(dataDir);
  const cwd = join(dataDir, 'project');
  const targets = [];
  for (let wave = 0; wave < rounds; wave++) {
    const target = join(`wave-${wave}`, 'level1', 'level2', 'level3', 'target.txt');
    targets.push(target);
    await mkdir(join(cwd, dirname(target)), { recursive: true });
    await writeFile(join(cwd, target), 'Benchmark target\n');
    let directory = join(cwd, dirname(target));
    for (let depth = 0; depth < 4; depth++) {
      await writeFile(join(directory, depth % 2 ? 'CLAUDE.md' : 'AGENTS.md'), `# Fixture rule ${wave}/${depth}\n`);
      directory = dirname(directory);
    }
  }
  await writeFile(join(cwd, '.git'), 'Benchmark rules search boundary\n');
  const stream = mockStream((context) => {
    const prompt = context.messages.findLast(({ role }) => role === 'user');
    const input = typeof prompt?.content === 'string' ? prompt.content : prompt?.content?.[0]?.text;
    if (context.messages.at(-1)?.role === 'toolResult' || !input?.startsWith('Persist wave ')) {
      return [textBlock('Persist done')];
    }
    const wave = Number(input.slice('Persist wave '.length));
    return Array.from({ length: 28 }, (_, i) =>
      toolCall('fixture_output', `persist-${wave}-${i}`, i === 0 ? { path: targets[wave], large: true } : {}));
  });
  const core = runtime(dataDir, stream, {
    sessionStorage: storage, truncation: { maxBytes: 4096 },
    tools: [{
      name: 'fixture_output', label: 'Fixture output', description: 'Return benchmark text',
      parameters: { type: 'object', properties: { path: { type: 'string' }, large: { type: 'boolean' } } },
      execute: async ({ large }) => toolResult(large ? largeText : 'Small benchmark result'),
    }],
  });
  const eventLoop = monitorEventLoopDelay({ resolution: 1 });
  try {
    const sessions = [];
    const initialRecords = new Map();
    for (let i = 0; i < concurrency; i++) {
      const id = `persist-${i}`;
      const session = await core.createSession({ id, cwd });
      // Establish the system message and tool declarations outside the measured turns.
      const warmup = await session.prompt('Prepare persistence fixture');
      assert.equal(warmup.state, 'completed', warmup.error);
      initialRecords.set(id, 1 + (await storage.getBranch(id)).length);
      session.subscribe((event) => {
        const turn = storage.openTurns.get(id);
        if (!turn) return;
        if (event.type === 'turn_end') turn.turnEnd = performance.now();
        if (event.type === 'tool_end') {
          turn.tools++;
          if (event.isError) turn.errors++;
          if (event.truncated) turn.outputPath = event.outputPath;
        }
      });
      sessions.push(session);
    }
    const allTurns = [];
    eventLoop.enable();
    const start = performance.now();
    try {
      for (let wave = 0; wave < rounds; wave++) {
        signal.throwIfAborted();
        const batch = await Promise.allSettled(sessions.map(async (session) => {
          const turn = { messages: 0, rulePaths: 0, tools: 0, errors: 0 };
          storage.openTurns.set(session.id, turn);
          try {
            const result = await session.prompt(`Persist wave ${wave}`);
            turn.returned = performance.now();
            assert.equal(result.state, 'completed', result.error);
            assert.equal(result.text, 'Persist done');
            assert.equal(turn.messages, 32, 'Expected 32 new stored messages per prompt');
            assert.equal(turn.rulePaths, 4, 'Expected one rule update covering four directories');
            assert.equal(turn.tools, 28);
            assert.equal(turn.errors, 0);
            assert(turn.outputPath && turn.turnEnd !== undefined, 'Missing truncation or final turn_end');
            return turn;
          } finally {
            storage.openTurns.delete(session.id);
          }
        }));
        for (const result of batch) {
          if (result.status === 'rejected') throw result.reason;
          allTurns.push(result.value);
        }
      }
    } finally {
      eventLoop.disable();
    }
    const workloadMs = ms(performance.now() - start);
    assert(eventLoop.count > 0, 'No event loop delay samples');
    for (const { outputPath } of allTurns) assert.equal((await stat(outputPath)).size, outputBytes);
    for (const { id } of sessions) {
      const jsonl = await readFile(join(dataDir, 'sessions', id + '.jsonl'), 'utf8');
      assert.equal(jsonl.trimEnd().split('\n').length, initialRecords.get(id) + rounds * 32);
    }
    return {
      concurrency, rounds, turns: allTurns.length, newMessagesPerTurn: 32, outputBytes, workloadMs,
      persistence: percentiles(allTurns.map((turn) => turn.lastAppend - turn.firstAppend)),
      finalTurnEndToPromptReturn: percentiles(allTurns.map((turn) => turn.returned - turn.turnEnd)),
      eventLoopDelayP99Ms: ms(eventLoop.percentile(99) / 1e6), eventLoopSamples: eventLoop.count,
    };
  } finally {
    await core.dispose();
  }
}

async function compactionBench(dataDir) {
  const directory = join(dataDir, 'sessions');
  await mkdir(directory, { recursive: true });
  const id = 'compaction';
  const tree = transcript(id, 'Compaction fixture');
  const nextText = seededText(SEED ^ 2);
  const messageModel = { id: mockModel.modelId, provider: mockModel.providerId, api: mockModel.protocol };
  for (let i = 0; i < 400; i++) {
    const timestamp = FIXTURE_TIME + i;
    const text = nextText(1024);
    const message = i % 2 === 0 ? { role: 'user', content: text, timestamp } :
      assistant(messageModel, [textBlock(text)], timestamp);
    tree.message(`m${i}`, message);
    if (i === 159) tree.checkpoint('m136');
  }
  await tree.write(directory);

  const latencyMs = 12;
  const prepSamples = [];
  let before;
  let peak = 0;
  let readyAt;
  const sampleHeap = () => { peak = Math.max(peak, process.memoryUsage().heapUsed); };
  const stream = mockStream((context) => {
    prepSamples.push(performance.now() - readyAt);
    sampleHeap();
    const { content } = context.messages.at(-1);
    const prompt = typeof content === 'string' ? content : content.map(({ text }) => text).join('');
    assert(prompt.includes('Previous benchmark checkpoint'), 'Old checkpoint absent from summary input');
    return [textBlock('Benchmark summary')];
  }, latencyMs, () => { readyAt = performance.now(); sampleHeap(); });
  const core = runtime(dataDir, stream, { compaction: { enabled: true, keepRecentTokens: 8192 } });
  try {
    const session = await core.openSession(id);
    session.subscribe((event) => {
      if (event.type === 'compaction_end') sampleHeap();
    });
    before = collectedMemory().heapUsed;
    peak = before;
    const timer = setInterval(sampleHeap, 1);
    try {
      const start = performance.now();
      readyAt = start;
      const result = await session.compact();
      const elapsedMs = ms(performance.now() - start);
      sampleHeap();
      assert.equal(result.status, 'compacted', JSON.stringify(result));
      assert.equal(result.summary, 'Benchmark summary');
      assert(prepSamples.length > 0, 'No summary requests');
      return {
        historyMessages: 400, mockSummaryDelayMs: latencyMs, requests: prepSamples.length,
        preRequestMs: ms(prepSamples[0]), preRequestTotalMs: ms(prepSamples.reduce((a, b) => a + b, 0)),
        elapsedMs, sampledPeakHeapDeltaMiB: mib(peak - before),
      };
    } finally {
      clearInterval(timer);
    }
  } finally {
    await core.dispose();
  }
}

function inFlight() {
  let active = 0;
  let peak = 0;
  let completed = 0;
  let startedAt;
  let endedAt;
  return {
    start() {
      startedAt ??= performance.now();
      peak = Math.max(peak, ++active);
    },
    end() {
      active--;
      completed++;
      endedAt = performance.now();
    },
    result(expected) {
      assert.equal(active, 0);
      assert.equal(completed, expected);
      return { completed, peak, batchElapsedMs: ms(endedAt - startedAt) };
    },
  };
}

async function mcpBench(dataDir, signal) {
  const servers = ['slow', 'fast'].map((id) => ({ id, transport: 'stdio', command: 'mock' }));
  const stop = () => [textBlock('Benchmark done')];

  async function runCold(label, slowDelayMs, slowTimeoutMs) {
    const connected = inFlight();
    const fastServer = servers[1];
    const configured = [{ ...servers[0], connectTimeoutMs: slowTimeoutMs }, fastServer];
    let slowAttempt;
    let slowClosed = 0;
    let resolveSlowClosed;
    const slowClosedPromise = new Promise((resolve) => { resolveSlowClosed = resolve; });
    let opening;
    let fastDiscoveredAtMs;
    const core = runtime(join(dataDir, label), mockStream(stop), {
      mcpServers: configured,
      mcpClientFactory({ id }) {
        connected.start();
        const attempt = (async () => {
          try {
            await delay(id === 'slow' ? slowDelayMs : 10, undefined, { signal });
            return {
              listTools: async () => {
                if (id === 'fast') fastDiscoveredAtMs = ms(performance.now() - opening);
                return [{ name: 'measure', inputSchema: { type: 'object' } }];
              },
              callTool: async () => toolResult('MCP done'),
              close: async () => {
                if (id === 'slow') {
                  slowClosed++;
                  resolveSlowClosed();
                }
              },
            };
          } finally {
            connected.end();
          }
        })();
        if (id === 'slow') slowAttempt = attempt;
        return attempt;
      },
    });
    try {
      opening = performance.now();
      await core.createSession();
      const coldInitMs = ms(performance.now() - opening);
      const catalog = core.getCatalog().mcpServers;
      assert.equal(catalog.find(({ id }) => id === 'fast').state, 'ready');
      const slow = catalog.find(({ id }) => id === 'slow');
      assert.equal(slow.state, slowTimeoutMs === undefined ? 'ready' : 'error');
      if (slowTimeoutMs !== undefined) {
        assert.match(slow.error, new RegExp(`timed out after ${slowTimeoutMs}ms`));
        await slowAttempt;
        await Promise.race([slowClosedPromise, delay(200)]);
        assert.equal(slowClosed, 1);
      }
      return {
        coldInitMs, fastDiscoveredAtMs, slowState: slow.state,
        ...(slowTimeoutMs === undefined ? {} : { lateClientClosed: slowClosed === 1 }),
        mockConnects: connected.result(servers.length),
      };
    } finally {
      await Promise.allSettled([slowAttempt]);
      await core.dispose();
    }
  }

  async function runBatches(label, concurrency) {
    const sessionCount = 3;
    const callsPerSession = 4;
    const expected = sessionCount * callsPerSession;
    const workMs = 35;
    const toolTracker = inFlight();
    const childTracker = inFlight();
    const perParent = Array.from({ length: sessionCount }, inFlight);
    const stream = mockStream((context, { sessionId }) => {
      if (context.messages.at(-1)?.role === 'user') {
        if (sessionId.startsWith('bench-tool-')) {
          return Array.from({ length: callsPerSession }, (_, i) => toolCall('fast__measure', `mcp-${i}`));
        }
        if (sessionId.startsWith('bench-parent-')) {
          return Array.from({ length: callsPerSession }, (_, i) => childCall(`child-${i}`));
        }
      }
      return stop();
    });
    const childStream = mockStream(stop, workMs);
    const core = runtime(join(dataDir, label), (model, context, options) =>
      (options.sessionId.startsWith('bench-') ? stream : childStream)(model, context, options), {
      concurrency,
      mcpServers: [servers[1]],
      mcpClientFactory: () => ({
        listTools: async () => [{ name: 'measure', inputSchema: { type: 'object' } }],
        async callTool() {
          toolTracker.start();
          try {
            await delay(workMs, undefined, { signal });
            return toolResult('MCP done');
          } finally {
            toolTracker.end();
          }
        },
        close: async () => {},
      }),
    });
    async function prompts(parents, input) {
      const start = performance.now();
      const results = await Promise.all(parents.map((parent) => parent.prompt(input)));
      assert(results.every(({ state }) => state === 'completed'));
      return ms(performance.now() - start);
    }
    try {
      const [tools, parents] = await Promise.all(['tool', 'parent'].map((type) => Promise.all(
        Array.from({ length: sessionCount }, (_, i) => core.createSession({ id: `bench-${type}-${i}` })),
      )));
      let failures = 0;
      for (const session of tools) {
        session.subscribe((event) => { if (event.type === 'tool_end' && event.isError) failures++; });
      }
      parents.forEach((parent, i) => parent.subscribe((event) => {
        if (event.type === 'subagent_started') {
          childTracker.start();
          perParent[i].start();
        }
        if (event.type === 'subagent_completed') {
          childTracker.end();
          perParent[i].end();
          if (event.state !== 'completed') failures++;
        }
      }));
      const toolPromptMs = await prompts(tools, 'Run benchmark tools');
      const subagentPromptMs = await prompts(parents, 'Run benchmark children');
      assert.equal(failures, 0);
      const toolStats = toolTracker.result(expected);
      const childStats = childTracker.result(expected);
      const perParentPeak = perParent.map((parent) => parent.result(callsPerSession).peak);
      assert.equal(toolStats.peak, concurrency?.maxToolExecutions ?? expected);
      assert.equal(childStats.peak, concurrency?.maxSubagentsPerRuntime ?? expected);
      assert.equal(Math.max(...perParentPeak), concurrency?.maxSubagentsPerParent ?? callsPerSession);
      assert(perParentPeak.every((peak) => peak <= (concurrency?.maxSubagentsPerParent ?? callsPerSession)));
      return {
        sessionCount, callsPerSession, simulatedWorkMs: workMs,
        tools: { ...toolStats, promptMs: toolPromptMs },
        subagents: { ...childStats, perParentPeak, promptMs: subagentPromptMs },
      };
    } finally {
      await core.dispose();
    }
  }

  const bothFast10Ms = await runCold('cold-fast', 10);
  const baselineA = {
    mcp: await runCold('cold-baseline', 400),
    concurrency: await runBatches('concurrency-baseline'),
  };
  const p2 = {
    mcp: await runCold('cold-isolated', 400, 80),
    concurrency: await runBatches('concurrency-p2', {
      maxSubagentsPerRuntime: 4, maxSubagentsPerParent: 2, maxToolExecutions: 8,
    }),
  };
  return { bothFast10Ms, baselineA, p2 };
}

const scenarios = [
  ['p0', 'P0-memory-subagent', subagentMemoryBench],
  ['p0', 'P0-list', listBench],
  ['p1', 'P1-persist-el', persistBench],
  ['p1', 'P1-compaction', compactionBench],
  ['p2', 'P2-mcp-concurrency', mcpBench],
];

async function main(args) {
  const usage = 'Usage: node --expose-gc scripts/perf-bench.mjs [--suite=all|p0|p1|p2]';
  if (args.length === 1 && args[0] === '--help') {
    console.log(usage);
    return;
  }
  const suite = args.length === 0 ? 'all' :
    args.length === 1 ? /^--suite=(all|p0|p1|p2)$/.exec(args[0])?.[1] : undefined;
  if (!suite) throw new Error(usage);
  const controller = new AbortController();
  const interrupt = (signal) => controller.abort(new Error(`Interrupted by ${signal}`));
  process.on('SIGINT', interrupt);
  process.on('SIGTERM', interrupt);
  const started = performance.now();
  let root;
  try {
    const tempParent = await realpath(tmpdir());
    root = await mkdtemp(join(tempParent, 'agent-core-perf-'));
    console.log(JSON.stringify({
      suite, seed: SEED, node: process.version, platform: process.platform, arch: process.arch,
      gcExposed: typeof globalThis.gc === 'function', mockModelDelayMs: MODEL_MS, root,
    }));
    if (!globalThis.gc) console.warn('Run with --expose-gc to make retained-memory measurements comparable.');
    for (const [group, name, run] of scenarios) {
      if (suite !== 'all' && group !== suite) continue;
      controller.signal.throwIfAborted();
      const result = await run(join(root, name), controller.signal);
      console.log(JSON.stringify({ scenario: name, ...result }));
    }
    controller.signal.throwIfAborted();
  } finally {
    try {
      if (root) {
        await rm(root, { recursive: true, force: true, maxRetries: 3 });
        await assert.rejects(lstat(root), { code: 'ENOENT' });
        console.log(JSON.stringify({ tempDirRemoved: true, root, elapsedMs: ms(performance.now() - started) }));
      }
    } finally {
      process.off('SIGINT', interrupt);
      process.off('SIGTERM', interrupt);
    }
  }
}

try {
  await main(process.argv.slice(2));
} catch (error) {
  console.error(error);
  process.exitCode = 1;
}
