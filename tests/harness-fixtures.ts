import { mkdtemp, realpath } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import type { StreamFn } from '@earendil-works/pi-agent-core';
import { createAssistantMessageEventStream, type Api, type AssistantMessage, type JsonObject, type Model } from '@earendil-works/pi-ai';
import { createAgentRuntime, type AgentRuntimeConfig, type CoreToolDefinition, type SessionStorageAdapter } from '../src/index.js';

export async function testDirectory() {
  return mkdtemp(join(await realpath(tmpdir()), 'agent-core-test-'));
}

export function legacyStorage(store: SessionStorageAdapter): SessionStorageAdapter {
  return {
    create: store.create.bind(store), open: store.open.bind(store), list: store.list.bind(store),
    delete: store.delete.bind(store), append: store.append.bind(store), branch: store.branch.bind(store),
    getBranch: store.getBranch.bind(store), projectActiveMessages: store.projectActiveMessages.bind(store),
  };
}

export function runtime(dataDir: string, streamFnOverride: StreamFn, config: Partial<AgentRuntimeConfig> = {}) {
  return createAgentRuntime({
    dataDir,
    models: {
      defaultAlias: 'test',
      profiles: [{
        alias: 'test', providerId: 'test', modelId: 'test', protocol: 'openai-completions',
        baseUrl: 'http://127.0.0.1:9999/v1', contextWindow: 4096, maxTokens: 256,
      }],
      streamFnOverride,
    },
    ...config,
  });
}

export function reply(model: Model<Api>, ...content: AssistantMessage['content']) {
  const stream = createAssistantMessageEventStream();
  const reason = content.some(({ type }) => type === 'toolCall') ? 'toolUse' : 'stop';
  stream.push({
    type: 'done', reason,
    message: {
      role: 'assistant', model: model.id, provider: model.provider, api: model.api,
      content, stopReason: reason, timestamp: Date.now(),
      usage: {
        input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0,
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
      },
    },
  });
  return stream;
}

export function call(name: string, args: JsonObject = {}, id = `call-${name}`) {
  return { type: 'toolCall' as const, id, name, arguments: args };
}

export function textResult(text: string) {
  return { content: [{ type: 'text' as const, text }] };
}

export function tool(name: string, options: Partial<CoreToolDefinition> = {}): CoreToolDefinition {
  return {
    name, label: name, description: name, parameters: { type: 'object' },
    execute: async () => textResult(name),
    ...options,
  };
}
