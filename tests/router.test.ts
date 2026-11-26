import { createServer } from 'node:http';
import { once } from 'node:events';
import type { StreamFn } from '@earendil-works/pi-agent-core';
import { createAssistantMessageEventStream, normalizeContext } from '@earendil-works/pi-ai';
import { describe, expect, it, vi } from 'vitest';
import { ModelRouter, type ModelProfile, type ModelProtocol } from '../src/index.js';

const profile: ModelProfile = {
  alias: 'local', modelId: 'local-model', providerId: 'test-gateway',
  protocol: 'openai-completions', baseUrl: 'http://127.0.0.1:9999/v1',
  contextWindow: 8000, maxTokens: 500, reasoning: true, defaultThinkingLevel: 'low',
};
const emptyContext = normalizeContext({ messages: [] });

describe('ModelRouter', () => {
  it('resolves the default, auto and all four protocol aliases', () => {
    const protocols: ModelProtocol[] = [
      'openai-completions', 'openai-responses', 'anthropic-messages', 'google-generative-ai',
    ];
    const router = new ModelRouter({
      defaultAlias: profile.alias,
      profiles: protocols.map((protocol, index) => ({
        ...profile, protocol, alias: index === 0 ? profile.alias : protocol,
      })),
    });
    expect(router.resolve()).toEqual(router.resolve('auto'));
    expect(router.resolve('auto')).toMatchObject({
      model: { id: 'local-model', provider: 'test-gateway', api: 'openai-completions' },
      thinkingLevel: 'low',
    });
    for (const protocol of protocols.slice(1)) {
      expect(router.resolve(protocol).model.api).toBe(protocol);
    }
    expect(() => router.resolve('missing')).toThrow(/unknown model alias.*missing/i);
  });

  it('honors an explicitly configured auto and hot-reloads new requests without changing a resolved route', async () => {
    const oldStream = vi.fn<StreamFn>(createAssistantMessageEventStream);
    const newStream = vi.fn<StreamFn>(createAssistantMessageEventStream);
    const router = new ModelRouter({
      defaultAlias: 'local', profiles: [profile, { ...profile, alias: 'auto', modelId: 'auto-model' }],
      streamFnOverride: oldStream,
    });
    const oldModel = router.resolve('auto').model;
    const call = router.streamFn;
    router.updateConfig({
      defaultAlias: 'replacement', profiles: [{ ...profile, alias: 'replacement', modelId: 'new-model' }],
      streamFnOverride: newStream,
    });
    expect(oldModel.id).toBe('auto-model');
    expect(router.resolve('auto').model.id).toBe('new-model');
    expect(() => router.resolve('local')).toThrow();
    await call(oldModel, emptyContext);
    await call(router.resolve().model, emptyContext);
    expect(oldStream).toHaveBeenCalledOnce();
    expect(newStream).toHaveBeenCalledOnce();
  });

  it('resolves static and asynchronous keys per request and respects an Authorization header', async () => {
    const stream = vi.fn<StreamFn>(createAssistantMessageEventStream);
    const key = vi.fn().mockResolvedValueOnce('first').mockResolvedValueOnce('second');
    const router = new ModelRouter({
      defaultAlias: 'local', streamFnOverride: stream,
      profiles: [
        profile,
        { ...profile, alias: 'dynamic', apiKey: key },
        { ...profile, alias: 'static', apiKey: 'static-key' },
        { ...profile, alias: 'header', headers: { Authorization: 'Custom token' } },
        { ...profile, alias: 'external', baseUrl: 'https://api.openai.com/v1' },
      ],
    });
    for (const alias of ['local', 'dynamic', 'dynamic', 'static', 'header', 'external']) {
      await router.streamFn(router.resolve(alias).model, emptyContext);
    }
    expect(stream.mock.calls.map(([, , options]) => options?.apiKey)).toEqual([
      'local-no-key', 'first', 'second', 'static-key', undefined, undefined,
    ]);
    expect(stream.mock.calls[4]?.[2]?.headers).toMatchObject({ Authorization: 'Custom token' });
  });

  it('snapshots the headers of a model prepared by a hook', async () => {
    const stream = vi.fn<StreamFn>(createAssistantMessageEventStream);
    const router = new ModelRouter({ defaultAlias: 'local', profiles: [profile], streamFnOverride: stream });
    const original = router.resolve().model;
    const headers = { Authorization: 'fixture-before' };
    const prepared = router.prepareModel(original, { ...original, headers });
    headers.Authorization = 'fixture-after';
    await router.streamFn(prepared, emptyContext);
    expect(stream.mock.calls[0][2]?.headers?.Authorization).toBe('fixture-before');
  });

  it('isolates concurrent request headers while credential resolution is pending', async () => {
    const stream = vi.fn<StreamFn>(createAssistantMessageEventStream);
    const pending = Promise.withResolvers<string>();
    const router = new ModelRouter({ defaultAlias: 'local', streamFnOverride: stream, profiles: [
      { ...profile, headers: { Authorization: 'fixture-first' }, apiKey: () => pending.promise },
    ] });
    const model = router.resolve().model;
    const headers = { 'X-Request': 'first' };
    const first = router.streamFn(model, emptyContext, { headers });
    model.headers!.Authorization = 'fixture-second';
    headers['X-Request'] = 'second';
    const second = router.streamFn(model, emptyContext, { headers });
    pending.resolve('fixture-key');
    await Promise.all([first, second]);
    expect(stream.mock.calls.map(([model, , opts]) =>
      [model.headers?.Authorization, opts?.headers?.Authorization, opts?.headers?.['X-Request']])).toEqual([
      ['fixture-first', 'fixture-first', 'first'], ['fixture-second', 'fixture-second', 'second'],
    ]);
  });

  it.each([
    [undefined, 'Bearer local-no-key'],
    ['Custom local token', 'Custom local token'],
  ])('handles local Authorization %s through the real OpenAI Completions adapter', async (header, expected) => {
    let authorization: string | undefined;
    const server = createServer((request, response) => {
      authorization = request.headers.authorization;
      request.resume();
      response.writeHead(200, { 'content-type': 'text/event-stream' });
      const chunk = { id: 'local-test', model: 'local-model', object: 'chat.completion.chunk' };
      response.end([
        { ...chunk, choices: [{ index: 0, delta: { content: 'hello' }, finish_reason: null }] },
        { ...chunk, choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] },
      ].map((value) => `data: ${JSON.stringify(value)}\n\n`).join('') + 'data: [DONE]\n\n');
    });
    try {
      server.listen(0, '127.0.0.1');
      await once(server, 'listening');
      const address = server.address();
      if (!address || typeof address === 'string') throw new Error('Expected a TCP address');
      const router = new ModelRouter({
        defaultAlias: 'local',
        profiles: [{
          ...profile, baseUrl: `http://127.0.0.1:${address.port}/v1`,
          headers: header ? { authorization: header } : undefined,
        }],
      });
      const stream = await router.streamFn(router.resolve().model, normalizeContext({
        messages: [{ role: 'user', content: 'hi', timestamp: Date.now() }],
      }));
      const result = await stream.result();
      expect(result).toMatchObject({ stopReason: 'stop', content: [{ type: 'text', text: 'hello' }] });
      expect(authorization).toBe(expected);
    } finally {
      await new Promise<void>((resolve, reject) => {
        server.close((error) => error ? reject(error) : resolve());
      });
    }
  });
});
