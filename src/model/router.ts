import type { StreamFn, ThinkingLevel } from '@earendil-works/pi-agent-core';
import { createAssistantMessageEventStream, type Api, type Model } from '@earendil-works/pi-ai';
import { anthropicMessagesApi } from '@earendil-works/pi-ai/api/anthropic-messages.lazy';
import { googleGenerativeAIApi } from '@earendil-works/pi-ai/api/google-generative-ai.lazy';
import { openAICompletionsApi } from '@earendil-works/pi-ai/api/openai-completions.lazy';
import { openAIResponsesApi } from '@earendil-works/pi-ai/api/openai-responses.lazy';
import { createModels, createProvider, type Models } from '@earendil-works/pi-ai/models';
import { raceWithAbortSignal } from '@earendil-works/pi-ai/utils/abort';
import type { ModelProfile, ModelRouterConfig } from '../types/config.js';

const adapters = {
  'openai-completions': openAICompletionsApi,
  'openai-responses': openAIResponsesApi,
  'anthropic-messages': anthropicMessagesApi,
  'google-generative-ai': googleGenerativeAIApi,
};

export interface ResolvedModel {
  profile: ModelProfile;
  model: Model<Api>;
  thinkingLevel: ThinkingLevel;
}

interface Route extends ResolvedModel {
  models: Models;
  override?: StreamFn;
}

function sameEndpoint(a: Model<Api>, b: Model<Api>): boolean {
  return a.provider === b.provider && a.api === b.api && a.id === b.id && a.baseUrl === b.baseUrl;
}

function snapshotModel(model: Model<Api>): Model<Api> {
  return { ...model, headers: model.headers && { ...model.headers } };
}

function isLocalGateway(baseUrl: string): boolean {
  const hostname = URL.parse(baseUrl)?.hostname;
  if (!hostname) return false;
  return hostname === '[::1]' || hostname.endsWith('.local') ||
    (!hostname.includes('.') && !hostname.startsWith('[')) ||
    /^(?:127\.|10\.|192\.168\.|172\.(?:1[6-9]|2\d|3[01])\.)[\d.]+$/.test(hostname) ||
    /^\[(?:fc|fd|fe80:)/.test(hostname);
}

export class ModelRouter {
  private defaultAlias = '';
  private routes = new Map<string, Route>();
  // A run can keep using its resolved model after a config reload.
  private readonly routesByModel = new WeakMap<Model<Api>, Route>();

  constructor(config: ModelRouterConfig) {
    this.updateConfig(config);
  }

  resolve(alias = this.defaultAlias): ResolvedModel {
    const route = this.routes.get(alias) ?? (alias === 'auto' ? this.routes.get(this.defaultAlias) : undefined);
    if (!route) throw new Error(`Unknown model alias "${alias}" (default: "${this.defaultAlias}")`);
    const { profile, model, thinkingLevel } = route;
    return { profile, model, thinkingLevel };
  }

  listProfiles(): ModelProfile[] {
    return [...this.routes.values()].map(({ profile }) => profile);
  }

  prepareModel(original: Model<Api>, updated: Model<Api>): Model<Api> {
    // A hook may reuse its object across sessions. Bind a request-local copy to this route.
    const model = snapshotModel(updated);
    const route = this.routesByModel.get(updated) ??
      (sameEndpoint(original, updated) ? this.routesByModel.get(original) : undefined);
    if (route) this.routesByModel.set(model, route);
    return model;
  }

  updateConfig(config: ModelRouterConfig): void {
    const routes = new Map<string, Route>();
    for (const profile of config.profiles) {
      const { alias, modelId, providerId, protocol, baseUrl } = profile;
      const model: Model<Api> = {
        id: modelId, name: modelId, provider: providerId, api: protocol, baseUrl,
        input: ['text', 'image'],
        headers: profile.headers,
        reasoning: profile.reasoning ?? false,
        contextWindow: profile.contextWindow,
        maxTokens: profile.maxTokens,
        compat: profile.compat,
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      };
      // Pi indexes providers by ID. A collection per route allows aliases to
      // share IDs while keeping their endpoints, adapters and credentials independent.
      const models = createModels();
      models.setProvider(createProvider({
        id: providerId,
        baseUrl,
        models: [model],
        api: adapters[protocol](),
        auth: { apiKey: { name: 'Model profile', resolve: async () => ({ auth: {} }) } },
      }));
      const route = {
        profile, model, models, thinkingLevel: profile.defaultThinkingLevel ?? 'off',
        override: config.streamFnOverride,
      } satisfies Route;
      routes.set(alias, route);
      this.routesByModel.set(model, route);
    }
    this.defaultAlias = config.defaultAlias;
    this.routes = routes;
  }

  readonly streamFn: StreamFn = async (model, context, options) => {
    const signal = options?.signal;
    try {
      signal?.throwIfAborted();
      const route = this.routesByModel.get(model) ?? [...this.routes.values()].find(({ model: candidate }) =>
        sameEndpoint(candidate, model));
      if (!route) throw new Error(`No route for model "${model.provider}/${model.id}"`);
      const requestModel = snapshotModel(model);
      const headerEntries = new Map(
        [...Object.entries(requestModel.headers ?? {}), ...Object.entries(options?.headers ?? {})]
          .map(([name, value]) => [name.toLowerCase(), { name, value }]),
      );
      let apiKey = options?.apiKey;
      if (!apiKey) {
        const key = route.profile.apiKey;
        if (typeof key === 'function') {
          const pending = Promise.resolve(key());
          apiKey = await (signal ? raceWithAbortSignal(pending, signal) : pending);
        } else {
          apiKey = key;
        }
      }
      const hasAuth = headerEntries.get('authorization')?.value?.trim() ||
        headerEntries.get('cf-aig-authorization')?.value?.trim();
      if (!apiKey && !hasAuth && (requestModel.api === 'openai-completions' || requestModel.api === 'openai-responses') &&
        isLocalGateway(requestModel.baseUrl)) {
        apiKey = 'local-no-key';
      }
      const headers = Object.fromEntries([...headerEntries.values()].map(({ name, value }) => [name, value]));
      const requestOptions = { ...options, apiKey, headers };
      if (!route.override) return route.models.streamSimple(requestModel, context, requestOptions);
      const pending = Promise.resolve(route.override(requestModel, context, requestOptions));
      return await (signal ? raceWithAbortSignal(pending, signal) : pending);
    } catch (error) {
      const stream = createAssistantMessageEventStream();
      const reason = signal?.aborted ? 'aborted' : 'error';
      const message = error instanceof Error ? error.message : String(error);
      const status = error && typeof error === 'object' ? error as { status?: unknown; statusCode?: unknown } : undefined;
      const errorMessage = status?.status === 429 || status?.statusCode === 429 ? `HTTP 429: ${message}` : message;
      stream.push({
        type: 'error', reason,
        error: {
          role: 'assistant', api: model.api, provider: model.provider, model: model.id, content: [],
          usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0,
            cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
          stopReason: reason, errorMessage, timestamp: Date.now(),
        },
      });
      return stream;
    }
  };
}
