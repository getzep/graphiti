import { createCaptureHook } from './hooks/capture.ts';
import { createPackInjector } from './hooks/pack-injector.ts';
import { createRecallHook } from './hooks/recall.ts';
import {
  loadCompositionRules,
  loadIntentRules,
  loadPackRegistry,
  normalizeConfig,
} from './config.ts';
import { GraphitiClient } from './client.ts';
import type { CompositionRuleSet } from './composition/types.ts';
import type { IntentRuleSet } from './intent/types.ts';
import type { PackRegistry, PluginConfig } from './config.ts';

export interface GraphitiPluginOptions {
  config?: Partial<PluginConfig>;
  intentRules?: IntentRuleSet | null;
  compositionRules?: CompositionRuleSet | null;
  packRegistry?: PackRegistry | null;
}

export interface OpenClawPlugin {
  name: string;
  hooks: {
    before_agent_start: ReturnType<typeof createRecallHook>;
    agent_end: ReturnType<typeof createCaptureHook>;
  };
}

const loadConfigFromEnv = (): Partial<PluginConfig> => {
  const raw = process.env.GRAPHITI_PLUGIN_CONFIG;
  if (!raw) {
    return {};
  }
  try {
    return JSON.parse(raw) as Partial<PluginConfig>;
  } catch {
    return {};
  }
};

export const createGraphitiPlugin = (options?: GraphitiPluginOptions): OpenClawPlugin => {
  const config = normalizeConfig({
    ...loadConfigFromEnv(),
    ...(options?.config ?? {}),
  });
  const logger = config.debug ? (message: string) => console.log(message) : () => undefined;
  const configRoots = config.configPathRoots;

  const safeLoad = <T>(label: string, loader: () => T | null): T | null => {
    try {
      return loader();
    } catch (error) {
      const message = `Config load failed for ${label}: ${(error as Error).message}`;
      console.warn(`[graphiti-openclaw] ${message}`);
      logger(message);
      return null;
    }
  };

  const intentRules =
    options?.intentRules ??
    safeLoad('intent rules', () => loadIntentRules(config.intentRulesPath, configRoots)) ?? {
      schema_version: 1,
      rules: [],
    };
  const compositionRules =
    options?.compositionRules ??
    safeLoad('composition rules', () =>
      loadCompositionRules(config.compositionRulesPath, configRoots),
    );
  const packRegistry =
    options?.packRegistry ??
    safeLoad('pack registry', () => loadPackRegistry(config.packRegistryPath, configRoots));

  const client = new GraphitiClient({
    baseUrl: config.graphitiBaseUrl,
    apiKey: config.graphitiApiKey,
    recallTimeoutMs: config.recallTimeoutMs,
    captureTimeoutMs: config.captureTimeoutMs,
    maxFacts: config.maxFacts,
  });

  const packInjector = createPackInjector({
    intentRules,
    compositionRules,
    packRegistry,
    config,
  });

  return {
    name: 'graphiti-openclaw',
    hooks: {
      before_agent_start: createRecallHook({
        client,
        packInjector,
        config,
      }),
      agent_end: createCaptureHook({
        client,
        config,
      }),
    },
  };
};

// OpenClaw plugin host runs in a single-threaded event loop per process;
// a per-process lazy singleton avoids import-time side effects while keeping
// initialization predictable.
let defaultPlugin: OpenClawPlugin | null = null;
const getDefaultPlugin = (): OpenClawPlugin => {
  if (!defaultPlugin) {
    defaultPlugin = createGraphitiPlugin();
  }
  return defaultPlugin;
};

const plugin: OpenClawPlugin = {
  get name() {
    return getDefaultPlugin().name;
  },
  get hooks() {
    return getDefaultPlugin().hooks;
  },
};

export default plugin;
