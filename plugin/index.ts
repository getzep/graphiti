import type { OpenClawPluginApi } from 'openclaw/plugin-sdk';

import { createCaptureHook } from './hooks/capture.ts';
import {
  createLegacyBeforeAgentStartHook,
  hasPromptBuildExecuted,
  markPromptBuildExecuted,
} from './hooks/legacy-before-agent-start.ts';
import { createModelResolveHook } from './hooks/model-resolve.ts';
import { createPackInjector } from './hooks/pack-injector.ts';
import { createRecallHook } from './hooks/recall.ts';
import { createContextMapAnchorHooks } from './hooks/context-map-anchor.ts';
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

const loadConfigFromEnv = (): Partial<PluginConfig> => {
  const raw = process.env.BICAMERAL_PLUGIN_CONFIG ?? process.env.GRAPHITI_PLUGIN_CONFIG;
  if (!raw) {
    return {};
  }
  try {
    return JSON.parse(raw) as Partial<PluginConfig>;
  } catch {
    return {};
  }
};

export const buildGraphitiHooks = (options?: GraphitiPluginOptions) => {
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
      console.warn(`[bicameral] ${message}`);
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

  const promptBuildHook = createRecallHook({
    client,
    packInjector,
    config,
  });
  const contextMapAnchorHooks = createContextMapAnchorHooks({ config });

  const beforePromptBuildHook: ReturnType<typeof createRecallHook> = async (event, ctx) => {
    if (hasPromptBuildExecuted(ctx)) {
      return {};
    }
    markPromptBuildExecuted(ctx);

    const [anchorResult, promptResult] = await Promise.all([
      contextMapAnchorHooks.before_prompt_build(event, ctx),
      promptBuildHook(event, ctx),
    ]);

    const prependContext = [anchorResult.prependContext, promptResult.prependContext]
      .filter((part): part is string => Boolean(part && part.trim().length > 0))
      .join('\n\n');

    return prependContext.length > 0 ? { prependContext } : {};
  };

  return {
    session_start: contextMapAnchorHooks.session_start,
    after_compaction: contextMapAnchorHooks.after_compaction,
    before_reset: contextMapAnchorHooks.before_reset,
    before_model_resolve: createModelResolveHook({ config }),
    before_prompt_build: beforePromptBuildHook,
    before_agent_start: createLegacyBeforeAgentStartHook(promptBuildHook),
    agent_end: createCaptureHook({
      client,
      config,
    }),
  };
};

export interface OpenClawPlugin {
  name: string;
  hooks: ReturnType<typeof buildGraphitiHooks>;
}

/**
 * @deprecated Prefer the default register() plugin export.
 * Kept for backwards compatibility with older consumers that construct hooks directly.
 */
export const createGraphitiPlugin = (options?: GraphitiPluginOptions): OpenClawPlugin => ({
  name: 'bicameral',
  hooks: buildGraphitiHooks(options),
});

const bicameralPlugin = {
  id: 'bicameral',
  name: 'Bicameral',
  description: 'Bicameral runtime context injection plugin',

  register(api: OpenClawPluginApi) {
    const hooks = buildGraphitiHooks({
      config: (api.pluginConfig as Partial<PluginConfig> | undefined) ?? {},
    });

    api.on('session_start', hooks.session_start);
    api.on('after_compaction', hooks.after_compaction);
    api.on('before_reset', hooks.before_reset);
    api.on('before_model_resolve', hooks.before_model_resolve);
    api.on('before_prompt_build', hooks.before_prompt_build);
    api.on('before_agent_start', hooks.before_agent_start);
    api.on('agent_end', hooks.agent_end);
  },
};

export default bicameralPlugin;
