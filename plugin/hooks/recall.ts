import type { GraphitiClient } from '../client.ts';
import type { GraphitiSearchResults } from '../client.ts';
import { normalizeConfig } from '../config.ts';
import type { PluginConfig } from '../config.ts';
import type { PackContextResult } from './pack-injector.ts';
import type { PackInjectorContext } from './pack-injector.ts';

export interface BeforeAgentStartEvent {
  prompt: string;
  messages?: Array<{ role?: string; content: string }>;
}

export type RecallHook = (
  event: BeforeAgentStartEvent,
  ctx: PackInjectorContext,
) => Promise<{ prependContext: string }>;

export interface RecallHookDeps {
  client: GraphitiClient;
  packInjector: (input: {
    prompt: string;
    graphitiResults?: GraphitiSearchResults | null;
    ctx: PackInjectorContext;
  }) => Promise<PackContextResult | null>;
  config?: Partial<PluginConfig>;
}

const formatGraphitiContext = (results: GraphitiSearchResults): string => {
  const lines: string[] = [];
  lines.push('<graphiti-context>');
  lines.push('## Graphiti Recall');
  if (results.facts.length === 0) {
    lines.push('No relevant facts found.');
  } else {
    for (const fact of results.facts) {
      lines.push(`- ${fact.fact}`);
    }
  }
  lines.push('</graphiti-context>');
  return lines.join('\n');
};

const formatFallback = (reason: string): string => {
  return [
    '<graphiti-fallback>',
    `Graphiti recall unavailable (${reason}). Use memory_search or memory_get for QMD fallback.`,
    '</graphiti-fallback>',
  ].join('\n');
};

const resolveGroupIds = (ctx: PackInjectorContext): string[] | undefined => {
  const groupId = ctx.messageProvider?.groupId ?? ctx.sessionKey;
  return groupId ? [groupId] : undefined;
};

export const createRecallHook = (deps: RecallHookDeps): RecallHook => {
  const config = normalizeConfig(deps.config);
  const logger = config.debug ? (message: string) => console.log(message) : () => undefined;

  return async (event, ctx) => {
    const parts: string[] = [];
    const prompt = event.prompt ?? '';
    let graphitiResults: GraphitiSearchResults | null = null;

    if (prompt.trim().length >= config.minPromptChars) {
      try {
        graphitiResults = await deps.client.search(prompt, resolveGroupIds(ctx));
        parts.push(formatGraphitiContext(graphitiResults));
      } catch (error) {
        const message = (error as Error).message || 'unknown error';
        logger(`Graphiti recall failed: ${message}`);
        parts.push(formatFallback(message));
      }
    } else {
      parts.push('');
    }

    try {
      const packResult = await deps.packInjector({
        prompt,
        graphitiResults,
        ctx,
      });
      if (packResult?.context) {
        parts.push(packResult.context);
      }
    } catch (error) {
      logger(`Pack injection failed: ${(error as Error).message}`);
    }

    return { prependContext: parts.filter((part) => part.trim().length > 0).join('\n\n') };
  };
};
