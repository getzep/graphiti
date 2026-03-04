import { createHash } from 'node:crypto';

import type { GraphitiClient } from '../client.ts';
import type { GraphitiSearchResults } from '../client.ts';
import { normalizeConfig } from '../config.ts';
import type { PluginConfig } from '../config.ts';
import { deriveGroupLane } from '../lane-utils.ts';
import type { PackContextResult } from './pack-injector.ts';
import type { PackInjectorContext } from './pack-injector.ts';
import { createCapabilityInjector } from './capability-injector.ts';

export interface BeforeAgentStartEvent {
  prompt: string;
  messages?: Array<{ role?: string; content: string }>;
}

export type RecallHook = (
  event: BeforeAgentStartEvent,
  ctx: PackInjectorContext,
) => Promise<{ prependContext?: string }>;

export interface RecallHookDeps {
  client: GraphitiClient;
  packInjector: (input: {
    prompt: string;
    graphitiResults?: GraphitiSearchResults | null;
    ctx: PackInjectorContext;
  }) => Promise<PackContextResult | null>;
  config?: Partial<PluginConfig>;
}

/**
 * Escape XML/HTML special characters in recalled fact text so that adversarial
 * or malformed fact content cannot inject tags that break the surrounding
 * <graphiti-context> block or confuse the model parser.
 */
const escapeXml = (text: string): string =>
  text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&apos;');
const LOW_INFO_SINGLE_TOKENS = new Set([
  'ok',
  'yes',
  'no',
  'k',
  'yep',
  'nope',
  'sure',
  'thanks',
  'ty',
  'thx',
  'cool',
  'nice',
  'agreed',
  'done',
  'noted',
]);

const LOW_INFO_SHORT_ACKS = new Set([
  'sounds good',
  'got it',
  'makes sense',
  'lgtm',
  'go ahead',
  'do it',
  'perfect',
  'great',
  'good call',
]);

const NOVELTY_WINDOW_TURNS = 3;
const MAX_NOVELTY_SESSIONS = 512;

interface PreparedFact {
  text: string;
  fingerprint: string;
}

const normalizeWhitespace = (text: string): string => text.trim().replace(/\s+/g, ' ');

const normalizeFactText = (text: string): string => normalizeWhitespace(text).toLowerCase();

const fingerprint = (value: string): string => createHash('sha256').update(value).digest('hex');

const isEmojiOnly = (prompt: string): boolean => {
  const compact = prompt.replace(/\s+/g, '');
  if (!compact) {
    return false;
  }

  if (!/[\p{Extended_Pictographic}]/u.test(compact)) {
    return false;
  }

  const withoutEmoji = compact.replace(
    /[\p{Extended_Pictographic}\u{1F3FB}-\u{1F3FF}\u200D\uFE0F]/gu,
    '',
  );
  return withoutEmoji.length === 0;
};

const stripTrailingAckPunctuation = (value: string): string =>
  value.replace(/[.!?]+$/g, '').trim();

const isLowInformationPrompt = (prompt: string): boolean => {
  const normalizedRaw = normalizeWhitespace(prompt).toLowerCase();
  if (!normalizedRaw) {
    return false;
  }

  if (/^[?.!]+$/.test(normalizedRaw)) {
    return true;
  }

  if (isEmojiOnly(normalizedRaw)) {
    return true;
  }

  const normalized = stripTrailingAckPunctuation(normalizedRaw);
  if (!normalized) {
    return true;
  }

  if (LOW_INFO_SHORT_ACKS.has(normalized)) {
    return true;
  }

  const tokens = normalized.split(' ');
  return tokens.length === 1 && LOW_INFO_SINGLE_TOKENS.has(tokens[0]);
};

const dedupeFactsWithinTurn = (facts: Array<{ fact: string }>): {
  prepared: PreparedFact[];
  removed: number;
} => {
  const seen = new Set<string>();
  const prepared: PreparedFact[] = [];
  let removed = 0;

  for (const fact of facts) {
    const text = normalizeWhitespace(fact.fact ?? '');
    const normalized = normalizeFactText(text);
    if (!normalized) {
      continue;
    }
    if (seen.has(normalized)) {
      removed += 1;
      continue;
    }

    seen.add(normalized);
    prepared.push({
      text,
      fingerprint: fingerprint(normalized),
    });
  }

  return { prepared, removed };
};

const collectUsefulEntitySummaries = (
  entities: GraphitiSearchResults['entities'],
): string[] => {
  const seen = new Set<string>();
  const summaries: string[] = [];

  for (const entity of entities ?? []) {
    const summary = normalizeWhitespace(entity.summary ?? '');
    const normalized = normalizeFactText(summary);
    if (!normalized || seen.has(normalized)) {
      continue;
    }
    seen.add(normalized);
    summaries.push(summary);
  }

  return summaries;
};

const pushNoveltyTurn = (
  state: Map<string, string[][]>,
  sessionStateKey: string | undefined,
  fingerprintsForTurn: string[],
): void => {
  if (!sessionStateKey) {
    return;
  }

  const current = state.get(sessionStateKey) ?? [];
  current.push(fingerprintsForTurn);
  while (current.length > NOVELTY_WINDOW_TURNS) {
    current.shift();
  }

  // Refresh insertion order so we can evict least-recently-updated keys.
  state.delete(sessionStateKey);
  state.set(sessionStateKey, current);

  while (state.size > MAX_NOVELTY_SESSIONS) {
    const oldestKey = state.keys().next().value;
    if (typeof oldestKey !== 'string') {
      break;
    }
    state.delete(oldestKey);
  }
};

const getSeenFingerprints = (
  state: Map<string, string[][]>,
  sessionStateKey: string | undefined,
): Set<string> => {
  if (!sessionStateKey) {
    return new Set<string>();
  }

  const seen = new Set<string>();
  for (const turn of state.get(sessionStateKey) ?? []) {
    for (const value of turn) {
      seen.add(value);
    }
  }
  return seen;
};

const getSessionStateKey = (ctx: PackInjectorContext): string | undefined =>
  ctx.sessionKey ?? ctx.messageProvider?.groupId;

const formatGraphitiContext = (facts: string[], entitySummaries: string[]): string => {
  const lines: string[] = [];
  lines.push('<graphiti-context>');
  lines.push('## Graphiti Recall');
  for (const fact of facts) {
    lines.push(`- ${escapeXml(fact)}`);
  }
  if (entitySummaries.length > 0) {
    lines.push('## Related Entities');
    for (const summary of entitySummaries) {
      lines.push(`- ${escapeXml(summary)}`);
    }
  }
  lines.push('</graphiti-context>');
  return lines.join('\n');
};

const FALLBACK_ERROR_CODE = 'GRAPHITI_QMD_FAILOVER';

const sanitizeReason = (reason: string): string => {
  const compact = reason.replace(/\s+/g, ' ').trim();
  const chars = Array.from(compact);
  if (chars.length <= 180) {
    return compact;
  }
  return `${chars.slice(0, 180).join('')}...`;
};

/**
 * Sanitize a group/session identifier before emitting it in log output.
 * Raw identifiers can contain platform-specific data (e.g. Telegram chat IDs,
 * session routing keys) that should not leak verbatim into system logs.
 * Truncates to 32 chars and replaces whitespace so the output is safe to
 * embed in a single log line without quoting.
 */
const sanitizeIdentifier = (id: string): string => {
  const compact = id.replace(/\s+/g, '_').trim();
  if (compact.length <= 32) return compact;
  return `${compact.slice(0, 32)}…`;
};

const formatFallback = (): string => {
  // Surface only a generic message to the model / user-visible output.
  // The real error reason is logged internally via console.warn so it is
  // never leaked through the model context.
  return [
    '<graphiti-fallback>',
    `ERROR_CODE: ${FALLBACK_ERROR_CODE}`,
    'WARNING: Graphiti recall failed (Service unavailable). This turn is using QMD fallback.',
    'Use memory_search or memory_get if you want to inspect fallback retrieval directly.',
    '</graphiti-fallback>',
  ].join('\n');
};

const resolveGroupIds = (
  ctx: PackInjectorContext,
  config: PluginConfig,
): string[] | undefined => {
  // SAFETY: multi-lane override — only active when the operator has explicitly
  // declared singleTenant: true. memoryGroupIds takes precedence over the
  // scalar memoryGroupId when both are set, enabling fan-out recall across
  // sessions, observational memory, self-audit, and any other named lanes.
  // In multi-tenant mode (the safe default) this block is skipped entirely so
  // different users cannot read each other's memories.
  if (config.singleTenant && config.memoryGroupIds && config.memoryGroupIds.length > 0) {
    return config.memoryGroupIds;
  }
  // SAFETY: scalar single-lane override — same singleTenant guard applies.
  if (config.singleTenant && config.memoryGroupId) {
    return [config.memoryGroupId];
  }
  if (ctx.messageProvider?.groupId) {
    return [ctx.messageProvider.groupId];
  }
  // SECURITY: never forward the raw sessionKey to Graphiti — it may embed
  // sensitive platform identifiers (e.g. Telegram chat IDs, routing tokens).
  // Derive a deterministic, non-reversible lane id from it instead so that
  // recall and capture are still scoped to the same lane without leaking the
  // original value to an external service.
  if (ctx.sessionKey) {
    return [deriveGroupLane(ctx.sessionKey)];
  }
  return undefined;
};

export const createRecallHook = (deps: RecallHookDeps): RecallHook => {
  const config = normalizeConfig(deps.config);
  const logger = config.debug ? (message: string) => console.log(message) : () => undefined;
  const capabilityInjector = createCapabilityInjector({ config });
  const noveltyState = new Map<string, string[][]>();

  return async (event, ctx) => {
    const parts: string[] = [];
    const sessionStateKey = getSessionStateKey(ctx);
    const prompt = event.prompt ?? '';
    if (isLowInformationPrompt(prompt)) {
      logger('skip_low_info');
      pushNoveltyTurn(noveltyState, sessionStateKey, []);
      return { prependContext: '' };
    }

    let graphitiResults: GraphitiSearchResults | null = null;

    if (prompt.trim().length >= config.minPromptChars) {
      const groupIds = resolveGroupIds(ctx, config);

      // Fail safe: never issue an unscoped Graphiti search.
      // If we cannot resolve a group/session lane, force fallback instead of
      // calling search() with undefined and risking cross-tenant recall.
      if (!groupIds || groupIds.length === 0) {
        const safeGroup = sanitizeIdentifier('missing-group-scope');
        console.warn(
          `[bicameral] ${FALLBACK_ERROR_CODE} group=${safeGroup} reason=missing_group_scope`,
        );
        logger('Graphiti recall skipped: missing group scope');
        parts.push(formatFallback());
        pushNoveltyTurn(noveltyState, sessionStateKey, []);
      } else {
        try {
          graphitiResults = await deps.client.search(prompt, groupIds);
          const { prepared, removed } = dedupeFactsWithinTurn(graphitiResults.facts ?? []);
          if (removed > 0) {
            logger(`deduped_${removed}_facts`);
          }

          const previouslySeen = getSeenFingerprints(noveltyState, sessionStateKey);
          const novelFacts = prepared.filter((fact) => !previouslySeen.has(fact.fingerprint));
          const suppressed = prepared.length - novelFacts.length;
          if (suppressed > 0) {
            logger(`novelty_suppressed_${suppressed}`);
          }

          const usefulEntities = collectUsefulEntitySummaries(graphitiResults.entities);
          if (novelFacts.length === 0 && usefulEntities.length === 0) {
            logger('skip_empty_recall');
            pushNoveltyTurn(noveltyState, sessionStateKey, []);
          } else {
            parts.push(
              formatGraphitiContext(
                novelFacts.map((fact) => fact.text),
                usefulEntities,
              ),
            );
            pushNoveltyTurn(
              noveltyState,
              sessionStateKey,
              novelFacts.map((fact) => fact.fingerprint),
            );
          }
        } catch (error) {
          const message = (error as Error).message || 'unknown error';
          const safeReason = sanitizeReason(message);
          // Sanitize identifier to avoid leaking raw session keys / platform IDs into logs.
          const safeGroup = sanitizeIdentifier(groupIds[0] ?? 'unknown');
          // Always emit failover warnings, even when debug logging is disabled.
          console.warn(
            `[bicameral] ${FALLBACK_ERROR_CODE} group=${safeGroup} reason=${safeReason}`,
          );
          logger(`Graphiti recall failed: ${safeReason}`);
          parts.push(formatFallback());
          pushNoveltyTurn(noveltyState, sessionStateKey, []);
        }
      }
    } else {
      parts.push('');
      pushNoveltyTurn(noveltyState, sessionStateKey, []);
    }

    let packIntentId: string | undefined;
    try {
      const packResult = await deps.packInjector({
        prompt,
        graphitiResults,
        ctx,
      });
      if (packResult?.context) {
        parts.push(packResult.context);
        packIntentId = packResult.intentId;
      }
    } catch (error) {
      logger(`Pack injection failed: ${(error as Error).message}`);
    }

    try {
      const capabilityContext = await capabilityInjector({
        prompt,
        intentId: packIntentId,
      });
      if (capabilityContext) {
        parts.push(capabilityContext);
      }
    } catch (error) {
      logger(`Capability injection failed: ${(error as Error).message}`);
    }

    return { prependContext: parts.filter((part) => part.trim().length > 0).join('\n\n') };
  };
};
