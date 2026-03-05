import type { GraphitiSearchResults } from '../client.ts';
import type {
  EntityBoost,
  IntentDecision,
  IntentDetectionInput,
  IntentDetectionOutput,
  IntentRule,
  IntentRuleSet,
} from './types.ts';

const DEFAULT_MIN_CONFIDENCE = 0.3;
const DEFAULT_KEYWORD_WEIGHT = 1;
const DEFAULT_STICKY_MAX_WORDS = 20;
const DEFAULT_STICKY_SIGNALS = ['also', 'and', 'continue', 'what about', 'follow up'];
const SCORE_TIE_EPSILON = 1e-6;

interface BoostMatch {
  matched: boolean;
  reason: string;
}

type SuppressionSet = Set<string>;

const toLower = (value: string): string => value.toLowerCase();

const wordCount = (text: string): number => {
  const trimmed = text.trim();
  if (!trimmed) {
    return 0;
  }
  return trimmed.split(/\s+/).length;
};

const shouldStick = (
  prompt: string,
  signals: string[],
  maxWords: number,
): boolean => {
  const promptLower = toLower(prompt);
  if (wordCount(prompt) <= maxWords) {
    return true;
  }
  return signals.some((signal) => promptLower.includes(signal));
};

const safeRegex = (
  pattern: string,
  logger?: (message: string) => void,
  label?: string,
): RegExp | null => {
  try {
    return new RegExp(pattern, 'i');
  } catch (error) {
    if (logger) {
      const context = label ? ` for ${label}` : '';
      logger(
        `Invalid regex pattern${context}: ${pattern}. ${(error as Error).message}`,
      );
    }
    return null;
  }
};

const matchBoost = (
  boost: EntityBoost,
  graphitiResults?: GraphitiSearchResults | null,
  logger?: (message: string) => void,
  ruleId?: string,
): BoostMatch => {
  if (!graphitiResults) {
    return { matched: false, reason: 'no graphiti results' };
  }

  const summaryRegex = safeRegex(
    boost.summaryPattern,
    logger,
    ruleId ? `${ruleId} summaryPattern` : 'summaryPattern',
  );
  const factRegex = boost.factPattern
    ? safeRegex(
        boost.factPattern,
        logger,
        ruleId ? `${ruleId} factPattern` : 'factPattern',
      )
    : null;

  const entitySummaries = graphitiResults.entities?.map((entity) => entity.summary) ?? [];
  const factTexts = graphitiResults.facts?.map((fact) => fact.fact) ?? [];

  const summaryHit = summaryRegex
    ? entitySummaries.some((summary) => summaryRegex.test(summary)) ||
      factTexts.some((fact) => summaryRegex.test(fact))
    : false;

  const factHit = factRegex ? factTexts.some((fact) => factRegex.test(fact)) : true;

  if (summaryHit && factHit) {
    return { matched: true, reason: 'entity boost matched' };
  }

  return { matched: false, reason: 'no boost match' };
};

const computeKeywordHits = (prompt: string, keywords: string[]): string[] => {
  const promptLower = toLower(prompt);
  const hits = keywords.filter((keyword) => promptLower.includes(toLower(keyword)));
  return Array.from(new Set(hits));
};

const applySuppression = (
  candidates: IntentDecision[],
  suppressions: SuppressionSet,
): IntentDecision[] => {
  if (suppressions.size === 0) {
    return candidates;
  }
  return candidates.filter((candidate) => !suppressions.has(candidate.intentId ?? ''));
};

const buildDecision = (rule: IntentRule, score: number, explanation: string[]): IntentDecision => {
  return {
    matched: true,
    intentId: rule.id,
    score,
    explanation,
    rule,
    packType: rule.packType,
    consumerProfile: rule.consumerProfile,
  };
};

const pickWinner = (candidates: IntentDecision[]): IntentDecision => {
  if (candidates.length === 0) {
    return { matched: false, score: 0, explanation: ['no candidates above threshold'] };
  }

  const sorted = [...candidates].sort((a, b) => b.score - a.score);
  const top = sorted[0];
  const runnerUp = sorted[1];

  if (runnerUp && Math.abs(top.score - runnerUp.score) <= SCORE_TIE_EPSILON) {
    return { matched: false, score: top.score, explanation: ['tie between intents'] };
  }

  return top;
};

export const detectIntent = (
  ruleset: IntentRuleSet,
  input: IntentDetectionInput,
): IntentDetectionOutput => {
  const prompt = input.prompt ?? '';
  const defaultMinConfidence = input.defaultMinConfidence ?? DEFAULT_MIN_CONFIDENCE;
  const enableSticky = input.enableSticky ?? true;
  const stickyMaxWords = input.stickyMaxWords ?? DEFAULT_STICKY_MAX_WORDS;
  const stickySignals = input.stickySignals ?? DEFAULT_STICKY_SIGNALS;

  if (enableSticky && input.previousIntentId) {
    if (shouldStick(prompt, stickySignals, stickyMaxWords)) {
      const stickyRule = ruleset.rules.find((rule) => rule.id === input.previousIntentId);
      if (stickyRule) {
        const decision = buildDecision(stickyRule, defaultMinConfidence, [
          'sticky follow-up intent applied',
        ]);
        return { decision, candidates: [decision] };
      }
    }
  }

  const suppressions: SuppressionSet = new Set();
  const candidates: IntentDecision[] = [];

  for (const rule of ruleset.rules) {
    const keywordHits = computeKeywordHits(prompt, rule.keywords ?? []);
    if (keywordHits.length === 0) {
      continue;
    }

    const keywordWeight = rule.keywordWeight ?? DEFAULT_KEYWORD_WEIGHT;
    let score = keywordHits.length * keywordWeight;
    const explanation = [`keywords matched: ${keywordHits.join(', ')}`];

    for (const boost of rule.entityBoosts ?? []) {
      const match = matchBoost(boost, input.graphitiResults ?? null, input.logger, rule.id);
      if (match.matched) {
        score += boost.weight;
        explanation.push(`entity boost +${boost.weight}`);
        for (const suppressed of boost.suppresses ?? []) {
          suppressions.add(suppressed);
        }
      }
    }

    candidates.push(buildDecision(rule, score, explanation));
  }

  const minFiltered = applySuppression(candidates, suppressions).filter((candidate) => {
    const minConfidence = candidate.rule?.minConfidence ?? defaultMinConfidence;
    return candidate.score >= minConfidence;
  });

  const decision = pickWinner(minFiltered);
  return { decision, candidates: minFiltered };
};
