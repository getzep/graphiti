import type { GraphitiSearchResults } from '../client.ts';

export type IntentScope = 'public' | 'private';

export interface EntityBoost {
  summaryPattern: string;
  factPattern?: string;
  weight: number;
  suppresses?: string[];
}

export interface IntentRule {
  id: string;
  consumerProfile: string;
  workflowId?: string;
  stepId?: string;
  task?: string;
  injectionText?: string;
  keywords: string[];
  keywordWeight?: number;
  entityBoosts?: EntityBoost[];
  minConfidence?: number;
  scope?: IntentScope;
  packType?: string;
}

export interface IntentRuleSet {
  schema_version: number;
  rules: IntentRule[];
}

export interface IntentDecision {
  matched: boolean;
  intentId?: string;
  score: number;
  explanation: string[];
  rule?: IntentRule;
  packType?: string;
  consumerProfile?: string;
}

export interface IntentDetectionInput {
  prompt: string;
  graphitiResults?: GraphitiSearchResults | null;
  previousIntentId?: string | null;
  enableSticky?: boolean;
  stickyMaxWords?: number;
  stickySignals?: string[];
  defaultMinConfidence?: number;
  logger?: (message: string) => void;
}

export interface IntentDetectionOutput {
  decision: IntentDecision;
  candidates: IntentDecision[];
}
