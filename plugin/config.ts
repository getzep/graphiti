import fs from 'node:fs';
import path from 'node:path';

import type { CompositionRuleSet } from './composition/types.ts';
import type { IntentRuleSet } from './intent/types.ts';
import { isPathWithinRoot, toCanonicalPath } from './path-utils.ts';

export interface PackRegistryEntry {
  pack_id: string;
  pack_type?: string;
  path: string;
  scope: 'public' | 'group-safe' | 'private';
}

export interface PackRegistry {
  schema_version: number;
  packs: PackRegistryEntry[];
}

export interface PluginConfig {
  graphitiBaseUrl: string;
  graphitiApiKey?: string;
  /** Optional explicit provider override (used by before_model_resolve). */
  providerOverride?: string;
  /** Optional explicit model override (used by before_model_resolve). */
  modelOverride?: string;
  /** Explicit opt-in for model/provider overrides (secure-by-default). */
  allowModelRoutingOverride: boolean;
  /** Allowed provider override values. Required when providerOverride is set. */
  allowedProviderOverrides: string[];
  /** Allowed model override values. Required when modelOverride is set. */
  allowedModelOverrides: string[];
  /** Max allowed length for model/provider override tokens. */
  maxOverrideTokenLength: number;
  recallTimeoutMs: number;
  captureTimeoutMs: number;
  maxFacts: number;
  /**
   * Canonical Graphiti group lane used when sessionKey is unavailable.
   *
   * SAFETY: this override pins every request to a single group lane.
   * It is ONLY safe in single-tenant deployments (i.e. exactly one user /
   * logical namespace behind the plugin instance).  Multi-tenant operators
   * MUST leave this unset and rely on per-session lanes (`provider.groupId`
   * or `sessionKey`) to prevent cross-tenant memory leakage.
   *
   * This field has no effect unless `singleTenant: true` is also set.
   */
  memoryGroupId?: string;
  /**
   * Explicit multi-lane override for singleTenant deployments.
   *
   * When non-empty and `singleTenant: true`, recall searches ALL listed group
   * IDs simultaneously (fan-out across sessions, OM, self-audit, etc.).
   * This takes precedence over `memoryGroupId` when both are set.
   *
   * Lane ordering is stable (insertion-order preserved, duplicates removed).
   * Empty-string entries are silently stripped during normalization.
   *
   * SAFETY: same tenant-isolation constraint as `memoryGroupId` — only
   * effective when `singleTenant: true` is explicitly declared.
   *
   * Example deployment target:
   *   ["s1_sessions_main", "s1_observational_memory", "learning_self_audit"]
   */
  memoryGroupIds?: string[];
  /**
   * Declare that this plugin instance serves a single tenant.
   *
   * Required to unlock `memoryGroupId` overrides.  When `false` (the safe
   * default), `memoryGroupId` is ignored and per-session lanes are used
   * instead, preventing accidental cross-tenant memory fan-out.
   */
  singleTenant: boolean;
  minPromptChars: number;
  enableSticky: boolean;
  stickyMaxWords: number;
  stickySignals: string[];
  intentRulesPath?: string;
  compositionRulesPath?: string;
  packRegistryPath?: string;
  packRouterCommand?: string | string[];
  packRouterRepoRoot?: string;
  packRouterTimeoutMs: number;
  defaultMinConfidence: number;
  debug: boolean;
  configPathRoots?: string[];
  trustedGroupIds?: string[];
  /** Enable per-turn capability subset injection. Default false. */
  enableCapabilityInjection: boolean;
  /** Command to run the capability subset selector (e.g. "python3 scripts/select_capability_subset.py"). */
  capabilitySelectorCommand?: string | string[];
  /** Path to capability-index.json (passed as --index). */
  capabilityIndexPath?: string;
  /** Path to capability_intent_overrides.json (passed as --overrides). */
  capabilityOverridesPath?: string;
  /** Timeout for the capability selector process. Default 2000ms. */
  capabilitySelectorTimeoutMs: number;
  /** Number of Top-N capabilities to inject per turn. Default 8. */
  capabilityTopN: number;
  /** Only inject capabilities when a pack intent is detected. Default false. */
  capabilityRequireIntent: boolean;
  /** Minimum capability score for inclusion. Default 0 (no floor). */
  capabilityMinScore: number;
  /** Enable context-map anchor injection scaffold. Default false. */
  enableContextMapAnchor: boolean;
  /** Path to the primary context-map file (optional). */
  contextMapPath?: string;
  /** Path to companion context-map metadata file (optional). */
  contextMapMetaPath?: string;
  /** Optional custom anchor text injected when scaffold gates pass. */
  contextMapAnchorText?: string;
}

export const DEFAULT_CONFIG: PluginConfig = {
  graphitiBaseUrl: 'http://localhost:8000',
  recallTimeoutMs: 1500,
  captureTimeoutMs: 1500,
  maxFacts: 8,
  minPromptChars: 6,
  enableSticky: true,
  stickyMaxWords: 20,
  stickySignals: ['also', 'continue', 'what about', 'follow up'],
  packRouterTimeoutMs: 2000,
  defaultMinConfidence: 0.3,
  allowModelRoutingOverride: false,
  allowedProviderOverrides: [],
  allowedModelOverrides: [],
  maxOverrideTokenLength: 128,
  // Safe default: multi-tenant mode. memoryGroupId overrides are disabled
  // unless the operator explicitly opts in via singleTenant: true.
  singleTenant: false,
  debug: false,
  enableCapabilityInjection: false,
  capabilitySelectorTimeoutMs: 2000,
  capabilityTopN: 8,
  capabilityRequireIntent: false,
  capabilityMinScore: 0,
  enableContextMapAnchor: false,
};

const normalizeOptionalString = (value?: string): string | undefined => {
  if (!value) {
    return undefined;
  }
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : undefined;
};

/**
 * Normalize a multi-lane group-ID list:
 *  - Trim whitespace from each entry.
 *  - Strip empty strings (after trim).
 *  - Deduplicate while preserving insertion order (first occurrence wins).
 *  - Returns undefined when the result would be empty (so callers can fall
 *    through to the single-lane memoryGroupId or session-key paths).
 */
const normalizeGroupIds = (ids?: string[]): string[] | undefined => {
  if (!ids || ids.length === 0) return undefined;
  const seen = new Set<string>();
  const out: string[] = [];
  for (const raw of ids) {
    const trimmed = raw.trim();
    if (trimmed.length > 0 && !seen.has(trimmed)) {
      seen.add(trimmed);
      out.push(trimmed);
    }
  }
  return out.length > 0 ? out : undefined;
};

export const normalizeConfig = (config?: Partial<PluginConfig>): PluginConfig => {
  return {
    ...DEFAULT_CONFIG,
    ...config,
    memoryGroupId: normalizeOptionalString(config?.memoryGroupId),
    memoryGroupIds: normalizeGroupIds(config?.memoryGroupIds),
    stickySignals: config?.stickySignals ?? DEFAULT_CONFIG.stickySignals,
    allowedProviderOverrides:
      config?.allowedProviderOverrides ?? DEFAULT_CONFIG.allowedProviderOverrides,
    allowedModelOverrides: config?.allowedModelOverrides ?? DEFAULT_CONFIG.allowedModelOverrides,
    contextMapPath: normalizeOptionalString(config?.contextMapPath),
    contextMapMetaPath: normalizeOptionalString(config?.contextMapMetaPath),
    contextMapAnchorText: normalizeOptionalString(config?.contextMapAnchorText),
  };
};

const toCanonicalRoot = (candidate: string): string => {
  const absolute = path.resolve(candidate);
  return toCanonicalPath(absolute, `config root ${absolute}`);
};

const resolveSafePath = (filePath: string, allowedRoots?: string[]): string => {
  const absolute = path.resolve(filePath);
  const canonicalPath = toCanonicalPath(absolute, `config path ${absolute}`);
  const roots = (allowedRoots && allowedRoots.length > 0 ? allowedRoots : [process.cwd()]).map(
    toCanonicalRoot,
  );
  const allowed = roots.some((root) => isPathWithinRoot(root, canonicalPath));

  if (!allowed) {
    throw new Error(
      `Config path ${canonicalPath} is outside allowed roots: ${roots.join(', ')}`,
    );
  }
  return canonicalPath;
};

const readConfigFile = <T>(filePath: string, allowedRoots?: string[]): T => {
  const absolute = resolveSafePath(filePath, allowedRoots);
  const raw = fs.readFileSync(absolute, 'utf8');
  try {
    return JSON.parse(raw) as T;
  } catch (error) {
    throw new Error(
      `Config file ${absolute} must be JSON for this scaffold. ${(error as Error).message}`,
    );
  }
};

export const loadIntentRules = (
  filePath?: string,
  allowedRoots?: string[],
): IntentRuleSet | null => {
  if (!filePath) {
    return null;
  }
  return readConfigFile<IntentRuleSet>(filePath, allowedRoots);
};

export const loadCompositionRules = (
  filePath?: string,
  allowedRoots?: string[],
): CompositionRuleSet | null => {
  if (!filePath) {
    return null;
  }
  return readConfigFile<CompositionRuleSet>(filePath, allowedRoots);
};

export const loadPackRegistry = (
  filePath?: string,
  allowedRoots?: string[],
): PackRegistry | null => {
  if (!filePath) {
    return null;
  }
  return readConfigFile<PackRegistry>(filePath, allowedRoots);
};
