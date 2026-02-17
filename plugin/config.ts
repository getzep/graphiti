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
  recallTimeoutMs: number;
  captureTimeoutMs: number;
  maxFacts: number;
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
}

export const DEFAULT_CONFIG: PluginConfig = {
  graphitiBaseUrl: 'http://localhost:8000',
  recallTimeoutMs: 1500,
  captureTimeoutMs: 1500,
  maxFacts: 8,
  minPromptChars: 6,
  enableSticky: true,
  stickyMaxWords: 20,
  stickySignals: ['also', 'and', 'continue', 'what about', 'follow up'],
  packRouterTimeoutMs: 2000,
  defaultMinConfidence: 0.3,
  debug: false,
};

export const normalizeConfig = (config?: Partial<PluginConfig>): PluginConfig => {
  return {
    ...DEFAULT_CONFIG,
    ...config,
    stickySignals: config?.stickySignals ?? DEFAULT_CONFIG.stickySignals,
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
