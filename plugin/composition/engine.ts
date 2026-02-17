import type { CompositionRuleSet, ResolvedComposition } from './types.ts';

export const resolveComposition = (
  ruleset: CompositionRuleSet | null | undefined,
  primaryIntent: string,
): ResolvedComposition[] => {
  if (!ruleset) {
    return [];
  }

  const rules = ruleset.rules.filter((rule) => rule.primary_intent === primaryIntent);
  if (rules.length === 0) {
    return [];
  }

  const resolved: ResolvedComposition[] = [];
  for (const rule of rules) {
    for (const injection of rule.inject_additional ?? []) {
      resolved.push({
        packType: injection.pack_type,
        mode: injection.mode,
        required: injection.required ?? false,
      });
    }
  }

  return resolved;
};
