export interface CompositionRuleSet {
  schema_version: number;
  rules: CompositionRule[];
}

export interface CompositionRule {
  primary_intent: string;
  inject_additional: CompositionInjection[];
}

export interface CompositionInjection {
  pack_type: string;
  mode?: string;
  required?: boolean;
  condition?: 'always';
}

export interface ResolvedComposition {
  packType: string;
  mode?: string;
  required: boolean;
}
