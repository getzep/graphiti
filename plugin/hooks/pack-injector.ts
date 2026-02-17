import fs from 'node:fs';
import path from 'node:path';
import { spawn } from 'node:child_process';

import { resolveComposition } from '../composition/engine.ts';
import type { CompositionRuleSet, ResolvedComposition } from '../composition/types.ts';
import { normalizeConfig } from '../config.ts';
import type { PackRegistry } from '../config.ts';
import type { GraphitiSearchResults } from '../client.ts';
import { detectIntent } from '../intent/detector.ts';
import type { IntentRuleSet, IntentDecision } from '../intent/types.ts';
import type { PluginConfig } from '../config.ts';
import { isPathWithinRoot, toCanonicalPath } from '../path-utils.ts';

export interface PackContextResult {
  context: string;
  intentId: string;
  primaryPackId: string;
  scope: string;
}

export interface PackInjectorContext {
  sessionKey?: string;
  messageProvider?: {
    chatType?: string;
    groupId?: string;
  };
}

export interface PackInjectorInput {
  prompt: string;
  graphitiResults?: GraphitiSearchResults | null;
  ctx: PackInjectorContext;
}

export interface PackMaterialized {
  packId: string;
  content: string;
  scope: string;
  mode?: string;
}

interface PackPlan {
  consumer: string;
  workflow_id: string;
  step_id: string;
  scope: string;
  task: string;
  injection_text: string;
  packs: { pack_id: string; query: string }[];
}

interface PackInjectorDeps {
  intentRules: IntentRuleSet;
  compositionRules?: CompositionRuleSet | null;
  packRegistry?: PackRegistry | null;
  config?: Partial<PluginConfig>;
}

const isGroupChat = (ctx: PackInjectorContext): boolean => {
  return ctx.messageProvider?.chatType === 'group';
};

const resolveRegistryEntry = (registry: PackRegistry, packType: string) => {
  return registry.packs.find(
    (pack) => pack.pack_type === packType || pack.pack_id === packType,
  );
};

const isRecord = (value: unknown): value is Record<string, unknown> => {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
};

const isNonEmptyString = (value: unknown): value is string => {
  return typeof value === 'string' && value.trim().length > 0;
};

const escapeXmlAttr = (value: string): string => {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&apos;');
};

const escapeXmlText = (value: string): string => {
  return value.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
};

const resolvePackPath = (repoRoot: string, packPath: string): string => {
  const resolvedRoot = path.resolve(repoRoot);
  const resolvedPath = path.resolve(resolvedRoot, packPath);
  const canonicalRoot = toCanonicalPath(resolvedRoot, 'pack repo root');
  const canonicalPath = toCanonicalPath(resolvedPath, `pack path ${resolvedPath}`);

  if (!isPathWithinRoot(canonicalRoot, canonicalPath)) {
    throw new Error(
      `Pack path ${canonicalPath} is outside repo root ${canonicalRoot}`,
    );
  }
  return canonicalPath;
};

const loadPackContent = (repoRoot: string, packPath: string): string => {
  const resolved = resolvePackPath(repoRoot, packPath);
  return fs.readFileSync(resolved, 'utf8').trim();
};

const formatPackContext = (
  intentId: string,
  primary: PackMaterialized,
  plan: PackPlan | null,
  additional: PackMaterialized[],
): string => {
  const lines: string[] = [];
  const safeIntentId = escapeXmlAttr(intentId);
  const safePackId = escapeXmlAttr(primary.packId);
  const safeScope = escapeXmlAttr(primary.scope);
  lines.push(
    `<pack-context intent="${safeIntentId}" primary-pack="${safePackId}" scope="${safeScope}">`,
  );
  if (plan) {
    lines.push(`## Active Workflow: ${escapeXmlText(plan.workflow_id)}`);
    if (plan.task) {
      lines.push(`Task: ${escapeXmlText(plan.task)}`);
    }
    if (plan.injection_text) {
      lines.push(escapeXmlText(plan.injection_text));
    }
  } else {
    lines.push(`## Active Workflow: ${escapeXmlText(primary.packId)}`);
  }
  lines.push('');
  // Pack files are trusted operator-authored markdown/yaml content and are intentionally
  // injected verbatim (not XML-escaped) so instructions remain machine-readable.
  lines.push(primary.content);

  for (const pack of additional) {
    lines.push('');
    const safePackId = escapeXmlText(pack.packId);
    const modeLabel = pack.mode ? ` (${escapeXmlText(pack.mode)})` : '';
    lines.push(`### Composition: ${safePackId}${modeLabel}`);
    lines.push(pack.content);
  }

  lines.push('</pack-context>');
  return lines.join('\n');
};

const parseRouterOutput = (raw: string): PackPlan => {
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch (error) {
    throw new Error(`Pack router returned invalid JSON: ${(error as Error).message}`);
  }

  if (!isRecord(parsed)) {
    throw new Error('Pack router returned invalid plan');
  }

  const packs = parsed.packs;
  if (!Array.isArray(packs) || packs.length === 0) {
    throw new Error('Pack router returned invalid plan');
  }

  const validatedPacks = packs.map((pack, index) => {
    if (!isRecord(pack)) {
      throw new Error(`Pack router returned invalid packs[${index}]`);
    }
    const packId = pack.pack_id;
    const query = pack.query;
    if (!isNonEmptyString(packId) || !isNonEmptyString(query)) {
      throw new Error(`Pack router returned invalid packs[${index}]`);
    }
    return { pack_id: packId, query };
  });

  const consumer = parsed.consumer;
  if (!isNonEmptyString(consumer)) {
    throw new Error('Pack router returned invalid consumer');
  }
  const workflowId = parsed.workflow_id;
  if (!isNonEmptyString(workflowId)) {
    throw new Error('Pack router returned invalid workflow_id');
  }
  const stepId = parsed.step_id;
  if (!isNonEmptyString(stepId)) {
    throw new Error('Pack router returned invalid step_id');
  }
  const scope = parsed.scope;
  if (!isNonEmptyString(scope)) {
    throw new Error('Pack router returned invalid scope');
  }

  const task = typeof parsed.task === 'string' ? parsed.task : '';
  const injectionText =
    typeof parsed.injection_text === 'string' ? parsed.injection_text : '';

  return {
    consumer,
    workflow_id: workflowId,
    step_id: stepId,
    scope,
    task,
    injection_text: injectionText,
    packs: validatedPacks,
  };
};

const containsNullByte = (value: string): boolean => {
  return value.includes('\u0000');
};

const MAX_ROUTER_OUTPUT_BYTES = 1_000_000;

const splitCommandString = (command: string): string[] => {
  const trimmed = command.trim();
  if (!trimmed) {
    throw new Error('Pack router command is empty');
  }

  const parts: string[] = [];
  let current = '';
  let quote: 'single' | 'double' | null = null;

  for (let i = 0; i < trimmed.length; i += 1) {
    const char = trimmed[i];

    // Support shell-like escapes outside single quotes and inside double quotes
    // (spaces, quotes, and backslashes) for quoted path compatibility.
    if (char === '\\' && quote !== 'single') {
      const next = trimmed[i + 1];
      if (next && /[\s"'\\]/.test(next)) {
        current += next;
        i += 1;
        continue;
      }
    }

    if (char === "'" && quote !== 'double') {
      quote = quote === 'single' ? null : 'single';
      continue;
    }

    if (char === '"' && quote !== 'single') {
      quote = quote === 'double' ? null : 'double';
      continue;
    }

    if (!quote && /\s/.test(char)) {
      if (current) {
        parts.push(current);
        current = '';
      }
      continue;
    }

    current += char;
  }

  if (quote) {
    throw new Error('Pack router command has unterminated quotes');
  }

  if (current) {
    parts.push(current);
  }

  if (parts.length === 0) {
    throw new Error('Pack router command is empty');
  }

  return parts;
};

const normalizeRouterCommand = (command: string | string[]): string[] => {
  const parts = Array.isArray(command)
    ? command.map((part) => part.trim()).filter((part) => part.length > 0)
    : splitCommandString(command);

  if (parts.length === 0) {
    throw new Error('Pack router command is empty');
  }

  if (parts.some((part) => containsNullByte(part))) {
    throw new Error('Pack router command contains invalid null bytes');
  }

  return parts;
};

const runPackRouter = (
  command: string | string[],
  args: string[],
  timeoutMs: number,
): Promise<string> => {
  return new Promise((resolve, reject) => {
    let settled = false;
    const resolveOnce = (value: string) => {
      if (settled) {
        return;
      }
      settled = true;
      resolve(value);
    };
    const rejectOnce = (error: Error) => {
      if (settled) {
        return;
      }
      settled = true;
      reject(error);
    };

    let commandParts: string[];
    try {
      commandParts = normalizeRouterCommand(command);
    } catch (error) {
      rejectOnce(error as Error);
      return;
    }

    if (args.some((arg) => containsNullByte(arg))) {
      rejectOnce(new Error('Pack router args contain invalid null bytes'));
      return;
    }

    const [cmd, ...baseArgs] = commandParts;
    const child = spawn(cmd, [...baseArgs, ...args], {
      stdio: ['ignore', 'pipe', 'pipe'],
      shell: false,
    });
    let stdout = '';
    let stderr = '';
    let stdoutBytes = 0;
    let stderrBytes = 0;

    const timeout = setTimeout(() => {
      child.kill('SIGKILL');
      rejectOnce(new Error('Pack router timed out'));
    }, timeoutMs);

    child.stdout.on('data', (chunk: Buffer) => {
      stdoutBytes += chunk.length;
      if (stdoutBytes > MAX_ROUTER_OUTPUT_BYTES) {
        child.kill('SIGKILL');
        rejectOnce(new Error('Pack router exceeded stdout size limit'));
        return;
      }
      stdout += chunk.toString();
    });
    child.stderr.on('data', (chunk: Buffer) => {
      stderrBytes += chunk.length;
      if (stderrBytes > MAX_ROUTER_OUTPUT_BYTES) {
        child.kill('SIGKILL');
        rejectOnce(new Error('Pack router exceeded stderr size limit'));
        return;
      }
      stderr += chunk.toString();
    });
    child.on('error', (error) => {
      clearTimeout(timeout);
      rejectOnce(error as Error);
    });
    child.on('close', (code) => {
      clearTimeout(timeout);
      if (code !== 0) {
        const detail = stderr.trim() || `exit code ${code}`;
        rejectOnce(new Error(`Pack router failed: ${detail}`));
        return;
      }
      resolveOnce(stdout.trim());
    });
  });
};

const resolvePrimaryPackViaRegistry = (
  registry: PackRegistry,
  packType: string,
  repoRoot: string,
): PackMaterialized => {
  const entry = resolveRegistryEntry(registry, packType);
  if (!entry) {
    throw new Error(`Pack registry missing pack type: ${packType}`);
  }
  const content = loadPackContent(repoRoot, entry.path);
  return {
    packId: entry.pack_id,
    scope: entry.scope,
    content,
  };
};

const resolveAdditionalPacks = (
  registry: PackRegistry | null | undefined,
  repoRoot: string,
  additional: ResolvedComposition[],
  logger: (message: string) => void,
): PackMaterialized[] => {
  if (!registry) {
    if (additional.length > 0) {
      logger('Pack registry missing; skipping composition packs.');
    }
    return [];
  }

  const resolved: PackMaterialized[] = [];
  for (const item of additional) {
    const entry = resolveRegistryEntry(registry, item.packType);
    if (!entry) {
      const message = `Composition pack missing: ${item.packType}`;
      if (item.required) {
        logger(`${message} (required)`);
      } else {
        logger(`${message} (optional)`);
      }
      continue;
    }

    resolved.push({
      packId: entry.pack_id,
      scope: entry.scope,
      content: loadPackContent(repoRoot, entry.path),
      mode: item.mode,
    });
  }

  return resolved;
};

export const createPackInjector = (deps: PackInjectorDeps) => {
  const config = normalizeConfig(deps.config);
  const logger = config.debug ? (message: string) => console.log(message) : () => undefined;
  const state = new Map<string, string>();

  return async (input: PackInjectorInput): Promise<PackContextResult | null> => {
    try {
      const sessionKey = input.ctx.sessionKey ?? input.ctx.messageProvider?.groupId ?? 'default';
      const previousIntentId = state.get(sessionKey);
      const { decision } = detectIntent(deps.intentRules, {
        prompt: input.prompt,
        graphitiResults: input.graphitiResults ?? undefined,
        previousIntentId,
        enableSticky: config.enableSticky,
        stickyMaxWords: config.stickyMaxWords,
        stickySignals: config.stickySignals,
        defaultMinConfidence: config.defaultMinConfidence,
        logger,
      });

      if (!decision.matched || !decision.rule) {
        state.delete(sessionKey);
        return null;
      }

      if (decision.rule.scope === 'private' && isGroupChat(input.ctx)) {
        logger('Skipping private intent in group chat.');
        state.delete(sessionKey);
        return null;
      }

      const repoRoot = config.packRouterRepoRoot ?? process.cwd();
      let primaryPack: PackMaterialized | null = null;
      let plan: PackPlan | null = null;

      if (config.packRouterCommand) {
        const args = [
          '--consumer',
          decision.rule.consumerProfile,
          '--workflow-id',
          decision.rule.workflowId ?? decision.rule.id,
          '--step-id',
          decision.rule.stepId ?? 'draft',
          '--task',
          decision.rule.task ?? '',
          '--injection-text',
          decision.rule.injectionText ?? '',
          '--repo',
          repoRoot,
        ];
        const output = await runPackRouter(
          config.packRouterCommand,
          args,
          config.packRouterTimeoutMs,
        );
        plan = parseRouterOutput(output);
        const primaryEntry = plan.packs[0];
        primaryPack = {
          packId: primaryEntry.pack_id,
          scope: plan.scope,
          content: loadPackContent(repoRoot, primaryEntry.query),
        };
      } else if (deps.packRegistry) {
        const packType = decision.rule.packType ?? decision.rule.id;
        primaryPack = resolvePrimaryPackViaRegistry(deps.packRegistry, packType, repoRoot);
      }

      if (!primaryPack) {
        logger('Primary pack resolution failed.');
        state.delete(sessionKey);
        return null;
      }

      if (primaryPack.scope === 'private' && isGroupChat(input.ctx)) {
        logger('Skipping private pack in group chat.');
        state.delete(sessionKey);
        return null;
      }

      const composition = resolveComposition(deps.compositionRules ?? null, decision.rule.id);
      let additional = resolveAdditionalPacks(
        deps.packRegistry,
        repoRoot,
        composition,
        logger,
      );

      if (isGroupChat(input.ctx)) {
        additional = additional.filter((pack) => pack.scope !== 'private');
      }

      const context = formatPackContext(decision.rule.id, primaryPack, plan, additional);
      state.set(sessionKey, decision.rule.id);

      return {
        context,
        intentId: decision.rule.id,
        primaryPackId: primaryPack.packId,
        scope: primaryPack.scope,
      };
    } catch (error) {
      logger(`Pack injector error: ${(error as Error).message}`);
      return null;
    }
  };
};
