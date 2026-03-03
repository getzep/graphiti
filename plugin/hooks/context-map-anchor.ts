import { createHash } from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';

import { isPathWithinRoot, toCanonicalPath } from '../path-utils.ts';
import { normalizeConfig } from '../config.ts';
import type { PluginConfig } from '../config.ts';

const CONTEXT_MAP_ANCHOR_STATE = Symbol.for('bicameral.context-map-anchor.state');
const CONTEXT_MAP_ANCHOR_STATE_CACHE = new WeakMap<object, ContextMapAnchorState>();
const DEFAULT_ANCHOR_TEXT =
  'Use the context map as authoritative structure for this session when relevant.';

const MAX_ANCHOR_TEXT_LENGTH = 512;
const MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024; // 10 MB

interface ContextMapAnchorState {
  pendingTrigger: boolean;
  lastFingerprint?: string;
}

type ContextMapAnchorContext = object;

type ContextMapAnchorHookResult = { prependContext?: string };

type ContextMapAnchorHook = (
  event: unknown,
  ctx: ContextMapAnchorContext,
) => Promise<ContextMapAnchorHookResult>;

interface ContextMapAnchorHooks {
  session_start: ContextMapAnchorHook;
  after_compaction: ContextMapAnchorHook;
  before_reset: ContextMapAnchorHook;
  before_prompt_build: ContextMapAnchorHook;
}

interface ContextMapAnchorHookDeps {
  config?: Partial<PluginConfig>;
}

type MarkableContext = ContextMapAnchorContext & {
  [CONTEXT_MAP_ANCHOR_STATE]?: ContextMapAnchorState;
};

const normalizeOptionalString = (value?: string): string | undefined => {
  if (!value) {
    return undefined;
  }
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : undefined;
};

/**
 * Strip ASCII control characters (except \t \n \r) and enforce a max length.
 * Prevents injection of escape sequences or overly-long config text into prompts.
 */
const sanitizeAnchorText = (text: string): string => {
  // Strip C0 controls except HT (0x09), LF (0x0a), CR (0x0d), and DEL (0x7f)
  const stripped = text.replace(/[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]/g, '');
  return stripped.length > MAX_ANCHOR_TEXT_LENGTH
    ? stripped.slice(0, MAX_ANCHOR_TEXT_LENGTH)
    : stripped;
};

const getState = (ctx: ContextMapAnchorContext): ContextMapAnchorState => {
  if (!ctx || typeof ctx !== 'object') {
    return { pendingTrigger: false };
  }

  if (CONTEXT_MAP_ANCHOR_STATE_CACHE.has(ctx)) {
    return CONTEXT_MAP_ANCHOR_STATE_CACHE.get(ctx)!;
  }

  let state: ContextMapAnchorState | undefined;
  try {
    state = (ctx as MarkableContext)[CONTEXT_MAP_ANCHOR_STATE];
  } catch {
    state = undefined;
  }

  if (!state) {
    state = { pendingTrigger: false };
    try {
      (ctx as MarkableContext)[CONTEXT_MAP_ANCHOR_STATE] = state;
    } catch {
      // Some runtimes freeze/lock context objects.
    }
  }

  CONTEXT_MAP_ANCHOR_STATE_CACHE.set(ctx, state);
  return state;
};

const markTriggered = (ctx: ContextMapAnchorContext): void => {
  const state = getState(ctx);
  state.pendingTrigger = true;
};

const clearState = (ctx: ContextMapAnchorContext): void => {
  if (!ctx || typeof ctx !== 'object') {
    return;
  }
  const state = getState(ctx);
  state.pendingTrigger = false;
  state.lastFingerprint = undefined;
};

/**
 * Resolve a configured file path and validate it stays within the canonical root
 * (when a root is explicitly configured via packRouterRepoRoot).
 *
 * - Relative paths are always resolved against the canonical root.
 * - Absolute paths are accepted as-is when no explicit root is configured, but
 *   are still validated against root when packRouterRepoRoot is set.
 * - Returns undefined when the path is absent or unsafe (logs via logger).
 */
const resolveAndValidatePath = (
  filePath: string | undefined,
  canonicalRoot: string | undefined,
  explicitRoot: boolean,
  logger: (msg: string) => void,
): string | undefined => {
  const normalized = normalizeOptionalString(filePath);
  if (!normalized) {
    return undefined;
  }

  // Resolve relative paths against root (or cwd if no root); absolute paths kept as-is.
  const base = canonicalRoot ?? process.cwd();
  const resolved = path.resolve(base, normalized);

  // Enforce within-root constraint only when the operator explicitly set packRouterRepoRoot.
  if (explicitRoot && canonicalRoot && !isPathWithinRoot(canonicalRoot, resolved)) {
    logger(
      `[security] context-map path ${resolved} escapes configured root ${canonicalRoot}; ignoring`,
    );
    return undefined;
  }

  return resolved;
};

const computeContextMapFingerprint = (
  config: PluginConfig,
  canonicalRoot: string | undefined,
  explicitRoot: boolean,
  logger: (msg: string) => void,
): string | undefined => {
  const mapPath = resolveAndValidatePath(config.contextMapPath, canonicalRoot, explicitRoot, logger);
  const metaPath = resolveAndValidatePath(config.contextMapMetaPath, canonicalRoot, explicitRoot, logger);

  const hasher = createHash('sha256');
  let consumed = false;

  const maybeAdd = (label: 'map' | 'meta', filePath?: string): void => {
    if (!filePath) {
      return;
    }
    try {
      // Stat first: regular-file check + size guard (avoids large or non-regular reads).
      let stats: fs.Stats;
      try {
        stats = fs.statSync(filePath);
      } catch {
        // ENOENT or similar — treat as absent.
        return;
      }
      if (!stats.isFile()) {
        return;
      }
      if (stats.size > MAX_FILE_SIZE_BYTES) {
        logger(
          `context-map ${label} at ${filePath} too large (${stats.size} bytes > ${MAX_FILE_SIZE_BYTES}); skipping`,
        );
        return;
      }

      // Symlink-escape check: canonicalize and re-validate within root when root is configured.
      if (explicitRoot && canonicalRoot) {
        let canonical: string;
        try {
          canonical = fs.realpathSync(filePath);
        } catch {
          return;
        }
        if (!isPathWithinRoot(canonicalRoot, canonical)) {
          logger(
            `[security] context-map ${label} at ${filePath} resolves to ${canonical} outside configured root; skipping`,
          );
          return;
        }
      }

      const content = fs.readFileSync(filePath);
      hasher.update(`${label}\0${filePath}\0`, 'utf8');
      hasher.update(content);
      consumed = true;
    } catch {
      // Any other unexpected error — treat as absent.
    }
  };

  maybeAdd('map', mapPath);
  maybeAdd('meta', metaPath);

  if (!consumed) {
    return undefined;
  }

  return hasher.digest('hex');
};

const escapeXml = (text: string): string =>
  text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

const renderAnchorContext = (config: PluginConfig): string => {
  const mapPath = normalizeOptionalString(config.contextMapPath);
  const metaPath = normalizeOptionalString(config.contextMapMetaPath);
  const rawAnchorText =
    normalizeOptionalString(config.contextMapAnchorText) ?? DEFAULT_ANCHOR_TEXT;
  // Sanitize before embedding into prompt context.
  const anchorText = sanitizeAnchorText(rawAnchorText);

  const lines: string[] = ['<context-map-anchor>', escapeXml(anchorText)];
  if (mapPath) {
    lines.push(`map_path: ${escapeXml(mapPath)}`);
  }
  if (metaPath) {
    lines.push(`meta_path: ${escapeXml(metaPath)}`);
  }
  lines.push('</context-map-anchor>');
  return lines.join('\n');
};

export const createContextMapAnchorHooks = (
  deps?: ContextMapAnchorHookDeps,
): ContextMapAnchorHooks => {
  const config = normalizeConfig(deps?.config);
  const logger = config.debug ? (message: string) => console.log(message) : () => undefined;

  // Compute and cache canonical root only when explicitly configured.
  // If packRouterRepoRoot is not set, root-based validation is relaxed (no root to validate against).
  const explicitRoot = !!normalizeOptionalString(config.packRouterRepoRoot);
  let canonicalRoot: string | undefined;
  if (explicitRoot) {
    const rawRoot = path.resolve(config.packRouterRepoRoot!);
    try {
      canonicalRoot = toCanonicalPath(rawRoot, `packRouterRepoRoot ${rawRoot}`);
    } catch {
      // Root may not exist yet on first run; fall back to unresolved but string-checked path.
      canonicalRoot = rawRoot;
    }
  }

  return {
    session_start: async (_event, ctx) => {
      if (!config.enableContextMapAnchor) {
        return {};
      }
      markTriggered(ctx);
      return {};
    },
    after_compaction: async (_event, ctx) => {
      if (!config.enableContextMapAnchor) {
        return {};
      }
      markTriggered(ctx);
      return {};
    },
    before_reset: async (_event, ctx) => {
      clearState(ctx);
      return {};
    },
    before_prompt_build: async (_event, ctx) => {
      if (!config.enableContextMapAnchor) {
        return {};
      }

      const state = getState(ctx);
      if (!state.pendingTrigger) {
        return {};
      }
      state.pendingTrigger = false;

      const fingerprint = computeContextMapFingerprint(
        config,
        canonicalRoot,
        explicitRoot,
        logger,
      );
      if (!fingerprint) {
        return {};
      }
      if (state.lastFingerprint === fingerprint) {
        return {};
      }

      state.lastFingerprint = fingerprint;
      return { prependContext: renderAnchorContext(config) };
    },
  };
};
