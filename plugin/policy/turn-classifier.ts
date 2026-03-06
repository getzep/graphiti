/**
 * Turn classifier for context orchestration policy.
 *
 * Classifies user turns into policy categories that control recall/injection behavior:
 * - low-info: operationally empty turns that should receive no context injection
 * - task-update: narrow task-state queries that should get constrained recall
 * - normal: everything else — full recall pipeline
 *
 * Design principles:
 * - Carveouts are checked BEFORE low-info suppression (short commands are not noise)
 * - Classification is deterministic and regex-free for the core path
 * - All thresholds are explicit constants, not magic numbers
 */

export type TurnClassification = 'low-info' | 'task-update' | 'normal';

export interface ClassificationResult {
  classification: TurnClassification;
  reason: string;
}

/**
 * Maximum token count (whitespace-split words) for a turn to be eligible
 * for low-info suppression. Turns longer than this are always 'normal'.
 */
const LOW_INFO_MAX_TOKENS = 3;

/**
 * Canonical low-info patterns — operationally empty turns.
 * Matched case-insensitively after trimming.
 */
const LOW_INFO_EXACT: Set<string> = new Set([
  'ok',
  'okay',
  'k',
  'kk',
  '?',
  'thanks',
  'thank you',
  'thx',
  'ty',
  'sounds good',
  'lol',
  'lmao',
  'haha',
  'nice',
  'cool',
  'great',
  'awesome',
  'got it',
  'gotcha',
  'noted',
  'sure',
  'right',
  'alright',
  'fine',
  'good',
  'perfect',
  'agreed',
  'ack',
  'np',
  'no worries',
  'all good',
  'makes sense',
  'fair enough',
  'understood',
  'copy',
  'roger',
  'bet',
  'word',
  'aight',
]);

/**
 * Short-turn carveouts — these look short but carry operational intent.
 * Must NOT be classified as low-info.
 * Matched case-insensitively after trimming and punctuation stripping.
 */
const CARVEOUT_EXACT: Set<string> = new Set([
  // Approval/stop decisions should preserve router semantics, even when terse.
  'yes',
  'no',
  'yep',
  'yup',
  'yeah',
  'nope',
  'nah',
  'approve',
  'approved',
  'confirm',
  'confirmed',
  'reject',
  'rejected',
  'stop',
  'cancel',
  'abort',
  'send it',
  'do it',
  'ship it',
  'run it',
  'go ahead',
  'go',
  'merge it',
  'deploy it',
  'push it',
  'cancel it',
  'stop it',
  'kill it',
  'retry',
  'redo it',
  'revert it',
  'book it',
  'schedule it',
]);

/**
 * Carveout patterns that match with a regex — handles parameterized short commands.
 * Examples: "11 works", "3pm works", "today?", "tomorrow?"
 */
const CARVEOUT_PATTERNS: RegExp[] = [
  // Time/number + "works" (e.g., "11 works", "3pm works", "noon works")
  /^\S+\s+works$/,
  // Date/time questions (e.g., "today?", "tomorrow?", "tuesday?", "tonight?")
  /^(today|tomorrow|tonight|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\??$/,
  // Short scheduling (e.g., "at 3", "at noon")
  /^at\s+\S+$/,
];

/**
 * Task-update request patterns.
 * Matched case-insensitively against the full prompt.
 */
const TASK_UPDATE_PATTERNS: RegExp[] = [
  /\bupdate\s+(me\s+)?on\s+(the\s+)?(previous|last|current|latest)\s+(task|work|job|run)\b/i,
  /\b(what'?s|what\s+is)\s+the\s+(status|state|progress)\s+(of|on)\s+(the\s+)?(previous|last|current|latest)\s+(task|work|job|run)\b/i,
  /\btask\s+(status|update|progress|state)\b/i,
  /\b(status|progress)\s+update\b/i,
  /\bwhere\s+(are\s+)?(we|things)\s+(at|on|with)\s+(the\s+)?(task|work|job)\b/i,
  /\bhow'?s\s+the\s+(task|work|job|run)\s+(going|doing|progressing)\b/i,
];

/**
 * Emoji-only detection: returns true if the entire string is composed of
 * emoji, variation selectors, ZWJ sequences, skin tone modifiers, and whitespace.
 *
 * Uses a permissive approach: strip all known emoji-adjacent codepoints,
 * then check if nothing remains.
 */
const isEmojiOnly = (text: string): boolean => {
  const trimmed = text.trim();
  if (trimmed.length === 0) return false;
  // Strip: emoji characters, variation selectors (FE0E/FE0F), ZWJ (200D),
  // combining enclosing keycap (20E3), skin tone modifiers (1F3FB-1F3FF),
  // regional indicators (1F1E6-1F1FF), and whitespace.
  const stripped = trimmed.replace(
    /[\p{Emoji_Presentation}\p{Emoji}\uFE0E\uFE0F\u200D\u20E3\u{1F3FB}-\u{1F3FF}\u{1F1E6}-\u{1F1FF}\s]/gu,
    '',
  );
  // If nothing remains after stripping, it was emoji-only.
  // But also ensure there was at least one actual emoji (not just variation selectors).
  if (stripped.length > 0) return false;
  return /\p{Emoji_Presentation}/u.test(trimmed) || /\p{Emoji}/u.test(trimmed);
};

const tokenCount = (text: string): number => {
  const trimmed = text.trim();
  if (!trimmed) return 0;
  return trimmed.split(/\s+/).length;
};

/**
 * Strip trailing punctuation for matching purposes.
 * Keeps the semantic content: "send it!" → "send it", "today?" → "today"
 */
const stripTrailingPunctuation = (text: string): string =>
  text.replace(/[!?.,:;]+$/, '').trim();

/**
 * Classify a user turn for context orchestration policy.
 *
 * Evaluation order:
 * 1. Empty/whitespace → low-info
 * 2. Carveout match → normal (short commands are not noise)
 * 3. Task-update match → task-update
 * 4. Low-info match → low-info (exact match or emoji-only)
 * 5. Default → normal
 */
export const classifyTurn = (prompt: string): ClassificationResult => {
  const trimmed = prompt.trim();

  // 1. Empty / whitespace-only
  if (trimmed.length === 0) {
    return { classification: 'low-info', reason: 'empty' };
  }

  const lower = trimmed.toLowerCase();
  const stripped = stripTrailingPunctuation(lower);

  // 2. Carveout check — must come before low-info so "today?" etc. are not suppressed
  if (CARVEOUT_EXACT.has(stripped)) {
    return { classification: 'normal', reason: `carveout:${stripped}` };
  }
  for (const pattern of CARVEOUT_PATTERNS) {
    if (pattern.test(stripped) || pattern.test(lower)) {
      return { classification: 'normal', reason: `carveout-pattern:${stripped}` };
    }
  }

  // 3. Task-update check
  for (const pattern of TASK_UPDATE_PATTERNS) {
    if (pattern.test(trimmed)) {
      return { classification: 'task-update', reason: 'task-update-request' };
    }
  }

  // 4. Low-info check
  // 4a. Exact match
  if (LOW_INFO_EXACT.has(lower) || LOW_INFO_EXACT.has(stripped)) {
    return { classification: 'low-info', reason: `low-info-exact:${stripped}` };
  }

  // 4b. Emoji-only
  if (isEmojiOnly(trimmed)) {
    return { classification: 'low-info', reason: 'emoji-only' };
  }

  // 4c. Very short turns that don't match carveouts and have no operational content
  // Only suppress if ≤ LOW_INFO_MAX_TOKENS AND not a question (questions may be short but meaningful)
  const tokens = tokenCount(trimmed);
  if (tokens <= LOW_INFO_MAX_TOKENS && !trimmed.includes('?')) {
    // Check if it's a bare acknowledgement variant we haven't listed
    // (e.g., "mm", "mhm", "hmm" when not a question)
    const isAcknowledgement = /^(m{1,4}h?m?|hm{1,3}|uh\s*huh|ya+h?)$/i.test(stripped);
    if (isAcknowledgement) {
      return { classification: 'low-info', reason: `low-info-ack:${stripped}` };
    }
  }

  // 5. Default
  return { classification: 'normal', reason: 'default' };
};
