import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { createPackInjector } from '../hooks/pack-injector.ts';
import { stripInjectedContext } from '../hooks/capture.ts';
import { detectIntent } from '../intent/detector.ts';
import type { IntentRuleSet } from '../intent/types.ts';
import { loadIntentRules } from '../config.ts';
import type { PackRegistry } from '../config.ts';

const rules: IntentRuleSet = {
  schema_version: 1,
  rules: [
    {
      id: 'summary',
      consumerProfile: 'main_session_example_summary',
      workflowId: 'example_summary',
      stepId: 'draft',
      packType: 'example_summary_pack',
      keywords: ['summary', 'recap'],
      keywordWeight: 1,
      minConfidence: 0.3,
      scope: 'public',
      entityBoosts: [
        {
          summaryPattern: 'report',
          weight: 0.5,
        },
      ],
    },
    {
      id: 'research',
      consumerProfile: 'main_session_example_research',
      workflowId: 'example_research',
      stepId: 'synthesize',
      packType: 'example_research_pack',
      keywords: ['research', 'analysis'],
      keywordWeight: 1,
      minConfidence: 0.3,
      scope: 'private',
    },
  ],
};

const registry: PackRegistry = {
  schema_version: 1,
  packs: [
    {
      pack_id: 'example_summary_pack',
      pack_type: 'example_summary_pack',
      path: 'workflows/example_summary.pack.yaml',
      scope: 'group-safe',
    },
    {
      pack_id: 'example_research_pack',
      pack_type: 'example_research_pack',
      path: 'workflows/example_research.pack.yaml',
      scope: 'private',
    },
  ],
};

const makeTempDir = (t: { after: (fn: () => void) => void }, prefix: string): string => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), prefix));
  t.after(() => {
    fs.rmSync(dir, { recursive: true, force: true });
  });
  return dir;
};

test('P1: no keyword match yields no pack', () => {
  const { decision } = detectIntent(rules, { prompt: 'hello there', defaultMinConfidence: 0.3 });
  assert.equal(decision.matched, false);
});

test('P2: deterministic selection for same input', () => {
  const input = { prompt: 'summary please', defaultMinConfidence: 0.3 };
  const first = detectIntent(rules, input).decision;
  const second = detectIntent(rules, input).decision;
  assert.deepEqual(first, second);
});

test('P3: tie yields no pack', () => {
  const tieRules: IntentRuleSet = {
    schema_version: 1,
    rules: [
      { id: 'a', consumerProfile: 'a', keywords: ['hello'], keywordWeight: 1 },
      { id: 'b', consumerProfile: 'b', keywords: ['hello'], keywordWeight: 1 },
    ],
  };
  const { decision } = detectIntent(tieRules, { prompt: 'hello', defaultMinConfidence: 0.3 });
  assert.equal(decision.matched, false);
});

test('P4: entity boost increases score', () => {
  const base = detectIntent(rules, { prompt: 'summary', defaultMinConfidence: 0.3 });
  const boosted = detectIntent(rules, {
    prompt: 'summary',
    graphitiResults: { facts: [{ fact: 'report mentions key finding' }] },
    defaultMinConfidence: 0.3,
  });
  assert.ok(boosted.decision.score >= base.decision.score);
});

test('P5: group chat blocks private packs', async () => {
  const injector = createPackInjector({ intentRules: rules, packRegistry: registry });
  const result = await injector({
    prompt: 'research analysis',
    ctx: { messageProvider: { chatType: 'group' } },
    graphitiResults: null,
  });
  assert.equal(result, null);
});

test('P6: injector errors fallback to null', async () => {
  const injector = createPackInjector({
    intentRules: {
      schema_version: 1,
      rules: [
        {
          id: 'missing',
          consumerProfile: 'missing',
          keywords: ['missing'],
          packType: 'does_not_exist',
        },
      ],
    },
    packRegistry: registry,
  });
  const result = await injector({
    prompt: 'missing',
    ctx: {},
    graphitiResults: null,
  });
  assert.equal(result, null);
});

test('capture strips injected context blocks', () => {
  const raw = `hello\n<graphiti-context>facts</graphiti-context>\n<pack-context intent="x">pack</pack-context>`;
  assert.equal(stripInjectedContext(raw), 'hello');
});

test('invalid regex patterns log debug output', () => {
  const logs: string[] = [];
  detectIntent(
    {
      schema_version: 1,
      rules: [
        {
          id: 'bad_regex',
          consumerProfile: 'bad_regex',
          keywords: ['summary'],
          entityBoosts: [{ summaryPattern: '[', weight: 0.5 }],
        },
      ],
    },
    {
      prompt: 'summary',
      defaultMinConfidence: 0.3,
      graphitiResults: { facts: [], entities: [] },
      logger: (message) => logs.push(message),
    },
  );
  assert.ok(logs.some((entry) => entry.includes('Invalid regex pattern')));
});

test('pack context escapes XML attributes', async () => {
  const intentId = 'intent "alpha" & <beta>';
  const packId = 'pack "alpha" & <beta>';
  const injector = createPackInjector({
    intentRules: {
      schema_version: 1,
      rules: [
        {
          id: intentId,
          consumerProfile: 'main_session_example_summary',
          workflowId: 'example_summary',
          stepId: 'draft',
          packType: packId,
          keywords: ['summary'],
          keywordWeight: 1,
          minConfidence: 0.3,
          scope: 'public',
        },
      ],
    },
    packRegistry: {
      schema_version: 1,
      packs: [
        {
          pack_id: packId,
          pack_type: packId,
          path: 'workflows/example_summary.pack.yaml',
          scope: 'public',
        },
      ],
    },
  });

  const result = await injector({
    prompt: 'summary',
    ctx: {},
    graphitiResults: null,
  });

  assert.ok(result);
  assert.ok(
    result.context.includes(
      'intent="intent &quot;alpha&quot; &amp; &lt;beta&gt;"',
    ),
  );
  assert.ok(
    result.context.includes(
      'primary-pack="pack &quot;alpha&quot; &amp; &lt;beta&gt;"',
    ),
  );
});

test('pack router command supports quoted paths with spaces', async (t) => {
  const tempDir = makeTempDir(t, 'graphiti pack router ');
  const packFile = path.join(tempDir, 'pack.yaml');
  fs.writeFileSync(packFile, 'router pack content', 'utf8');

  const plan = {
    consumer: 'main_session_example_summary',
    workflow_id: 'example_summary',
    step_id: 'draft',
    scope: 'public',
    task: '',
    injection_text: '',
    packs: [{ pack_id: 'router_pack', query: 'pack.yaml' }],
  };

  const scriptPath = path.join(tempDir, 'pack router.js');
  fs.writeFileSync(
    scriptPath,
    `process.stdout.write(${JSON.stringify(JSON.stringify(plan))});`,
    'utf8',
  );

  const injector = createPackInjector({
    intentRules: {
      schema_version: 1,
      rules: [
        {
          id: 'summary',
          consumerProfile: 'main_session_example_summary',
          workflowId: 'example_summary',
          stepId: 'draft',
          keywords: ['summary'],
        },
      ],
    },
    config: {
      packRouterCommand: `node "${scriptPath}"`,
      packRouterRepoRoot: tempDir,
    },
  });

  const result = await injector({
    prompt: 'summary',
    ctx: {},
    graphitiResults: null,
  });

  assert.ok(result);
  assert.ok(result.context.includes('router pack content'));
});

test('invalid pack router output falls back to null', async (t) => {
  const tempDir = makeTempDir(t, 'graphiti pack router invalid ');
  const scriptPath = path.join(tempDir, 'pack router.js');
  fs.writeFileSync(scriptPath, 'process.stdout.write("{\\"packs\\": []}");', 'utf8');

  const injector = createPackInjector({
    intentRules: {
      schema_version: 1,
      rules: [
        {
          id: 'summary',
          consumerProfile: 'main_session_example_summary',
          workflowId: 'example_summary',
          stepId: 'draft',
          keywords: ['summary'],
        },
      ],
    },
    config: {
      packRouterCommand: `node "${scriptPath}"`,
      packRouterRepoRoot: tempDir,
    },
  });

  const result = await injector({
    prompt: 'summary',
    ctx: {},
    graphitiResults: null,
  });

  assert.equal(result, null);
});

test('pack router plan cannot escape repo root through symlink', async (t) => {
  const tempDir = makeTempDir(t, 'graphiti-pack-router-symlink-');
  const repoRoot = path.join(tempDir, 'repo');
  fs.mkdirSync(repoRoot, { recursive: true });

  const externalDir = makeTempDir(t, 'graphiti-pack-router-external-');
  const externalPackPath = path.join(externalDir, 'outside-pack.yaml');
  fs.writeFileSync(externalPackPath, 'outside content', 'utf8');

  const symlinkedPackPath = path.join(repoRoot, 'linked-pack.yaml');
  fs.symlinkSync(externalPackPath, symlinkedPackPath);

  const plan = {
    consumer: 'main_session_example_summary',
    workflow_id: 'example_summary',
    step_id: 'draft',
    scope: 'public',
    task: '',
    injection_text: '',
    packs: [{ pack_id: 'router_pack', query: 'linked-pack.yaml' }],
  };

  const scriptPath = path.join(tempDir, 'pack-router.js');
  fs.writeFileSync(
    scriptPath,
    `process.stdout.write(${JSON.stringify(JSON.stringify(plan))});`,
    'utf8',
  );

  const injector = createPackInjector({
    intentRules: {
      schema_version: 1,
      rules: [
        {
          id: 'summary',
          consumerProfile: 'main_session_example_summary',
          workflowId: 'example_summary',
          stepId: 'draft',
          keywords: ['summary'],
        },
      ],
    },
    config: {
      packRouterCommand: ['node', scriptPath],
      packRouterRepoRoot: repoRoot,
    },
  });

  const result = await injector({
    prompt: 'summary',
    ctx: {},
    graphitiResults: null,
  });

  assert.equal(result, null);
});

test('config path allowlist rejects outside roots', (t) => {
  const tempDir = makeTempDir(t, 'graphiti-config-');
  const rulesPath = path.join(tempDir, 'intent_rules.json');
  fs.writeFileSync(rulesPath, JSON.stringify({ schema_version: 1, rules: [] }), 'utf8');

  const allowedRoot = makeTempDir(t, 'graphiti-allowed-');
  assert.throws(
    () => loadIntentRules(rulesPath, [allowedRoot]),
    /outside allowed roots/,
  );
});

test('config path allowlist rejects symlink escapes', (t) => {
  const allowedRoot = makeTempDir(t, 'graphiti-allowed-root-');
  const externalRoot = makeTempDir(t, 'graphiti-external-root-');

  const externalFile = path.join(externalRoot, 'intent_rules.json');
  fs.writeFileSync(externalFile, JSON.stringify({ schema_version: 1, rules: [] }), 'utf8');

  const linkedPath = path.join(allowedRoot, 'intent_rules_link.json');
  fs.symlinkSync(externalFile, linkedPath);

  assert.throws(
    () => loadIntentRules(linkedPath, [allowedRoot]),
    /outside allowed roots/,
  );
});

test('pack context escapes XML text for workflow metadata', async (t) => {
  const tempDir = makeTempDir(t, 'graphiti-pack-router-xml-text-');
  const packFile = path.join(tempDir, 'pack.yaml');
  fs.writeFileSync(packFile, 'router pack content', 'utf8');

  const plan = {
    consumer: 'main_session_example_summary',
    workflow_id: 'wf <alpha> & beta',
    step_id: 'draft',
    scope: 'public',
    task: 'task <x> & y',
    injection_text: 'inject <tag> & z',
    packs: [{ pack_id: 'router_pack', query: 'pack.yaml' }],
  };

  const scriptPath = path.join(tempDir, 'pack-router.js');
  fs.writeFileSync(
    scriptPath,
    `process.stdout.write(${JSON.stringify(JSON.stringify(plan))});`,
    'utf8',
  );

  const injector = createPackInjector({
    intentRules: {
      schema_version: 1,
      rules: [
        {
          id: 'summary',
          consumerProfile: 'main_session_example_summary',
          workflowId: 'example_summary',
          stepId: 'draft',
          keywords: ['summary'],
        },
      ],
    },
    config: {
      packRouterCommand: ['node', scriptPath],
      packRouterRepoRoot: tempDir,
    },
  });

  const result = await injector({
    prompt: 'summary',
    ctx: {},
    graphitiResults: null,
  });

  assert.ok(result);
  assert.ok(result.context.includes('## Active Workflow: wf &lt;alpha&gt; &amp; beta'));
  assert.ok(result.context.includes('Task: task &lt;x&gt; &amp; y'));
  assert.ok(result.context.includes('inject &lt;tag&gt; &amp; z'));
});

test('pack router command path with spaces must be quoted or array form', async (t) => {
  const tempDir = makeTempDir(t, 'graphiti-router-unquoted-path-');
  const scriptPath = path.join(tempDir, 'pack router.js');
  fs.writeFileSync(scriptPath, 'process.stdout.write("{}")', 'utf8');

  const injector = createPackInjector({
    intentRules: {
      schema_version: 1,
      rules: [
        {
          id: 'summary',
          consumerProfile: 'main_session_example_summary',
          workflowId: 'example_summary',
          stepId: 'draft',
          keywords: ['summary'],
        },
      ],
    },
    config: {
      packRouterCommand: scriptPath,
      packRouterRepoRoot: tempDir,
    },
  });

  const result = await injector({
    prompt: 'summary',
    ctx: {},
    graphitiResults: null,
  });

  assert.equal(result, null);
});

test('config path allowlist rejects non-existent roots', (t) => {
  const tempDir = makeTempDir(t, 'graphiti-config-root-missing-');
  const rulesPath = path.join(tempDir, 'intent_rules.json');
  fs.writeFileSync(rulesPath, JSON.stringify({ schema_version: 1, rules: [] }), 'utf8');

  const missingRoot = path.join(tempDir, 'does-not-exist');
  assert.throws(
    () => loadIntentRules(rulesPath, [missingRoot]),
    /Unable to resolve config root/,
  );
});
