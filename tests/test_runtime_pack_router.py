from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'runtime_pack_router.py'
REPO_ROOT = Path(__file__).resolve().parents[1]


class RuntimePackRouterTests(unittest.TestCase):
    def _seed_repo(self, repo: Path) -> None:
        config_dir = repo / 'config'
        workflows_dir = repo / 'workflows'
        config_dir.mkdir(parents=True, exist_ok=True)
        workflows_dir.mkdir(parents=True, exist_ok=True)

        for filename in (
            'runtime_pack_registry.yaml',
            'runtime_consumer_profiles.yaml',
        ):
            source = REPO_ROOT / 'config' / filename
            destination = config_dir / filename
            destination.write_text(source.read_text(encoding='utf-8'), encoding='utf-8')

        for filename in (
            'example_summary.pack.yaml',
            'example_research.pack.yaml',
        ):
            source = REPO_ROOT / 'workflows' / filename
            destination = workflows_dir / filename
            destination.write_text(source.read_text(encoding='utf-8'), encoding='utf-8')

    def _route(
        self,
        repo: Path,
        *,
        consumer: str,
        workflow_id: str,
        step_id: str,
        task: str,
        materialize: bool = False,
        scope: str | None = None,
    ) -> dict:
        out = repo / 'plan.json'
        cmd = [
            sys.executable,
            str(SCRIPT),
            '--consumer',
            consumer,
            '--workflow-id',
            workflow_id,
            '--step-id',
            step_id,
            '--repo',
            str(repo),
            '--task',
            task,
            '--validate',
            '--out',
            str(out),
        ]
        if materialize:
            cmd.append('--materialize')
        if scope:
            cmd.extend(['--scope', scope])

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        return json.loads(out.read_text(encoding='utf-8'))

    def test_router_routes_example_consumers_deterministically(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / 'repo'
            repo.mkdir(parents=True)
            self._seed_repo(repo)

            summary_plan = self._route(
                repo,
                consumer='main_session_example_summary',
                workflow_id='example_summary',
                step_id='draft',
                task='Draft summary',
            )
            research_plan = self._route(
                repo,
                consumer='main_session_example_research',
                workflow_id='example_research',
                step_id='synthesize',
                task='Synthesize notes',
            )

            self.assertEqual(summary_plan['packs'][0]['pack_id'], 'example_summary_pack')
            self.assertEqual(summary_plan['packs'][0]['query'], 'workflows/example_summary.pack.yaml')
            self.assertEqual(research_plan['packs'][0]['pack_id'], 'example_research_pack')
            self.assertEqual(research_plan['packs'][0]['query'], 'workflows/example_research.pack.yaml')

            for plan in (summary_plan, research_plan):
                self.assertNotIn('repo_path', plan)
                self.assertTrue(plan['consumer'].startswith('main_session_example_'))
                self.assertIn('scope', plan)
                self.assertIn('packs', plan)
                self.assertIn('selected_packs', plan)
                self.assertIn('dropped_packs', plan)
                self.assertIn('decision_path', plan)
                self.assertIn('budget_summary', plan)

            replay = self._route(
                repo,
                consumer='main_session_example_summary',
                workflow_id='example_summary',
                step_id='draft',
                task='Draft summary',
            )
            self.assertEqual(summary_plan, replay)

    def test_router_accepts_scope_and_materialize_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / 'repo'
            repo.mkdir(parents=True)
            self._seed_repo(repo)

            plan = self._route(
                repo,
                consumer='main_session_example_research',
                workflow_id='example_research',
                step_id='synthesize',
                task='Synthesize notes',
                materialize=True,
                scope='private',
            )

            self.assertEqual(plan['scope'], 'private')
            self.assertEqual(len(plan['selected_packs']), 1)
            selected = plan['selected_packs'][0]
            self.assertEqual(selected['pack_id'], 'example_research_pack')
            self.assertIn('materialized_excerpt', selected)
            self.assertEqual(plan['budget_summary']['selected_count'], 1)


class RuntimePackRouterFixturesTests(unittest.TestCase):
    def test_misconfigured_profile_type_rejects_non_string_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source_repo = REPO_ROOT
            repo = Path(tmp) / 'repo'
            repo.mkdir(parents=True)
            (repo / 'config').mkdir(parents=True)

            profiles = json.loads((source_repo / 'config/runtime_consumer_profiles.yaml').read_text(encoding='utf-8'))
            profiles['profiles'][0]['task'] = 123
            (repo / 'config' / 'runtime_consumer_profiles.yaml').write_text(
                json.dumps(profiles, indent=2),
                encoding='utf-8',
            )

            shutil.copyfile(
                source_repo / 'config/runtime_pack_registry.yaml',
                repo / 'config/runtime_pack_registry.yaml',
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    '--consumer',
                    'main_session_example_summary',
                    '--workflow-id',
                    'example_summary',
                    '--step-id',
                    'draft',
                    '--repo',
                    str(repo),
                    '--task',
                    'Draft summary',
                    '--validate',
                ],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 1)
            self.assertIn('.profiles[0].task', result.stderr)
            self.assertIn('must be a string', result.stderr)

    def test_path_traversal_in_registry_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source_repo = REPO_ROOT
            repo = Path(tmp) / 'repo'
            repo.mkdir(parents=True)
            (repo / 'config').mkdir(parents=True)
            (repo / 'workflows').mkdir(parents=True)

            registry = json.loads((source_repo / 'config/runtime_pack_registry.yaml').read_text(encoding='utf-8'))
            registry['packs'][0]['path'] = '../../../etc/passwd'
            (repo / 'config' / 'runtime_pack_registry.yaml').write_text(
                json.dumps(registry, indent=2),
                encoding='utf-8',
            )
            shutil.copyfile(
                source_repo / 'config/runtime_consumer_profiles.yaml',
                repo / 'config/runtime_consumer_profiles.yaml',
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    '--consumer',
                    'main_session_example_summary',
                    '--workflow-id',
                    'example_summary',
                    '--step-id',
                    'draft',
                    '--repo',
                    str(repo),
                    '--task',
                    'Draft summary',
                    '--validate',
                ],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 1)
            self.assertIn('escapes repo root', result.stderr)
