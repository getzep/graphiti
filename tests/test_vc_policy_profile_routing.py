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


class ExamplePolicyProfileRoutingTests(unittest.TestCase):
    def _seed_configs(self, repo: Path) -> None:
        (repo / 'config').mkdir(parents=True, exist_ok=True)
        (repo / 'workflows').mkdir(parents=True, exist_ok=True)
        shutil.copyfile(
            REPO_ROOT / 'config/runtime_pack_registry.yaml',
            repo / 'config/runtime_pack_registry.yaml',
        )
        shutil.copyfile(
            REPO_ROOT / 'config/runtime_consumer_profiles.yaml',
            repo / 'config/runtime_consumer_profiles.yaml',
        )
        for filename in (
            'example_summary.pack.yaml',
            'example_research.pack.yaml',
        ):
            shutil.copyfile(
                REPO_ROOT / 'workflows' / filename,
                repo / 'workflows' / filename,
            )

    def _run(self, repo: Path, *, consumer: str, workflow_id: str, step_id: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
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
                'Unit test routing',
                '--validate',
            ],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_example_profiles_route_to_expected_packs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / 'repo'
            repo.mkdir(parents=True)
            self._seed_configs(repo)

            summary = self._run(
                repo,
                consumer='main_session_example_summary',
                workflow_id='example_summary',
                step_id='draft',
            )
            research = self._run(
                repo,
                consumer='main_session_example_research',
                workflow_id='example_research',
                step_id='synthesize',
            )

            self.assertEqual(summary.returncode, 0, msg=summary.stderr)
            self.assertEqual(research.returncode, 0, msg=research.stderr)

            self.assertIn('example_summary_pack', summary.stderr + summary.stdout)
            self.assertIn('example_research_pack', research.stderr + research.stdout)

    def test_invalid_pack_reference_is_a_misconfiguration_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / 'repo'
            repo.mkdir(parents=True)
            self._seed_configs(repo)

            profiles = json.loads(
                (repo / 'config/runtime_consumer_profiles.yaml').read_text(encoding='utf-8'),
            )
            profiles['profiles'][0]['pack_ids'] = ['nonexistent_pack', 'example_summary_pack']
            (repo / 'config/runtime_consumer_profiles.yaml').write_text(
                json.dumps(profiles, indent=2),
                encoding='utf-8',
            )

            result = self._run(
                repo,
                consumer='main_session_example_summary',
                workflow_id='example_summary',
                step_id='draft',
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn('pack_id not found in registry', result.stderr)
