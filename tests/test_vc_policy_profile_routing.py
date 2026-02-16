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


class VcPolicyProfileRoutingTests(unittest.TestCase):
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
            'vc_memo_drafting.pack.yaml',
            'vc_deal_brief.pack.yaml',
            'vc_diligence_questions.pack.yaml',
            'vc_ic_prep.pack.yaml',
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

    def test_diligence_and_ic_profiles_route_to_expected_packs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / 'repo'
            repo.mkdir(parents=True)
            self._seed_configs(repo)

            diligence = self._run(
                repo,
                consumer='main_session_vc_diligence_questions',
                workflow_id='vc_diligence_questions',
                step_id='draft',
            )
            ic = self._run(
                repo,
                consumer='main_session_vc_ic_prep',
                workflow_id='vc_ic_prep',
                step_id='synthesize',
            )

            self.assertEqual(diligence.returncode, 0, msg=diligence.stderr)
            self.assertEqual(ic.returncode, 0, msg=ic.stderr)

            self.assertIn('vc_diligence_questions', diligence.stderr + diligence.stdout)
            self.assertIn('vc_ic_prep', ic.stderr + ic.stdout)

    def test_invalid_pack_reference_is_a_misconfiguration_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / 'repo'
            repo.mkdir(parents=True)
            self._seed_configs(repo)

            profiles = json.loads(
                (repo / 'config/runtime_consumer_profiles.yaml').read_text(encoding='utf-8'),
            )
            profiles['profiles'][0]['pack_ids'] = ['nonexistent_pack', 'vc_memo_drafting']
            (repo / 'config/runtime_consumer_profiles.yaml').write_text(
                json.dumps(profiles, indent=2),
                encoding='utf-8',
            )

            result = self._run(
                repo,
                consumer='main_session_vc_memo',
                workflow_id='vc_memo_drafting',
                step_id='draft',
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn('pack_id not found in registry', result.stderr)

