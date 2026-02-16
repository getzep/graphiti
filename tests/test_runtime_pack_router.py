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
            destination.write_text(
                source.read_text(encoding='utf-8'),
                encoding='utf-8',
            )

        for filename in (
            'vc_memo_drafting.pack.yaml',
            'vc_deal_brief.pack.yaml',
            'vc_diligence_questions.pack.yaml',
            'vc_ic_prep.pack.yaml',
        ):
            source = REPO_ROOT / 'workflows' / filename
            destination = workflows_dir / filename
            destination.write_text(
                source.read_text(encoding='utf-8'),
                encoding='utf-8',
            )

    def _route(self, repo: Path, *, consumer: str, workflow_id: str, step_id: str, task: str) -> dict:
        out = repo / 'plan.json'
        result = subprocess.run(
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
                task,
                '--validate',
                '--out',
                str(out),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        payload = json.loads(out.read_text(encoding='utf-8'))
        return payload

    def test_router_routes_vc_consumers_deterministically(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / 'repo'
            repo.mkdir(parents=True)
            self._seed_repo(repo)

            memo_plan = self._route(
                repo,
                consumer='main_session_vc_memo',
                workflow_id='vc_memo_drafting',
                step_id='draft',
                task='Draft memo',
            )
            deal_plan = self._route(
                repo,
                consumer='main_session_vc_deal_brief',
                workflow_id='vc_deal_brief',
                step_id='compose',
                task='Prepare deal brief',
            )
            ic_plan = self._route(
                repo,
                consumer='main_session_vc_ic_prep',
                workflow_id='vc_ic_prep',
                step_id='synthesize',
                task='Prepare IC brief',
            )

            self.assertEqual(memo_plan['packs'][0]['pack_id'], 'vc_memo_drafting')
            self.assertEqual(memo_plan['packs'][0]['query'], 'workflows/vc_memo_drafting.pack.yaml')
            self.assertEqual(deal_plan['packs'][0]['pack_id'], 'vc_deal_brief')
            self.assertEqual(deal_plan['packs'][0]['query'], 'workflows/vc_deal_brief.pack.yaml')
            self.assertEqual(ic_plan['packs'][0]['pack_id'], 'vc_ic_prep')
            self.assertEqual(ic_plan['packs'][0]['query'], 'workflows/vc_ic_prep.pack.yaml')

            for plan in (memo_plan, deal_plan, ic_plan):
                self.assertNotIn('repo_path', plan)
                self.assertEqual(plan['consumer'].startswith('main_session_vc_'), True)
                self.assertIn('scope', plan)
                self.assertIn('packs', plan)

            memo_plan_replay = self._route(
                repo,
                consumer='main_session_vc_memo',
                workflow_id='vc_memo_drafting',
                step_id='draft',
                task='Draft memo',
            )
            self.assertEqual(memo_plan, memo_plan_replay)


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
                    'main_session_vc_memo',
                    '--workflow-id',
                    'vc_memo_drafting',
                    '--step-id',
                    'draft',
                    '--repo',
                    str(repo),
                    '--task',
                    'Draft memo',
                    '--validate',
                ],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 1)
            self.assertIn('.profiles[0].task', result.stderr)
            self.assertIn('must be a string', result.stderr)

