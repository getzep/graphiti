from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'delta_contract_check.py'


def _valid_policy() -> dict:
    return {
        'version': 1,
        'origin': {'remote': 'origin', 'branch': 'main'},
        'upstream': {'remote': 'upstream', 'url': 'https://example.com/upstream.git', 'branch': 'main'},
        'sync_button_policy': {
            'require_clean_worktree': True,
            'max_origin_only_commits': 0,
            'require_upstream_only_commits': True,
        },
        'scorecard': {
            'clean_foundation_threshold': 80,
            'weights': {
                'privacy_risk': 0.35,
                'simplicity': 0.35,
                'merge_conflict_risk': 0.2,
                'auditability': 0.1,
            },
        },
        'schedule': {
            'timezone': 'America/New_York',
            'weekly_day': 'monday',
            'cron_utc': '0 14 * * 1',
        },
    }


def _valid_state_manifest() -> dict:
    return {
        'version': 1,
        'package_name': 'delta-state',
        'required_files': [
            'config/migration_sync_policy.json',
            'config/state_migration_manifest.json',
            'config/delta_contract_policy.json',
        ],
        'optional_globs': ['scripts/*.py'],
        'exclude_globs': ['**/__pycache__/**'],
    }


def _valid_contract_policy() -> dict:
    return {
        'version': 1,
        'targets': {
            'migration_sync_policy': {
                'current_version': 1,
                'migration_script': 'scripts/delta_contract_migrate.py',
                'notes': 'Migration sync policy remains on schema v1.',
            },
            'state_migration_manifest': {
                'current_version': 1,
                'migration_script': 'scripts/delta_contract_migrate.py',
                'notes': 'State migration manifest remains on schema v1.',
            },
            'extension_command_contract': {
                'current_version': 1,
                'migration_script': 'scripts/delta_contract_migrate.py',
                'notes': 'Commands must use <namespace>/<command>.',
            },
        },
    }


class DeltaContractCheckTests(unittest.TestCase):
    def _init_repo(self, root: Path) -> None:
        subprocess.run(['git', 'init'], cwd=root, check=True, capture_output=True, text=True)
        subprocess.run(['git', 'config', 'user.email', 'test@example.com'], cwd=root, check=True)
        subprocess.run(['git', 'config', 'user.name', 'Test User'], cwd=root, check=True)

    def _seed(self, root: Path) -> None:
        (root / 'config').mkdir(parents=True, exist_ok=True)
        (root / 'extensions' / 'sample').mkdir(parents=True, exist_ok=True)
        (root / 'scripts').mkdir(parents=True, exist_ok=True)

        (root / 'scripts' / 'tool.py').write_text('print("ok")\n', encoding='utf-8')
        (root / 'scripts' / 'delta_contract_migrate.py').write_text('print("migrate")\n', encoding='utf-8')
        (root / 'config' / 'migration_sync_policy.json').write_text(
            f'{json.dumps(_valid_policy(), indent=2)}\n',
            encoding='utf-8',
        )
        (root / 'config' / 'state_migration_manifest.json').write_text(
            f'{json.dumps(_valid_state_manifest(), indent=2)}\n',
            encoding='utf-8',
        )
        (root / 'config' / 'delta_contract_policy.json').write_text(
            f'{json.dumps(_valid_contract_policy(), indent=2)}\n',
            encoding='utf-8',
        )
        (root / 'extensions' / 'sample' / 'manifest.json').write_text(
            f'{json.dumps({"name": "sample", "version": "0.1.0", "capabilities": ["sync"], "entrypoints": {"doctor": "scripts/tool.py"}}, indent=2)}\n',
            encoding='utf-8',
        )

    def test_contract_check_passes_for_valid_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            self._init_repo(repo)
            self._seed(repo)

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    '--repo',
                    str(repo),
                    '--strict',
                ],
                cwd=repo,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertIn('Delta contract check OK', result.stdout)

    def test_contract_check_fails_for_invalid_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            self._init_repo(repo)
            self._seed(repo)

            invalid_policy = _valid_policy()
            invalid_policy.pop('schedule')
            (repo / 'config' / 'migration_sync_policy.json').write_text(
                f'{json.dumps(invalid_policy, indent=2)}\n',
                encoding='utf-8',
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    '--repo',
                    str(repo),
                    '--strict',
                ],
                cwd=repo,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn('schedule is required in strict mode', result.stderr)

    def test_contract_check_fails_for_missing_migration_script_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            self._init_repo(repo)
            self._seed(repo)

            invalid_contract_policy = _valid_contract_policy()
            invalid_contract_policy['targets']['state_migration_manifest']['migration_script'] = 'scripts/missing.py'
            (repo / 'config' / 'delta_contract_policy.json').write_text(
                f'{json.dumps(invalid_contract_policy, indent=2)}\n',
                encoding='utf-8',
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    '--repo',
                    str(repo),
                    '--strict',
                ],
                cwd=repo,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn('migration_script for target `state_migration_manifest`', result.stderr)


if __name__ == '__main__':
    unittest.main()
