from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'delta_tool.py'


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


class DeltaToolTests(unittest.TestCase):
    def _init_repo(self, root: Path) -> None:
        subprocess.run(['git', 'init'], cwd=root, check=True, capture_output=True, text=True)
        subprocess.run(['git', 'config', 'user.email', 'test@example.com'], cwd=root, check=True)
        subprocess.run(['git', 'config', 'user.name', 'Test User'], cwd=root, check=True)

    def _seed_repo(self, repo: Path) -> None:
        (repo / 'config').mkdir(parents=True, exist_ok=True)
        (repo / 'extensions' / 'sample').mkdir(parents=True, exist_ok=True)
        (repo / 'scripts').mkdir(parents=True, exist_ok=True)

        (repo / 'scripts' / 'tool.py').write_text('print("ok")\n', encoding='utf-8')
        (repo / 'scripts' / 'delta_contract_migrate.py').write_text('print("migrate")\n', encoding='utf-8')
        (repo / 'config' / 'migration_sync_policy.json').write_text(
            f'{json.dumps(_valid_policy(), indent=2)}\n',
            encoding='utf-8',
        )
        (repo / 'config' / 'state_migration_manifest.json').write_text(
            f'{json.dumps(_valid_state_manifest(), indent=2)}\n',
            encoding='utf-8',
        )
        (repo / 'config' / 'delta_contract_policy.json').write_text(
            f'{json.dumps(_valid_contract_policy(), indent=2)}\n',
            encoding='utf-8',
        )
        (repo / 'extensions' / 'sample' / 'manifest.json').write_text(
            (
                f'{json.dumps({"name": "sample", "version": "0.1.0", "capabilities": ["sync"], "entrypoints": {"doctor": "scripts/tool.py"}, "command_contract": {"version": 1, "namespace": "sample"}, "commands": {"sample/tool": "scripts/tool.py"}}, indent=2)}\n'
            ),
            encoding='utf-8',
        )

    def test_dispatches_contract_check_subcommand(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            self._init_repo(repo)
            self._seed_repo(repo)

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    'contracts-check',
                    '--',
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

    def test_loads_extension_commands_and_lists_registry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            self._init_repo(repo)
            self._seed_repo(repo)

            list_result = subprocess.run(
                [sys.executable, str(SCRIPT), 'list-commands'],
                cwd=repo,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(list_result.returncode, 0, msg=list_result.stderr)
            self.assertIn('sample/tool', list_result.stdout)

            run_result = subprocess.run(
                [sys.executable, str(SCRIPT), 'sample/tool'],
                cwd=repo,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(run_result.returncode, 0, msg=run_result.stderr)
            self.assertIn('ok', run_result.stdout)

    def test_refuses_execution_when_registry_warnings_exist_without_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            self._init_repo(repo)
            self._seed_repo(repo)
            (repo / 'extensions' / 'broken').mkdir(parents=True, exist_ok=True)

            blocked = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    'contracts-migrate',
                    '--',
                    '--repo',
                    str(repo),
                    '--contract-policy',
                    'config/delta_contract_policy.json',
                ],
                cwd=repo,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(blocked.returncode, 1)
            self.assertIn('refusing to execute commands while extension registry warnings exist', blocked.stderr)

            allowed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    '--allow-registry-warnings',
                    'contracts-migrate',
                    '--',
                    '--repo',
                    str(repo),
                    '--contract-policy',
                    'config/delta_contract_policy.json',
                ],
                cwd=repo,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(allowed.returncode, 0, msg=allowed.stderr)
            self.assertIn('Delta contract migrate', allowed.stdout)


if __name__ == '__main__':
    unittest.main()
