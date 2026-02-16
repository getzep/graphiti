from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

DELTA_TOOL = Path(__file__).resolve().parents[1] / 'scripts' / 'delta_tool.py'


class DeltaPipelineE2ETests(unittest.TestCase):
    def _git(self, cwd: Path, *args: str) -> None:
        subprocess.run(['git', *args], cwd=cwd, check=True, capture_output=True, text=True)

    def _run_delta(self, cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(DELTA_TOOL), *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )

    def _seed_repo(self, repo: Path) -> None:
        self._git(repo, 'init')
        self._git(repo, 'config', 'user.email', 'test@example.com')
        self._git(repo, 'config', 'user.name', 'Test User')

        (repo / 'config').mkdir(parents=True, exist_ok=True)
        (repo / 'docs' / 'public').mkdir(parents=True, exist_ok=True)
        (repo / 'scripts').mkdir(parents=True, exist_ok=True)
        (repo / 'extensions' / 'demo').mkdir(parents=True, exist_ok=True)

        (repo / 'config' / 'public_export_allowlist.yaml').write_text(
            'version: 1\n\nallowlist:\n  - "docs/public/**"\n  - "config/public_export_*.yaml"\n',
            encoding='utf-8',
        )
        (repo / 'config' / 'public_export_denylist.yaml').write_text(
            'version: 1\n\ndenylist:\n  - "*.key"\n',
            encoding='utf-8',
        )

        policy = {
            'version': 1,
            'origin': {'remote': 'origin', 'branch': 'main'},
            'upstream': {
                'remote': 'upstream',
                'url': 'https://example.com/upstream.git',
                'branch': 'main',
            },
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
            'history_metrics': {
                'filtered_history': {
                    'privacy_risk': {'base': 100, 'block_penalty': 35, 'ambiguous_penalty': 0.5},
                    'simplicity': {'base': 100, 'commit_divisor': 15, 'commit_cap': 35, 'ambiguous_penalty': 0.3},
                    'merge_conflict_risk': {
                        'base': 100,
                        'commit_divisor': 20,
                        'commit_cap': 30,
                        'ambiguous_penalty': 0.2,
                    },
                    'auditability': {'base': 100, 'block_penalty': 20, 'ambiguous_penalty': 0.4},
                },
                'clean_foundation': {
                    'privacy_risk': {'base': 97},
                    'simplicity': {'base': 90, 'commit_bonus_divisor': 100, 'commit_bonus_cap': 6},
                    'merge_conflict_risk': {'base': 92},
                    'auditability': {'base': 90},
                },
            },
        }
        (repo / 'config' / 'migration_sync_policy.json').write_text(
            f'{json.dumps(policy, indent=2)}\n',
            encoding='utf-8',
        )

        contract_policy = {
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
        (repo / 'config' / 'delta_contract_policy.json').write_text(
            f'{json.dumps(contract_policy, indent=2)}\n',
            encoding='utf-8',
        )

        state_manifest = {
            'version': 1,
            'package_name': 'delta-state',
            'required_files': [
                'config/public_export_allowlist.yaml',
                'config/public_export_denylist.yaml',
                'config/migration_sync_policy.json',
                'config/state_migration_manifest.json',
                'config/delta_contract_policy.json',
            ],
            'optional_globs': ['docs/public/*.md', 'scripts/*.py', 'extensions/**'],
            'exclude_globs': ['**/__pycache__/**', '**/*.pyc'],
        }
        (repo / 'config' / 'state_migration_manifest.json').write_text(
            f'{json.dumps(state_manifest, indent=2)}\n',
            encoding='utf-8',
        )

        (repo / 'docs' / 'public' / 'MIGRATION-SYNC-TOOLKIT.md').write_text('# toolkit\n', encoding='utf-8')
        (repo / 'scripts' / 'tool.py').write_text('print("extension-command-ok")\n', encoding='utf-8')
        (repo / 'scripts' / 'delta_contract_migrate.py').write_text('print("migrate")\n', encoding='utf-8')

        extension_manifest = {
            'name': 'demo-extension',
            'version': '0.1.0',
            'capabilities': ['demo'],
            'entrypoints': {'run': 'scripts/tool.py'},
            'command_contract': {'version': 1, 'namespace': 'demo-extension'},
            'commands': {'demo-extension/tool-run': 'scripts/tool.py'},
        }
        (repo / 'extensions' / 'demo' / 'manifest.json').write_text(
            f'{json.dumps(extension_manifest, indent=2)}\n',
            encoding='utf-8',
        )

        self._git(repo, 'add', '.')
        self._git(repo, 'commit', '-m', 'seed delta pipeline test repo')
        self._git(repo, 'branch', '-M', 'main')

    def test_end_to_end_delta_pipeline_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / 'repo'
            repo.mkdir(parents=True, exist_ok=True)
            self._seed_repo(repo)

            out_dir = root / 'artifacts'
            out_dir.mkdir(parents=True, exist_ok=True)
            package_dir = out_dir / 'state-package'

            contracts = self._run_delta(
                repo,
                'contracts-check',
                '--',
                '--repo',
                str(repo),
                '--strict',
            )
            self.assertEqual(contracts.returncode, 0, msg=contracts.stderr)

            sync_doctor = self._run_delta(
                repo,
                'sync-doctor',
                '--',
                '--repo',
                str(repo),
                '--policy',
                'config/migration_sync_policy.json',
                '--dry-run',
                '--allow-missing-upstream',
            )
            self.assertEqual(sync_doctor.returncode, 0, msg=sync_doctor.stderr)

            filtered_json = out_dir / 'filtered-history.json'
            clean_json = out_dir / 'clean-foundation.json'

            filtered = self._run_delta(
                repo,
                'history-export',
                '--',
                '--repo',
                str(repo),
                '--mode',
                'filtered-history',
                '--manifest',
                'config/public_export_allowlist.yaml',
                '--denylist',
                'config/public_export_denylist.yaml',
                '--report',
                str(out_dir / 'filtered-history.md'),
                '--summary-json',
                str(filtered_json),
                '--dry-run',
            )
            self.assertEqual(filtered.returncode, 0, msg=filtered.stderr)

            clean = self._run_delta(
                repo,
                'history-export',
                '--',
                '--repo',
                str(repo),
                '--mode',
                'clean-foundation',
                '--manifest',
                'config/public_export_allowlist.yaml',
                '--denylist',
                'config/public_export_denylist.yaml',
                '--report',
                str(out_dir / 'clean-foundation.md'),
                '--summary-json',
                str(clean_json),
                '--dry-run',
            )
            self.assertEqual(clean.returncode, 0, msg=clean.stderr)

            scorecard = self._run_delta(
                repo,
                'history-scorecard',
                '--',
                '--filtered-summary',
                str(filtered_json),
                '--clean-summary',
                str(clean_json),
                '--policy',
                'config/migration_sync_policy.json',
                '--out',
                str(out_dir / 'history-scorecard.md'),
                '--summary-json',
                str(out_dir / 'history-scorecard.json'),
            )
            self.assertEqual(scorecard.returncode, 0, msg=scorecard.stderr)

            export_result = self._run_delta(
                repo,
                'state-export',
                '--',
                '--repo',
                str(repo),
                '--manifest',
                'config/state_migration_manifest.json',
                '--out',
                str(package_dir),
                '--dry-run',
                '--force',
            )
            self.assertEqual(export_result.returncode, 0, msg=export_result.stderr)

            check_result = self._run_delta(
                repo,
                'state-check',
                '--',
                '--package',
                str(package_dir),
                '--dry-run',
            )
            self.assertEqual(check_result.returncode, 0, msg=check_result.stderr)

            import_result = self._run_delta(
                repo,
                'state-import',
                '--',
                '--in',
                str(package_dir),
                '--target',
                str(repo),
                '--dry-run',
            )
            self.assertEqual(import_result.returncode, 0, msg=import_result.stderr)

            extension_check = self._run_delta(
                repo,
                'extension-check',
                '--',
                '--repo',
                str(repo),
                '--extensions-dir',
                'extensions',
                '--strict',
            )
            self.assertEqual(extension_check.returncode, 0, msg=extension_check.stderr)

            extension_command = self._run_delta(repo, 'demo-extension/tool-run')
            self.assertEqual(extension_command.returncode, 0, msg=extension_command.stderr)
            self.assertIn('extension-command-ok', extension_command.stdout)


if __name__ == '__main__':
    unittest.main()
