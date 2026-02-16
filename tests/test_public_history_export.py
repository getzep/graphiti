from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'public_history_export.py'


class PublicHistoryExportTests(unittest.TestCase):
    def _init_repo(self, repo: Path) -> None:
        subprocess.run(['git', 'init'], cwd=repo, check=True, capture_output=True, text=True)
        subprocess.run(['git', 'config', 'user.email', 'test@example.com'], cwd=repo, check=True)
        subprocess.run(['git', 'config', 'user.name', 'Test User'], cwd=repo, check=True)
        subprocess.run(['git', 'branch', '-M', 'main'], cwd=repo, check=True)

    def test_generates_reports_for_both_modes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / 'repo'
            repo.mkdir()
            self._init_repo(repo)

            (repo / 'config').mkdir(parents=True, exist_ok=True)
            (repo / 'docs' / 'public').mkdir(parents=True, exist_ok=True)
            (repo / 'docs' / 'public' / 'x.md').write_text('ok\n', encoding='utf-8')
            (repo / 'secret.key').write_text('nope\n', encoding='utf-8')
            (repo / 'config' / 'public_export_allowlist.yaml').write_text(
                'version: 1\n\nallowlist:\n  - "docs/public/**"\n  - "config/public_export_*.yaml"\n',
                encoding='utf-8',
            )
            (repo / 'config' / 'public_export_denylist.yaml').write_text(
                'version: 1\n\ndenylist:\n  - "*.key"\n',
                encoding='utf-8',
            )

            subprocess.run(['git', 'add', '.'], cwd=repo, check=True)
            subprocess.run(['git', 'commit', '-m', 'seed'], cwd=repo, check=True)

            filtered_json = repo / 'filtered.json'
            clean_json = repo / 'clean.json'

            filtered = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    '--repo',
                    str(repo),
                    '--mode',
                    'filtered-history',
                    '--manifest',
                    'config/public_export_allowlist.yaml',
                    '--denylist',
                    'config/public_export_denylist.yaml',
                    '--report',
                    str(repo / 'filtered.md'),
                    '--summary-json',
                    str(filtered_json),
                    '--dry-run',
                ],
                cwd=repo,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(filtered.returncode, 0, msg=filtered.stderr)

            clean = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    '--repo',
                    str(repo),
                    '--mode',
                    'clean-foundation',
                    '--manifest',
                    'config/public_export_allowlist.yaml',
                    '--denylist',
                    'config/public_export_denylist.yaml',
                    '--report',
                    str(repo / 'clean.md'),
                    '--summary-json',
                    str(clean_json),
                    '--dry-run',
                ],
                cwd=repo,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(clean.returncode, 0, msg=clean.stderr)

            filtered_payload = json.loads(filtered_json.read_text(encoding='utf-8'))
            clean_payload = json.loads(clean_json.read_text(encoding='utf-8'))

            self.assertEqual(filtered_payload['mode'], 'filtered-history')
            self.assertEqual(clean_payload['mode'], 'clean-foundation')
            self.assertIn('metrics', filtered_payload)
            self.assertIn('metrics', clean_payload)

    def test_uses_policy_overrides_for_metric_coefficients(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / 'repo'
            repo.mkdir()
            self._init_repo(repo)

            (repo / 'config').mkdir(parents=True, exist_ok=True)
            (repo / 'docs' / 'public').mkdir(parents=True, exist_ok=True)
            (repo / 'docs' / 'public' / 'x.md').write_text('ok\n', encoding='utf-8')
            (repo / 'config' / 'public_export_allowlist.yaml').write_text(
                'version: 1\n\nallowlist:\n  - "docs/public/**"\n  - "config/public_export_*.yaml"\n',
                encoding='utf-8',
            )
            (repo / 'config' / 'public_export_denylist.yaml').write_text(
                'version: 1\n\ndenylist:\n',
                encoding='utf-8',
            )
            (repo / 'config' / 'migration_sync_policy.json').write_text(
                json.dumps(
                    {
                        'version': 1,
                        'history_metrics': {
                            'filtered_history': {
                                'privacy_risk': {
                                    'base': 40,
                                    'block_penalty': 0,
                                    'ambiguous_penalty': 0,
                                },
                            },
                        },
                    },
                    indent=2,
                )
                + '\n',
                encoding='utf-8',
            )

            subprocess.run(['git', 'add', '.'], cwd=repo, check=True)
            subprocess.run(['git', 'commit', '-m', 'seed'], cwd=repo, check=True)

            filtered_json = repo / 'filtered.json'
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    '--repo',
                    str(repo),
                    '--mode',
                    'filtered-history',
                    '--manifest',
                    'config/public_export_allowlist.yaml',
                    '--denylist',
                    'config/public_export_denylist.yaml',
                    '--policy',
                    'config/migration_sync_policy.json',
                    '--report',
                    str(repo / 'filtered.md'),
                    '--summary-json',
                    str(filtered_json),
                ],
                cwd=repo,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)

            payload = json.loads(filtered_json.read_text(encoding='utf-8'))
            self.assertEqual(payload['metrics']['privacy_risk'], 40)


if __name__ == '__main__':
    unittest.main()
