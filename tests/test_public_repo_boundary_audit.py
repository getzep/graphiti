from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / 'scripts' / 'public_repo_boundary_audit.py'


class BoundaryAuditStrictModeTests(unittest.TestCase):
    def _init_repo(self, repo: Path) -> None:
        subprocess.run(['git', 'init'], cwd=repo, check=True, capture_output=True, text=True)

    def _write_policy(self, repo: Path, allowlist_extra: list[str] | None = None) -> None:
        allowlist_extra = allowlist_extra or []
        (repo / 'config').mkdir(parents=True, exist_ok=True)
        allow_rules = [
            'allowed.txt',
            'config/public_export_allowlist.yaml',
            'config/public_export_denylist.yaml',
        ] + allowlist_extra
        (repo / 'config' / 'public_export_allowlist.yaml').write_text(
            'version: 1\n\nallowlist:\n' + ''.join(f'  - "{rule}"\n' for rule in allow_rules),
            encoding='utf-8',
        )
        (repo / 'config' / 'public_export_denylist.yaml').write_text(
            'version: 1\n\ndenylist:\n',
            encoding='utf-8',
        )

    def _run(self, repo: Path, strict: bool) -> subprocess.CompletedProcess[str]:
        cmd = [
            sys.executable,
            str(SCRIPT_PATH),
            '--manifest',
            'config/public_export_allowlist.yaml',
            '--denylist',
            'config/public_export_denylist.yaml',
            '--report',
            'report.md',
        ]
        if strict:
            cmd.append('--strict')

        return subprocess.run(cmd, cwd=repo, capture_output=True, text=True, check=False)

    def test_strict_returns_zero_when_all_files_allowlisted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / 'repo_ok'
            repo.mkdir()
            self._init_repo(repo)

            (repo / 'allowed.txt').write_text('ok\n', encoding='utf-8')
            self._write_policy(repo)

            subprocess.run(['git', 'add', '.'], cwd=repo, check=True)

            result = self._run(repo, strict=True)
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertTrue((repo / 'report.md').exists())

    def test_strict_returns_nonzero_when_ambiguous_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / 'repo_ambiguous'
            repo.mkdir()
            self._init_repo(repo)

            (repo / 'allowed.txt').write_text('ok\n', encoding='utf-8')
            (repo / 'extra.txt').write_text('ambiguous\n', encoding='utf-8')
            self._write_policy(repo)

            subprocess.run(['git', 'add', '.'], cwd=repo, check=True)

            result = self._run(repo, strict=True)
            self.assertEqual(result.returncode, 1)
            report = (repo / 'report.md').read_text(encoding='utf-8')
            self.assertIn('AMBIGUOUS', report)

    def test_non_strict_returns_zero_with_ambiguous(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / 'repo_non_strict'
            repo.mkdir()
            self._init_repo(repo)

            (repo / 'allowed.txt').write_text('ok\n', encoding='utf-8')
            (repo / 'extra.txt').write_text('ambiguous\n', encoding='utf-8')
            self._write_policy(repo)

            subprocess.run(['git', 'add', '.'], cwd=repo, check=True)

            result = self._run(repo, strict=False)
            self.assertEqual(result.returncode, 0)
            self.assertTrue((repo / 'report.md').exists())


if __name__ == '__main__':
    unittest.main()
