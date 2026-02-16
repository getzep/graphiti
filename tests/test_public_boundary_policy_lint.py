from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

LINT_SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'public_boundary_policy_lint.py'


class BoundaryPolicyLintTests(unittest.TestCase):
    def _init_repo(self, repo: Path) -> None:
        subprocess.run(['git', 'init'], cwd=repo, check=True, capture_output=True, text=True)

    def _write_policies(self, repo: Path, allowlist: list[str], denylist: list[str]) -> None:
        (repo / 'config').mkdir(parents=True, exist_ok=True)
        (repo / 'config' / 'public_export_allowlist.yaml').write_text(
            'version: 1\n\nallowlist:\n' + ''.join(f'  - "{rule}"\n' for rule in allowlist),
            encoding='utf-8',
        )
        (repo / 'config' / 'public_export_denylist.yaml').write_text(
            'version: 1\n\ndenylist:\n' + ''.join(f'  - "{rule}"\n' for rule in denylist),
            encoding='utf-8',
        )

    def _run_lint(self, repo: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(LINT_SCRIPT),
                '--manifest',
                'config/public_export_allowlist.yaml',
                '--denylist',
                'config/public_export_denylist.yaml',
            ],
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_lint_passes_with_clean_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            self._init_repo(repo)
            self._write_policies(repo, allowlist=['docs/public/**'], denylist=['.env*'])
            result = self._run_lint(repo)
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertIn('OK', result.stdout)

    def test_lint_fails_on_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            self._init_repo(repo)
            self._write_policies(repo, allowlist=['docs/public/**', 'docs/public/**'], denylist=['.env*'])
            result = self._run_lint(repo)
            self.assertEqual(result.returncode, 1)
            self.assertIn('duplicates', result.stderr)

    def test_lint_fails_on_contradictory_rule(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            self._init_repo(repo)
            self._write_policies(repo, allowlist=['docs/**'], denylist=['docs/**'])
            result = self._run_lint(repo)
            self.assertEqual(result.returncode, 1)
            self.assertIn('both allowlist and denylist', result.stderr)


if __name__ == '__main__':
    unittest.main()
