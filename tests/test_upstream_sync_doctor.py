from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'upstream_sync_doctor.py'


class UpstreamSyncDoctorTests(unittest.TestCase):
    def _git(self, cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ['git', *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        )

    def _init_repo(self, repo: Path) -> None:
        self._git(repo, 'init')
        self._git(repo, 'config', 'user.email', 'test@example.com')
        self._git(repo, 'config', 'user.name', 'Test User')
        (repo / 'README.md').write_text('# repo\n', encoding='utf-8')
        self._git(repo, 'add', 'README.md')
        self._git(repo, 'commit', '-m', 'initial')
        self._git(repo, 'branch', '-M', 'main')

    def _write_policy(self, root: Path) -> Path:
        policy = {
            'version': 1,
            'origin': {'remote': 'origin', 'branch': 'main'},
            'upstream': {'remote': 'upstream', 'branch': 'main', 'url': ''},
            'sync_button_policy': {
                'require_clean_worktree': True,
                'max_origin_only_commits': 0,
                'require_upstream_only_commits': True,
            },
        }
        path = root / 'migration_sync_policy.json'
        path.write_text(f'{json.dumps(policy, indent=2)}\n', encoding='utf-8')
        return path

    def test_sync_button_safe_when_upstream_ahead_and_origin_not_ahead(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / 'repo'
            repo.mkdir()
            self._init_repo(repo)

            origin_bare = root / 'origin.git'
            upstream_bare = root / 'upstream.git'
            self._git(root, 'init', '--bare', str(origin_bare))
            self._git(root, 'init', '--bare', str(upstream_bare))

            self._git(repo, 'remote', 'add', 'origin', str(origin_bare))
            self._git(repo, 'remote', 'add', 'upstream', str(upstream_bare))
            self._git(repo, 'push', '-u', 'origin', 'main')
            self._git(repo, 'push', '-u', 'upstream', 'main')

            upstream_work = root / 'upstream-work'
            self._git(root, 'clone', str(upstream_bare), str(upstream_work))
            self._git(upstream_work, 'config', 'user.email', 'test@example.com')
            self._git(upstream_work, 'config', 'user.name', 'Test User')
            self._git(upstream_work, 'checkout', '-B', 'main', 'origin/main')
            (upstream_work / 'UPSTREAM.md').write_text('new\n', encoding='utf-8')
            self._git(upstream_work, 'add', 'UPSTREAM.md')
            self._git(upstream_work, 'commit', '-m', 'upstream-only change')
            self._git(upstream_work, 'push', 'origin', 'HEAD:main')

            policy_path = self._write_policy(root)
            output_json = root / 'doctor.json'

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    '--repo',
                    str(repo),
                    '--policy',
                    str(policy_path),
                    '--fetch',
                    '--check-sync-button-safety',
                    '--output-json',
                    str(output_json),
                ],
                cwd=repo,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)

            report = json.loads(output_json.read_text(encoding='utf-8'))
            self.assertTrue(report['sync_button_safe'])
            self.assertEqual(report['origin_only_commits'], 0)
            self.assertGreater(report['upstream_only_commits'], 0)

    def test_allow_missing_upstream_can_warn_without_failing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / 'repo'
            repo.mkdir()
            self._init_repo(repo)

            origin_bare = Path(tmp) / 'origin.git'
            self._git(Path(tmp), 'init', '--bare', str(origin_bare))
            self._git(repo, 'remote', 'add', 'origin', str(origin_bare))
            self._git(repo, 'push', '-u', 'origin', 'main')

            policy_path = self._write_policy(Path(tmp))
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    '--repo',
                    str(repo),
                    '--policy',
                    str(policy_path),
                    '--allow-missing-upstream',
                ],
                cwd=repo,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertIn('Missing `upstream` remote', result.stdout)


    def test_sync_button_safety_degrades_to_warning_when_allow_missing_upstream(self) -> None:
        """Must-fix: --allow-missing-upstream + --check-sync-button-safety should warn, not fail."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / 'repo'
            repo.mkdir()
            self._init_repo(repo)

            origin_bare = Path(tmp) / 'origin.git'
            self._git(Path(tmp), 'init', '--bare', str(origin_bare))
            self._git(repo, 'remote', 'add', 'origin', str(origin_bare))
            self._git(repo, 'push', '-u', 'origin', 'main')

            policy_path = self._write_policy(Path(tmp))
            output_json = Path(tmp) / 'doctor.json'
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    '--repo',
                    str(repo),
                    '--policy',
                    str(policy_path),
                    '--allow-missing-upstream',
                    '--check-sync-button-safety',
                    '--output-json',
                    str(output_json),
                ],
                cwd=repo,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, msg=f'stdout={result.stdout}\nstderr={result.stderr}')
            report = json.loads(output_json.read_text(encoding='utf-8'))
            self.assertFalse(report['sync_button_safe'])
            self.assertTrue(
                any('skipped' in w for w in report['warnings']),
                msg=f'Expected degraded warning in: {report["warnings"]}',
            )

    def test_sync_button_safety_still_fails_without_allow_missing_upstream(self) -> None:
        """Preserve: without --allow-missing-upstream, missing upstream is a hard failure."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / 'repo'
            repo.mkdir()
            self._init_repo(repo)

            origin_bare = Path(tmp) / 'origin.git'
            self._git(Path(tmp), 'init', '--bare', str(origin_bare))
            self._git(repo, 'remote', 'add', 'origin', str(origin_bare))
            self._git(repo, 'push', '-u', 'origin', 'main')

            policy_path = self._write_policy(Path(tmp))
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    '--repo',
                    str(repo),
                    '--policy',
                    str(policy_path),
                    '--check-sync-button-safety',
                ],
                cwd=repo,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 1, msg=f'Expected failure.\nstdout={result.stdout}')


if __name__ == '__main__':
    unittest.main()
