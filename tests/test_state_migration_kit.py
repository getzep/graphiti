from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / 'scripts'
EXPORT_SCRIPT = SCRIPTS_DIR / 'state_migration_export.py'
CHECK_SCRIPT = SCRIPTS_DIR / 'state_migration_check.py'
IMPORT_SCRIPT = SCRIPTS_DIR / 'state_migration_import.py'


class StateMigrationKitTests(unittest.TestCase):
    def _init_repo(self, repo: Path) -> None:
        subprocess.run(['git', 'init'], cwd=repo, check=True, capture_output=True, text=True)
        subprocess.run(['git', 'config', 'user.email', 'test@example.com'], cwd=repo, check=True)
        subprocess.run(['git', 'config', 'user.name', 'Test User'], cwd=repo, check=True)

    def _seed_files(self, repo: Path) -> None:
        (repo / 'config').mkdir(parents=True, exist_ok=True)
        (repo / 'docs' / 'public').mkdir(parents=True, exist_ok=True)
        (repo / 'scripts').mkdir(parents=True, exist_ok=True)
        (repo / 'state').mkdir(parents=True, exist_ok=True)

        (repo / 'config' / 'public_export_allowlist.yaml').write_text('version: 1\nallowlist:\n', encoding='utf-8')
        (repo / 'config' / 'public_export_denylist.yaml').write_text('version: 1\ndenylist:\n', encoding='utf-8')
        (repo / 'config' / 'migration_sync_policy.json').write_text('{"version":1}\n', encoding='utf-8')
        (repo / 'docs' / 'public' / 'MIGRATION-SYNC-TOOLKIT.md').write_text('# toolkit\n', encoding='utf-8')
        (repo / 'scripts' / 'public_repo_boundary_audit.py').write_text('print("ok")\n', encoding='utf-8')
        (repo / 'state' / 'fact_ledger.db').write_text('fact-ledger', encoding='utf-8')
        (repo / 'state' / 'ingest_registry.db').write_text('ingest-registry', encoding='utf-8')
        (repo / 'state' / 'candidates.db').write_text('candidates', encoding='utf-8')

        manifest = {
            'version': 1,
            'package_name': 'test-state',
            'required_files': [
                'config/public_export_allowlist.yaml',
                'config/public_export_denylist.yaml',
                'config/migration_sync_policy.json',
                'config/state_migration_manifest.json',
            ],
            'optional_globs': ['docs/public/*.md', 'scripts/*.py', 'state/*.db'],
            'exclude_globs': ['**/__pycache__/**', '**/*.pyc'],
        }
        (repo / 'config' / 'state_migration_manifest.json').write_text(
            f'{json.dumps(manifest, indent=2)}\n',
            encoding='utf-8',
        )

        subprocess.run(['git', 'add', '.'], cwd=repo, check=True)
        subprocess.run(['git', 'commit', '-m', 'seed'], cwd=repo, check=True)

    def _sha256_text(self, content: str) -> str:
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def test_export_check_import_dry_run_and_full(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / 'repo'
            repo.mkdir(parents=True, exist_ok=True)
            self._init_repo(repo)
            self._seed_files(repo)

            package = repo / 'out' / 'package'

            export_preview = subprocess.run(
                [
                    sys.executable,
                    str(EXPORT_SCRIPT),
                    '--repo',
                    str(repo),
                    '--manifest',
                    'config/state_migration_manifest.json',
                    '--out',
                    str(package),
                    '--dry-run',
                    '--force',
                ],
                cwd=repo,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(export_preview.returncode, 0, msg=export_preview.stderr)

            manifest_preview = json.loads((package / 'package_manifest.json').read_text(encoding='utf-8'))
            self.assertTrue(manifest_preview['dry_run_preview'])
            self.assertGreater(manifest_preview['entry_count'], 0)

            check_preview = subprocess.run(
                [sys.executable, str(CHECK_SCRIPT), '--package', str(package), '--dry-run'],
                cwd=repo,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(check_preview.returncode, 0, msg=check_preview.stderr)

            import_preview = subprocess.run(
                [sys.executable, str(IMPORT_SCRIPT), '--in', str(package), '--dry-run'],
                cwd=repo,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(import_preview.returncode, 0, msg=import_preview.stderr)

            export_full = subprocess.run(
                [
                    sys.executable,
                    str(EXPORT_SCRIPT),
                    '--repo',
                    str(repo),
                    '--manifest',
                    'config/state_migration_manifest.json',
                    '--out',
                    str(package),
                    '--force',
                ],
                cwd=repo,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(export_full.returncode, 0, msg=export_full.stderr)

            check_full = subprocess.run(
                [sys.executable, str(CHECK_SCRIPT), '--package', str(package)],
                cwd=repo,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(check_full.returncode, 0, msg=check_full.stderr)

    def test_import_blocks_tampered_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / 'repo'
            repo.mkdir(parents=True, exist_ok=True)
            self._init_repo(repo)
            self._seed_files(repo)

            package = repo / 'out' / 'package'
            export_full = subprocess.run(
                [
                    sys.executable,
                    str(EXPORT_SCRIPT),
                    '--repo',
                    str(repo),
                    '--manifest',
                    'config/state_migration_manifest.json',
                    '--out',
                    str(package),
                    '--force',
                ],
                cwd=repo,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(export_full.returncode, 0, msg=export_full.stderr)

            manifest = json.loads((package / 'package_manifest.json').read_text(encoding='utf-8'))
            first_entry = manifest['entries'][0]['path']
            payload_file = package / 'payload' / first_entry
            payload_file.write_text('tampered\n', encoding='utf-8')

            target = repo / 'import-target'
            target.mkdir(parents=True, exist_ok=True)
            import_result = subprocess.run(
                [
                    sys.executable,
                    str(IMPORT_SCRIPT),
                    '--in',
                    str(package),
                    '--target',
                    str(target),
                ],
                cwd=repo,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(import_result.returncode, 1)
            self.assertIn('payload integrity check failed', import_result.stderr)

    def test_import_is_idempotent_when_target_matches_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / 'repo'
            repo.mkdir(parents=True, exist_ok=True)
            self._init_repo(repo)
            self._seed_files(repo)

            package = repo / 'out' / 'package'
            export = subprocess.run(
                [
                    sys.executable,
                    str(EXPORT_SCRIPT),
                    '--repo',
                    str(repo),
                    '--manifest',
                    'config/state_migration_manifest.json',
                    '--out',
                    str(package),
                    '--force',
                ],
                cwd=repo,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(export.returncode, 0, msg=export.stderr)

            target = repo / 'import-target'
            first = subprocess.run(
                [
                    sys.executable,
                    str(IMPORT_SCRIPT),
                    '--in',
                    str(package),
                    '--target',
                    str(target),
                ],
                cwd=repo,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(first.returncode, 0, msg=first.stderr)
            self.assertIn('Imported', first.stdout)

            second = subprocess.run(
                [
                    sys.executable,
                    str(IMPORT_SCRIPT),
                    '--in',
                    str(package),
                    '--target',
                    str(target),
                ],
                cwd=repo,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(second.returncode, 0, msg=second.stderr)
            self.assertIn('No changes:', second.stdout)

    def test_check_fails_on_target_manifest_version_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'source'
            source.mkdir(parents=True, exist_ok=True)
            self._init_repo(source)
            self._seed_files(source)

            package = source / 'out' / 'package'
            export = subprocess.run(
                [
                    sys.executable,
                    str(EXPORT_SCRIPT),
                    '--repo',
                    str(source),
                    '--manifest',
                    'config/state_migration_manifest.json',
                    '--out',
                    str(package),
                    '--force',
                ],
                cwd=source,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(export.returncode, 0, msg=export.stderr)

            target = Path(tmp) / 'target'
            target.mkdir(parents=True, exist_ok=True)
            self._init_repo(target)
            target_manifest = {
                'version': 2,
                'package_name': 'test-state',
                'required_files': [
                    'config/public_export_allowlist.yaml',
                    'config/public_export_denylist.yaml',
                    'config/migration_sync_policy.json',
                    'config/state_migration_manifest.json',
                ],
                'optional_globs': ['docs/public/*.md', 'scripts/*.py', 'state/*.db'],
                'exclude_globs': ['**/__pycache__/**', '**/*.pyc'],
            }
            (target / 'config').mkdir(parents=True, exist_ok=True)
            (target / 'config' / 'state_migration_manifest.json').write_text(
                f'{json.dumps(target_manifest, indent=2)}\n',
                encoding='utf-8',
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(CHECK_SCRIPT),
                    '--package',
                    str(package),
                    '--target',
                    str(target),
                    '--dry-run',
                ],
                cwd=source,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 1, msg=result.stderr)
            self.assertIn('manifest version mismatch', result.stderr)

    def test_check_fails_on_target_manifest_scope_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'source'
            source.mkdir(parents=True, exist_ok=True)
            self._init_repo(source)
            self._seed_files(source)

            package = source / 'out' / 'package'
            export = subprocess.run(
                [
                    sys.executable,
                    str(EXPORT_SCRIPT),
                    '--repo',
                    str(source),
                    '--manifest',
                    'config/state_migration_manifest.json',
                    '--out',
                    str(package),
                    '--force',
                ],
                cwd=source,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(export.returncode, 0, msg=export.stderr)

            target = Path(tmp) / 'target'
            target.mkdir(parents=True, exist_ok=True)
            self._init_repo(target)
            target_manifest = {
                'version': 1,
                'package_name': 'test-state',
                'required_files': [
                    'config/public_export_allowlist.yaml',
                    'config/public_export_denylist.yaml',
                    'config/migration_sync_policy.json',
                    'config/state_migration_manifest.json',
                ],
                'optional_globs': ['docs/public/*.md', 'scripts/*.py'],
                'exclude_globs': ['**/*.pyc'],
            }
            (target / 'config').mkdir(parents=True, exist_ok=True)
            (target / 'config' / 'state_migration_manifest.json').write_text(
                f'{json.dumps(target_manifest, indent=2)}\n',
                encoding='utf-8',
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(CHECK_SCRIPT),
                    '--package',
                    str(package),
                    '--target',
                    str(target),
                    '--dry-run',
                ],
                cwd=source,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 1, msg=result.stderr)
            self.assertIn('optional_globs mismatch', result.stderr)
            self.assertIn('exclude_globs mismatch', result.stderr)

    def test_import_dry_run_fails_when_overwrite_conflicts_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / 'repo'
            repo.mkdir(parents=True, exist_ok=True)
            self._init_repo(repo)
            self._seed_files(repo)

            package = repo / 'out' / 'package'
            export = subprocess.run(
                [
                    sys.executable,
                    str(EXPORT_SCRIPT),
                    '--repo',
                    str(repo),
                    '--manifest',
                    'config/state_migration_manifest.json',
                    '--out',
                    str(package),
                    '--force',
                ],
                cwd=repo,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(export.returncode, 0, msg=export.stderr)

            target = repo / 'import-target'
            target.mkdir(parents=True, exist_ok=True)
            (target / 'config').mkdir(parents=True, exist_ok=True)
            (target / 'config' / 'public_export_allowlist.yaml').write_text(
                'different\n',
                encoding='utf-8',
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(IMPORT_SCRIPT),
                    '--in',
                    str(package),
                    '--target',
                    str(target),
                    '--dry-run',
                ],
                cwd=repo,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 1, msg=result.stderr)
            self.assertIn('Dry-run detected blocking destination conflicts', result.stderr)

    def test_atomic_import_rolls_back_on_write_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = root / 'pkg'
            payload = package / 'payload'
            payload.mkdir(parents=True, exist_ok=True)

            first_rel = 'a/ok.txt'
            second_rel = 'b/fail.txt'
            first_content = 'first-payload\n'
            second_content = 'second-payload\n'

            first_src = payload / first_rel
            second_src = payload / second_rel
            first_src.parent.mkdir(parents=True, exist_ok=True)
            second_src.parent.mkdir(parents=True, exist_ok=True)
            first_src.write_text(first_content, encoding='utf-8')
            second_src.write_text(second_content, encoding='utf-8')

            manifest = {
                'package_version': 1,
                'manifest_version': 1,
                'package_name': 'rollback-test',
                'created_at': '2026-02-15T00:00:00Z',
                'source_repo': '/tmp/source',
                'source_commit': 'abc123',
                'dry_run_preview': False,
                'entry_count': 2,
                'entries': [
                    {
                        'path': first_rel,
                        'sha256': self._sha256_text(first_content),
                        'size_bytes': len(first_content.encode('utf-8')),
                    },
                    {
                        'path': second_rel,
                        'sha256': self._sha256_text(second_content),
                        'size_bytes': len(second_content.encode('utf-8')),
                    },
                ],
            }
            (package / 'package_manifest.json').write_text(
                f'{json.dumps(manifest, indent=2)}\n',
                encoding='utf-8',
            )

            target = root / 'target'
            target.mkdir(parents=True, exist_ok=True)
            (target / 'b').write_text('not-a-directory\n', encoding='utf-8')

            result = subprocess.run(
                [
                    sys.executable,
                    str(IMPORT_SCRIPT),
                    '--in',
                    str(package),
                    '--target',
                    str(target),
                    '--allow-overwrite',
                ],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 1)
            self.assertFalse((target / first_rel).exists(), msg='atomic rollback should remove partial writes')
            self.assertTrue((target / 'b').is_file(), msg='original conflicting path should remain intact')

    def test_import_rejects_traversal_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = root / 'pkg'
            package.mkdir(parents=True, exist_ok=True)
            payload = package / 'payload'
            payload.mkdir(parents=True, exist_ok=True)

            manifest = {
                'package_version': 1,
                'manifest_version': 1,
                'package_name': 'unsafe',
                'created_at': '2026-02-15T00:00:00Z',
                'source_repo': '/tmp/source',
                'source_commit': 'abc123',
                'dry_run_preview': False,
                'entry_count': 1,
                'entries': [
                    {
                        'path': '../escape.txt',
                        'sha256': '0' * 64,
                        'size_bytes': 0,
                    },
                ],
            }
            (package / 'package_manifest.json').write_text(
                f'{json.dumps(manifest, indent=2)}\n',
                encoding='utf-8',
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(IMPORT_SCRIPT),
                    '--in',
                    str(package),
                    '--target',
                    str(root / 'target'),
                    '--dry-run',
                ],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn('Unsafe package entry path', result.stderr)

    def test_check_rejects_duplicate_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = root / 'pkg'
            package.mkdir(parents=True, exist_ok=True)
            payload = package / 'payload'
            payload.mkdir(parents=True, exist_ok=True)

            manifest = {
                'package_version': 1,
                'manifest_version': 1,
                'package_name': 'duplicate-test',
                'created_at': '2026-02-15T00:00:00Z',
                'source_repo': '/tmp/source',
                'source_commit': 'abc123',
                'dry_run_preview': False,
                'entry_count': 3,
                'entries': [
                    {
                        'path': 'a/file.txt',
                        'sha256': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
                        'size_bytes': 0,
                    },
                    {
                        'path': 'a/file.txt',
                        'sha256': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
                        'size_bytes': 0,
                    },
                    {
                        'path': 'b/file.txt',
                        'sha256': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
                        'size_bytes': 0,
                    },
                ],
            }
            payload.joinpath('a/file.txt').parent.mkdir(parents=True, exist_ok=True)
            payload.joinpath('a/file.txt').write_text('', encoding='utf-8')
            payload.joinpath('b/file.txt').parent.mkdir(parents=True, exist_ok=True)
            payload.joinpath('b/file.txt').write_text('', encoding='utf-8')
            (package / 'package_manifest.json').write_text(
                f'{json.dumps(manifest, indent=2)}\n',
                encoding='utf-8',
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(CHECK_SCRIPT),
                    '--package',
                    str(package),
                ],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 2, msg=result.stderr)
            self.assertIn('duplicate', result.stderr)


if __name__ == '__main__':
    unittest.main()
