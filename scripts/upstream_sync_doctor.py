#!/usr/bin/env python3
"""Preflight checks for upstream sync lane safety and maintainability."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from delta_contracts import validate_migration_sync_policy
from migration_sync_lib import dump_json, load_json, resolve_repo_root, run_git


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Validate upstream sync preconditions.')
    parser.add_argument('--repo', type=Path, default=Path('.'), help='Repository path')
    parser.add_argument(
        '--policy',
        type=Path,
        default=Path('config/migration_sync_policy.json'),
        help='Sync policy JSON path',
    )
    parser.add_argument('--dry-run', action='store_true', help='Do not mutate remotes even when ensure-upstream is set')
    parser.add_argument('--fetch', action='store_true', help='Fetch origin/upstream refs before divergence checks')
    parser.add_argument('--allow-dirty', action='store_true', help='Warn instead of fail for dirty working tree')
    parser.add_argument(
        '--check-sync-button-safety',
        action='store_true',
        help='Fail when GitHub sync-fork button policy requirements are not met',
    )
    parser.add_argument('--allow-missing-upstream', action='store_true', help='Warn instead of fail if upstream remote is absent')
    parser.add_argument('--ensure-upstream', action='store_true', help='Add upstream remote from policy when missing')
    parser.add_argument('--output-json', type=Path, help='Optional JSON report output path')
    return parser.parse_args()


def _git_ref_exists(repo_root: Path, ref_name: str) -> bool:
    result = run_git(repo_root, 'rev-parse', '--verify', ref_name, check=False)
    return result.returncode == 0


def main() -> int:
    args = parse_args()
    repo_root = resolve_repo_root(args.repo.resolve())
    policy_path = args.policy if args.policy.is_absolute() else (repo_root / args.policy).resolve()
    policy = validate_migration_sync_policy(load_json(policy_path), context=str(policy_path))

    upstream_cfg = policy.get('upstream', {})
    origin_cfg = policy.get('origin', {})
    sync_cfg = policy.get('sync_button_policy', {})

    upstream_remote = str(upstream_cfg.get('remote', 'upstream'))
    upstream_url = str(upstream_cfg.get('url', '')).strip()
    upstream_branch = str(upstream_cfg.get('branch', 'main'))

    origin_remote = str(origin_cfg.get('remote', 'origin'))
    origin_branch = str(origin_cfg.get('branch', 'main'))

    max_origin_only = int(sync_cfg.get('max_origin_only_commits', 0))
    require_upstream_only = bool(sync_cfg.get('require_upstream_only_commits', True))
    require_clean_worktree = bool(sync_cfg.get('require_clean_worktree', True))

    issues: list[str] = []
    warnings: list[str] = []

    remotes_output = run_git(repo_root, 'remote', check=False)
    if remotes_output.returncode != 0:
        raise ValueError(f'Failed to list git remotes: {remotes_output.stderr.strip()}')
    remotes = {line.strip() for line in remotes_output.stdout.splitlines() if line.strip()}

    if upstream_remote not in remotes and args.ensure_upstream:
        if not upstream_url:
            issues.append(f'Cannot add `{upstream_remote}` remote because policy has no upstream.url')
        elif args.dry_run:
            warnings.append(f'DRY RUN: would add remote `{upstream_remote}` -> {upstream_url}')
        else:
            add_result = run_git(repo_root, 'remote', 'add', upstream_remote, upstream_url, check=False)
            if add_result.returncode != 0:
                issues.append(f'Failed to add `{upstream_remote}` remote: {add_result.stderr.strip()}')
            else:
                remotes.add(upstream_remote)

    if upstream_remote not in remotes:
        message = f'Missing `{upstream_remote}` remote'
        if args.allow_missing_upstream:
            warnings.append(message)
        else:
            issues.append(message)

    status_output = run_git(repo_root, 'status', '--porcelain', check=False)
    clean_worktree = status_output.returncode == 0 and not status_output.stdout.strip()

    if require_clean_worktree and not clean_worktree:
        if args.allow_dirty:
            warnings.append('Working tree is dirty (allowed by --allow-dirty override)')
        else:
            issues.append('Working tree is dirty (sync preflight requires a clean tree)')

    if args.fetch:
        for remote_name, branch_name in ((origin_remote, origin_branch), (upstream_remote, upstream_branch)):
            if remote_name not in remotes:
                continue
            fetch_result = run_git(repo_root, 'fetch', remote_name, branch_name, '--prune', check=False)
            if fetch_result.returncode != 0:
                warnings.append(
                    f'Failed to fetch {remote_name}/{branch_name}: {fetch_result.stderr.strip()}',
                )

    origin_ref = f'{origin_remote}/{origin_branch}'
    upstream_ref = f'{upstream_remote}/{upstream_branch}'

    origin_ref_exists = _git_ref_exists(repo_root, origin_ref)
    upstream_ref_exists = _git_ref_exists(repo_root, upstream_ref)

    origin_only = -1
    upstream_only = -1
    if origin_ref_exists and upstream_ref_exists:
        divergence = run_git(repo_root, 'rev-list', '--left-right', '--count', f'{origin_ref}...{upstream_ref}', check=False)
        if divergence.returncode != 0:
            warnings.append(f'Failed to compute divergence: {divergence.stderr.strip()}')
        else:
            pieces = divergence.stdout.strip().split()
            if len(pieces) == 2:
                origin_only = int(pieces[0])
                upstream_only = int(pieces[1])
            else:
                warnings.append(f'Unexpected divergence output: {divergence.stdout.strip()}')
    else:
        missing_refs = []
        if not origin_ref_exists:
            missing_refs.append(origin_ref)
        if not upstream_ref_exists:
            missing_refs.append(upstream_ref)
        warnings.append(f'Missing refs for divergence check: {", ".join(missing_refs)}')

    sync_button_safe = False
    if origin_only >= 0 and upstream_only >= 0:
        sync_button_safe = origin_only <= max_origin_only
        if require_upstream_only:
            sync_button_safe = sync_button_safe and upstream_only > 0
        if require_clean_worktree:
            sync_button_safe = sync_button_safe and clean_worktree

    # When --allow-missing-upstream is set and upstream refs are absent,
    # degrade sync-button-safety to a warning instead of hard-failing.
    _upstream_refs_absent = upstream_remote not in remotes or not upstream_ref_exists
    if args.check_sync_button_safety and not sync_button_safe:
        if args.allow_missing_upstream and _upstream_refs_absent:
            warnings.append(
                'Sync button safety check skipped (--allow-missing-upstream; upstream refs absent)'
            )
        else:
            issues.append('Sync button safety check failed; use PR-based sync lane')

    report = {
        'repo_root': str(repo_root),
        'policy_path': str(policy_path),
        'origin_ref': origin_ref,
        'upstream_ref': upstream_ref,
        'clean_worktree': clean_worktree,
        'origin_only_commits': origin_only,
        'upstream_only_commits': upstream_only,
        'sync_button_safe': sync_button_safe,
        'issues': issues,
        'warnings': warnings,
    }

    if args.output_json:
        output_json = args.output_json if args.output_json.is_absolute() else (Path.cwd() / args.output_json).resolve()
        dump_json(output_json, report)

    print('Upstream sync doctor report')
    print('--------------------------')
    print(f'Repo: {repo_root}')
    print(f'Policy: {policy_path}')
    print(f'Clean worktree: {clean_worktree}')
    print(f'Origin-only commits: {origin_only}')
    print(f'Upstream-only commits: {upstream_only}')
    print(f'Sync button safe: {sync_button_safe}')

    if warnings:
        print('Warnings:')
        for warning in warnings:
            print(f'- {warning}')

    if issues:
        print('Issues:')
        for issue in issues:
            print(f'- {issue}')
        return 1

    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, subprocess.CalledProcessError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise SystemExit(2) from exc
