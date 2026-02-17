from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / 'scripts'

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from public_boundary_policy import (  # noqa: E402
    ALLOW,
    AMBIGUOUS,
    BLOCK,
    RuleDecision,
    build_markdown_report,
    classify_path,
    read_yaml_list,
    summarize_decisions,
)


class PublicBoundaryPolicyUnitTests(unittest.TestCase):
    def test_read_yaml_list_normalizes_prefix_and_ignores_comments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            policy = Path(tmp) / 'allow.yaml'
            policy.write_text(
                'version: 1\n\nallowlist:\n  - "./scripts/ci/**" # comment\n  - "docs/public/**"\n',
                encoding='utf-8',
            )

            rules = read_yaml_list(policy, 'allowlist')
            self.assertEqual(rules, ['scripts/ci/**', 'docs/public/**'])

    def test_classify_path_allowlist_overrides_denylist(self) -> None:
        decision = classify_path(
            path='docs/private/secrets.txt',
            allowlist=['docs/**'],
            denylist=['docs/private/**'],
        )
        self.assertEqual(decision.status, ALLOW)
        self.assertEqual(decision.reason_code, 'ALLOWLIST_MATCH')

    def test_summarize_decisions_splits_status_buckets(self) -> None:
        decisions = [
            RuleDecision(path='a', status=ALLOW, reason_code='ALLOWLIST_MATCH', matched_rule='a'),
            RuleDecision(path='b', status=BLOCK, reason_code='DENYLIST_MATCH', matched_rule='b'),
            RuleDecision(path='c', status=AMBIGUOUS, reason_code='NO_MATCH', matched_rule=None),
        ]

        counts, blocked, ambiguous = summarize_decisions(decisions)
        self.assertEqual(counts[ALLOW], 1)
        self.assertEqual(counts[BLOCK], 1)
        self.assertEqual(counts[AMBIGUOUS], 1)
        self.assertEqual([d.path for d in blocked], ['b'])
        self.assertEqual([d.path for d in ambiguous], ['c'])

    def test_build_markdown_report_contains_summary_and_offenders(self) -> None:
        decisions = [
            RuleDecision(path='safe.txt', status=ALLOW, reason_code='ALLOWLIST_MATCH', matched_rule='safe.txt'),
            RuleDecision(path='private.key', status=BLOCK, reason_code='DENYLIST_MATCH', matched_rule='*.key'),
            RuleDecision(path='misc.md', status=AMBIGUOUS, reason_code='NO_MATCH', matched_rule=None),
        ]
        counts, blocked, ambiguous = summarize_decisions(decisions)

        report = build_markdown_report(
            decisions=decisions,
            manifest_path=Path('config/public_export_allowlist.yaml'),
            denylist_path=Path('config/public_export_denylist.yaml'),
            include_untracked=False,
            status_counts=counts,
            block_list=blocked,
            ambiguous_list=ambiguous,
        )

        self.assertIn('| ALLOW | 1 |', report)
        self.assertIn('| BLOCK | 1 |', report)
        self.assertIn('| AMBIGUOUS | 1 |', report)
        self.assertIn('private.key', report)
        self.assertIn('misc.md', report)


class PublicBoundaryPolicyGitHelpersTests(unittest.TestCase):
    def test_script_module_importable_from_repo_context(self) -> None:
        # Sanity check: this mirrors subprocess-based script execution mode.
        result = subprocess.run(
            [sys.executable, '-c', 'import public_boundary_policy; print(public_boundary_policy.ALLOW)'],
            cwd=SCRIPTS_DIR,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(result.stdout.strip(), ALLOW)


if __name__ == '__main__':
    unittest.main()
