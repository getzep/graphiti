from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'public_history_scorecard.py'


class PublicHistoryScorecardTests(unittest.TestCase):
    def _run(
        self,
        repo: Path,
        filtered: dict,
        clean: dict,
        policy: dict,
    ) -> subprocess.CompletedProcess[str]:
        (repo / 'tmp').mkdir(parents=True, exist_ok=True)
        filtered_path = repo / 'tmp' / 'filtered.json'
        clean_path = repo / 'tmp' / 'clean.json'
        policy_path = repo / 'tmp' / 'policy.json'
        out_path = repo / 'tmp' / 'scorecard.md'
        summary_path = repo / 'tmp' / 'scorecard.json'

        filtered_path.write_text(f'{json.dumps(filtered, indent=2)}\n', encoding='utf-8')
        clean_path.write_text(f'{json.dumps(clean, indent=2)}\n', encoding='utf-8')
        policy_path.write_text(f'{json.dumps(policy, indent=2)}\n', encoding='utf-8')

        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                '--filtered-summary',
                str(filtered_path),
                '--clean-summary',
                str(clean_path),
                '--policy',
                str(policy_path),
                '--out',
                str(out_path),
                '--summary-json',
                str(summary_path),
            ],
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_fallback_to_clean_when_filtered_below_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            filtered = {
                'metrics': {
                    'privacy_risk': 70,
                    'simplicity': 72,
                    'merge_conflict_risk': 70,
                    'auditability': 74,
                },
                'risk_flags': {'unresolved_high': False},
            }
            clean = {
                'metrics': {
                    'privacy_risk': 92,
                    'simplicity': 90,
                    'merge_conflict_risk': 91,
                    'auditability': 90,
                },
                'risk_flags': {'unresolved_high': False},
            }
            policy = {
                'version': 1,
                'scorecard': {
                    'clean_foundation_threshold': 80,
                    'weights': {
                        'privacy_risk': 0.35,
                        'simplicity': 0.35,
                        'merge_conflict_risk': 0.2,
                        'auditability': 0.1,
                    },
                },
            }

            result = self._run(repo, filtered=filtered, clean=clean, policy=policy)
            self.assertEqual(result.returncode, 0, msg=result.stderr)

            summary = json.loads((repo / 'tmp' / 'scorecard.json').read_text(encoding='utf-8'))
            self.assertEqual(summary['decision'], 'clean-foundation')

    def test_forces_clean_when_filtered_has_unresolved_high(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            filtered = {
                'metrics': {
                    'privacy_risk': 95,
                    'simplicity': 95,
                    'merge_conflict_risk': 95,
                    'auditability': 95,
                },
                'risk_flags': {'unresolved_high': True},
            }
            clean = {
                'metrics': {
                    'privacy_risk': 80,
                    'simplicity': 80,
                    'merge_conflict_risk': 80,
                    'auditability': 80,
                },
                'risk_flags': {'unresolved_high': False},
            }
            policy = {
                'version': 1,
                'scorecard': {
                    'clean_foundation_threshold': 80,
                    'weights': {
                        'privacy_risk': 0.25,
                        'simplicity': 0.25,
                        'merge_conflict_risk': 0.25,
                        'auditability': 0.25,
                    },
                },
            }

            result = self._run(repo, filtered=filtered, clean=clean, policy=policy)
            self.assertEqual(result.returncode, 0, msg=result.stderr)

            summary = json.loads((repo / 'tmp' / 'scorecard.json').read_text(encoding='utf-8'))
            self.assertEqual(summary['decision'], 'clean-foundation')


if __name__ == '__main__':
    unittest.main()
