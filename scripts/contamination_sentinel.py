#!/usr/bin/env python3
"""Contamination sentinel — cross-lane extraction integrity check.

Detects when facts from one lane (group_id) have unexpectedly landed in
another lane's episodic or entity records.  This can happen when:
  - A graph operation writes with the wrong group_id
  - Dedup merges facts from separate lanes
  - An ingestion script uses a shared namespace by mistake

The sentinel runs **read-only** queries; it never modifies the graph.
Exit code 0 = clean; exit code 1 = contamination detected.

Usage
-----
# Check all lanes for cross-lane contamination
python scripts/contamination_sentinel.py

# Check specific lane pair
python scripts/contamination_sentinel.py --source-group s1_sessions_main \\
    --expect-clean-in s1_inspiration_short_form

# JSON output for CI
python scripts/contamination_sentinel.py --json

Environment variables
---------------------
NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD  — standard connection vars.
SENTINEL_SAMPLE_LIMIT                   — max edges to sample per pair (default 50).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_SAMPLE_LIMIT: int = 50

# Nodes with *multiple* group_ids set are a red-flag; a node should belong to
# one lane only (unless explicitly shared, e.g. a cross-lane entity merge).
#
# IMPORTANT: Do NOT use ``size(n.group_id) > 1`` here.  In Cypher, size() on a
# STRING returns its character count, not element count, so any group_id string
# longer than one character (e.g. "main") would be incorrectly flagged.
#
# Strategy: fetch all non-null group_id values (limited by $limit) and let
# _is_multi_group_value() perform type-safe Python-side filtering.  This covers
# both storage formats:
#   • str with comma separator  → "group_a,group_b"
#   • Neo4j native list property → ["group_a", "group_b"]
_MULTI_GROUP_NODES_QUERY = """
MATCH (n:Entity)
WHERE n.group_id IS NOT NULL
RETURN n.uuid AS uuid, n.name AS name, n.group_id AS group_id
LIMIT $limit
"""

# Episodic records whose group_id does not match their source episode's lane
_EPISODIC_GROUP_MISMATCH_QUERY = """
MATCH (ep:Episodic)-[he:HAS_EPISODE]-(n:Entity)
WHERE ep.group_id IS NOT NULL
  AND n.group_id IS NOT NULL
  AND ep.group_id <> n.group_id
  AND ($source_group IS NULL OR ep.group_id = $source_group)
  AND ($clean_group IS NULL OR n.group_id = $clean_group)
RETURN
    ep.uuid AS episode_uuid,
    ep.group_id AS episode_group,
    n.uuid AS entity_uuid,
    n.name AS entity_name,
    n.group_id AS entity_group
LIMIT $limit
"""

# EntityEdge whose group_id differs from both its source and target entity's group_ids
_EDGE_GROUP_MISMATCH_QUERY = """
MATCH (src:Entity)-[e:RELATES_TO]->(tgt:Entity)
WHERE e.group_id IS NOT NULL
  AND src.group_id IS NOT NULL
  AND tgt.group_id IS NOT NULL
  AND e.group_id <> src.group_id
  AND e.group_id <> tgt.group_id
  AND ($source_group IS NULL OR e.group_id = $source_group)
  AND ($clean_group IS NULL OR src.group_id = $clean_group OR tgt.group_id = $clean_group)
RETURN
    e.uuid AS edge_uuid,
    e.name AS edge_name,
    e.group_id AS edge_group,
    src.group_id AS src_group,
    tgt.group_id AS tgt_group
LIMIT $limit
"""


# ---------------------------------------------------------------------------
# Multi-group value detector (type-safe, unit-testable)
# ---------------------------------------------------------------------------

def _is_multi_group_value(value: Any) -> bool:
    """Return True iff *value* encodes multiple group_ids.

    Handles the two storage formats used in the graph:

    * **Python list** (Neo4j native list property): multi-group iff ``len > 1``.
      A single-element list ``["main"]`` is NOT flagged.
    * **str**: multi-group iff contains a comma separator (e.g. ``"a,b"``).
      A normal single group_id string such as ``"s1_sessions_main"`` is NOT
      flagged regardless of its length.

    Any other type (None, int, …) returns False so the caller's filter is
    conservative — better to miss an exotic edge case than to spam false
    positives.
    """
    if isinstance(value, list):
        return len(value) > 1
    if isinstance(value, str):
        return ',' in value
    return False


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class ContaminationResult:
    multi_group_nodes: list[dict] = field(default_factory=list)
    episodic_mismatches: list[dict] = field(default_factory=list)
    edge_mismatches: list[dict] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return (
            not self.multi_group_nodes
            and not self.episodic_mismatches
            and not self.edge_mismatches
        )

    @property
    def total_issues(self) -> int:
        return (
            len(self.multi_group_nodes)
            + len(self.episodic_mismatches)
            + len(self.edge_mismatches)
        )

    def to_dict(self) -> dict:
        return {
            'clean': self.is_clean,
            'total_issues': self.total_issues,
            'multi_group_nodes': self.multi_group_nodes,
            'episodic_mismatches': self.episodic_mismatches,
            'edge_mismatches': self.edge_mismatches,
        }


# ---------------------------------------------------------------------------
# Core scan logic (sync, pure Neo4j bolt)
# ---------------------------------------------------------------------------

def run_sentinel(
    *,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    source_group: Optional[str] = None,
    clean_group: Optional[str] = None,
    sample_limit: int = DEFAULT_SAMPLE_LIMIT,
) -> ContaminationResult:
    """Run the contamination sentinel and return results."""
    try:
        from neo4j import GraphDatabase, basic_auth
    except ImportError:
        logger.error('neo4j driver not installed.  Run: pip install neo4j')
        sys.exit(1)

    result = ContaminationResult()
    driver = GraphDatabase.driver(neo4j_uri, auth=basic_auth(neo4j_user, neo4j_password))

    try:
        with driver.session() as session:
            # 1. Multi-group nodes — Python-side filter for type-safe detection.
            #    The Cypher query returns all non-null group_id nodes (up to $limit).
            #    _is_multi_group_value() distinguishes single normal strings from
            #    comma-encoded or list-encoded multi-group values without relying on
            #    size() which returns string character count, not element count.
            recs = session.run(_MULTI_GROUP_NODES_QUERY, limit=sample_limit)
            result.multi_group_nodes = [
                dict(r) for r in recs
                if _is_multi_group_value(r.get('group_id'))
            ]

            # 2. Episodic group mismatches
            recs = session.run(
                _EPISODIC_GROUP_MISMATCH_QUERY,
                source_group=source_group,
                clean_group=clean_group,
                limit=sample_limit,
            )
            result.episodic_mismatches = [dict(r) for r in recs]

            # 3. Edge group mismatches
            recs = session.run(
                _EDGE_GROUP_MISMATCH_QUERY,
                source_group=source_group,
                clean_group=clean_group,
                limit=sample_limit,
            )
            result.edge_mismatches = [dict(r) for r in recs]
    finally:
        driver.close()

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Contamination sentinel — cross-lane extraction integrity check.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--source-group',
        default=None,
        metavar='GROUP_ID',
        help='Filter: episodic/edge group_id to check contamination from.',
    )
    parser.add_argument(
        '--expect-clean-in',
        default=None,
        dest='clean_group',
        metavar='GROUP_ID',
        help='Filter: entity group_id that should not be contaminated.',
    )
    parser.add_argument(
        '--json',
        action='store_true',
        default=False,
        help='Emit JSON result to stdout (machine-readable for CI).',
    )
    parser.add_argument(
        '--sample-limit',
        type=int,
        default=int(os.environ.get('SENTINEL_SAMPLE_LIMIT', str(DEFAULT_SAMPLE_LIMIT))),
        help=f'Max rows to return per check (default: {DEFAULT_SAMPLE_LIMIT}).',
    )
    parser.add_argument(
        '--neo4j-uri',
        default=os.environ.get('NEO4J_URI', 'bolt://localhost:7687'),
    )
    parser.add_argument(
        '--neo4j-user',
        default=os.environ.get('NEO4J_USER', 'neo4j'),
    )
    parser.add_argument(
        '--neo4j-password',
        default=os.environ.get('NEO4J_PASSWORD', ''),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    result = run_sentinel(
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        source_group=args.source_group,
        clean_group=args.clean_group,
        sample_limit=args.sample_limit,
    )

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, default=str))
    else:
        if result.is_clean:
            logger.info('Contamination sentinel: CLEAN — no cross-lane issues found.')
        else:
            logger.warning(
                'Contamination sentinel: %d issue(s) found!', result.total_issues
            )
            if result.multi_group_nodes:
                logger.warning(
                    '  multi-group nodes (%d):', len(result.multi_group_nodes)
                )
                for item in result.multi_group_nodes[:5]:
                    logger.warning('    %s', item)

            if result.episodic_mismatches:
                logger.warning(
                    '  episodic group mismatches (%d):', len(result.episodic_mismatches)
                )
                for item in result.episodic_mismatches[:5]:
                    logger.warning('    %s', item)

            if result.edge_mismatches:
                logger.warning(
                    '  edge group mismatches (%d):', len(result.edge_mismatches)
                )
                for item in result.edge_mismatches[:5]:
                    logger.warning('    %s', item)

    return 0 if result.is_clean else 1


if __name__ == '__main__':
    sys.exit(main())
