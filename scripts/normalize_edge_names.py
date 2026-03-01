#!/usr/bin/env python3
"""Offline edge-name normalization maintenance script.

Scans all EntityEdge records in Neo4j and normalizes their ``name`` /
``relation_type`` properties to canonical SCREAMING_SNAKE_CASE.

This is safe to run at any time — it is a cosmetic rename only and does not
alter the semantic meaning of any edge.  Run with ``--dry-run`` (default) to
preview changes before applying them.

Usage
-----
# Preview only (no writes)
python scripts/normalize_edge_names.py

# Apply in-place
python scripts/normalize_edge_names.py --apply

# Limit to a specific group_id (lane)
python scripts/normalize_edge_names.py --apply --group-id s1_sessions_main

Environment variables
---------------------
NEO4J_URI      e.g. bolt://localhost:7687
NEO4J_USER     default: neo4j
NEO4J_PASSWORD
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dependency guard — explain how to install if missing
# ---------------------------------------------------------------------------
try:
    from neo4j import GraphDatabase, basic_auth
except ImportError:
    logger.error(
        'neo4j driver not installed.  Run: pip install neo4j\n'
        'Or activate the project venv: source .venv/bin/activate'
    )
    sys.exit(1)

try:
    from graphiti_core.utils.maintenance import normalize_relation_type
except ImportError:
    logger.error(
        'graphiti_core not found on PYTHONPATH.  '
        'Run from the repo root with the project venv active.'
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Cypher helpers
# ---------------------------------------------------------------------------
#
# EntityEdge objects are persisted as Neo4j *relationships* with the type
# RELATES_TO between two Entity nodes — NOT as standalone ``Entityedge``
# nodes.  The incorrect label query would silently return 0 rows and be a
# no-op on every run.

_SCAN_QUERY = """
MATCH (n:Entity)-[e:RELATES_TO]->(m:Entity)
WHERE $group_id IS NULL OR e.group_id = $group_id
RETURN e.uuid AS uuid, e.name AS name
"""

_UPDATE_QUERY = """
UNWIND $updates AS u
MATCH (n:Entity)-[e:RELATES_TO {uuid: u.uuid}]->(m:Entity)
SET e.name = u.new_name
"""


def _scan(session, group_id: Optional[str]) -> list[dict]:
    result = session.run(_SCAN_QUERY, group_id=group_id)
    return [dict(r) for r in result]


def _apply_updates(session, updates: list[dict]) -> int:
    if not updates:
        return 0
    session.run(_UPDATE_QUERY, updates=updates)
    return len(updates)


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def run(
    *,
    apply: bool,
    group_id: Optional[str],
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
) -> int:
    """Return number of edges that required normalization."""
    driver = GraphDatabase.driver(neo4j_uri, auth=basic_auth(neo4j_user, neo4j_password))

    try:
        with driver.session() as session:
            edges = _scan(session, group_id)

        logger.info('Scanned %d EntityEdge records%s.',
                    len(edges),
                    f' (group_id={group_id!r})' if group_id else '')

        updates: list[dict] = []
        for edge in edges:
            name: str = edge.get('name') or ''
            normalized = normalize_relation_type(name)
            if normalized != name:
                updates.append({'uuid': edge['uuid'], 'old_name': name, 'new_name': normalized})

        if not updates:
            logger.info('All %d edges already have canonical names — nothing to do.', len(edges))
            return 0

        logger.info(
            'Found %d edges with non-canonical names (%.1f%% of total).',
            len(updates),
            100.0 * len(updates) / max(len(edges), 1),
        )

        for u in updates[:20]:  # preview first 20
            logger.info('  %s → %s  (uuid=%s)', u['old_name'], u['new_name'], u['uuid'])
        if len(updates) > 20:
            logger.info('  … and %d more.', len(updates) - 20)

        if not apply:
            logger.info(
                'DRY RUN — no changes written.  Re-run with --apply to persist.'
            )
            return len(updates)

        with driver.session() as session:
            written = _apply_updates(session, updates)
        logger.info('Normalized %d edge names (committed).', written)
        return written

    finally:
        driver.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Normalize EntityEdge names to SCREAMING_SNAKE_CASE.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--apply',
        action='store_true',
        default=False,
        help='Write changes to Neo4j.  Omit for dry-run preview (default).',
    )
    parser.add_argument(
        '--group-id',
        metavar='GROUP_ID',
        default=None,
        help='Restrict scan to a single lane/group_id.  Omit to scan all.',
    )
    parser.add_argument(
        '--neo4j-uri',
        default=os.environ.get('NEO4J_URI', 'bolt://localhost:7687'),
        help='Neo4j bolt URI (default: $NEO4J_URI or bolt://localhost:7687).',
    )
    parser.add_argument(
        '--neo4j-user',
        default=os.environ.get('NEO4J_USER', 'neo4j'),
        help='Neo4j user (default: $NEO4J_USER or neo4j).',
    )
    parser.add_argument(
        '--neo4j-password',
        default=os.environ.get('NEO4J_PASSWORD', ''),
        help='Neo4j password (default: $NEO4J_PASSWORD).',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    changed = run(
        apply=args.apply,
        group_id=args.group_id,
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
    )
    sys.exit(0 if changed >= 0 else 1)
