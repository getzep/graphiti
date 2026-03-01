#!/usr/bin/env python3
"""Offline closure semantics maintenance script.

Scans the graph for RESOLVES and SUPERSEDES edges and automatically marks the
*target* entity's active facts as invalid at the time of the closure event.

This implements the closure semantics pass described in Phase C, Slice 2.
Run with ``--dry-run`` (default) to preview; use ``--apply`` to commit.

Usage
-----
# Preview (no writes)
python scripts/apply_closure_semantics.py

# Apply for a specific lane
python scripts/apply_closure_semantics.py --apply --group-id s1_sessions_main

# Apply for all lanes
python scripts/apply_closure_semantics.py --apply

Environment variables
---------------------
NEO4J_URI       e.g. bolt://localhost:7687
NEO4J_USER      default: neo4j
NEO4J_PASSWORD
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
)
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Apply closure semantics (RESOLVES/SUPERSEDES → invalid_at).',
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
        help='Restrict scan to a single lane/group_id.  Omit to scan all lanes.',
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


async def _main(args: argparse.Namespace) -> int:
    try:
        from neo4j import AsyncGraphDatabase, basic_auth
    except ImportError:
        logger.error('neo4j driver not installed.  Run: pip install neo4j')
        return 1

    try:
        from graphiti_core.utils.maintenance.closure import apply_closure_semantics
    except ImportError:
        logger.error(
            'graphiti_core not found on PYTHONPATH.  '
            'Run from the repo root with the project venv active.'
        )
        return 1

    # Minimal async driver wrapper that satisfies closure.apply_closure_semantics
    class _DriverAdapter:
        def __init__(self, driver):
            self._driver = driver

        async def execute_query(self, query: str, routing_: str = 'r', **params):
            async with self._driver.session() as session:
                result = await session.run(query, **params)
                records = await result.data()
                # Return (records, keys, summary) tuple — keys/summary unused
                return records, [], None

    driver_raw = AsyncGraphDatabase.driver(
        args.neo4j_uri,
        auth=basic_auth(args.neo4j_user, args.neo4j_password),
    )
    adapter = _DriverAdapter(driver_raw)

    try:
        result = await apply_closure_semantics(
            adapter,
            group_id=args.group_id,
            dry_run=not args.apply,
        )
    finally:
        await driver_raw.close()

    logger.info('Result: %r', result)

    # Print detail per closure edge
    for item in result.detail:
        logger.info(
            '  [%s] %s --%s--> %s : %d fact(s) → invalid_at=%s',
            'DRY' if result.dry_run else 'APPLIED',
            item['source_name'],
            item['closure_name'],
            item['target_name'],
            item['facts_to_invalidate'],
            item['invalid_at'],
        )

    return 0


if __name__ == '__main__':
    args = _parse_args()
    sys.exit(asyncio.run(_main(args)))
