"""Closure semantics pass for temporal knowledge graphs.

When edges with the following semantic relation names are present in the graph,
this module triggers automatic state transitions on the *target* entity's active
(``invalid_at IS NULL``) facts:

``RESOLVES``
    Source entity *resolves* (closes / fixes / answers) a target entity.
    All currently-active outgoing facts on the target are invalidated at the
    time the RESOLVES edge became valid (``valid_at``), or at the edge's
    ``created_at`` timestamp if ``valid_at`` is not set.

    Example: Issue:PR-42 --RESOLVES--> Issue:BUG-7
    → All active facts about BUG-7 are closed at the time PR-42 was merged.

``SUPERSEDES``
    Source entity *supersedes* (replaces) a target entity.
    Same invalidation semantics as RESOLVES.

    Example: Policy:v2 --SUPERSEDES--> Policy:v1
    → All active facts about Policy:v1 are closed at the time v2 took effect.

These transitions are *additive and idempotent*: already-invalidated edges are
left untouched; the pass can be run multiple times safely.

The pass operates at the graph level and does NOT require LLM calls.  It is
safe to run after any ``add_episode`` call or as a scheduled maintenance job.

Usage
-----
In-process (e.g. inside a post-ingestion hook)::

    from graphiti_core.utils.maintenance.closure import apply_closure_semantics
    await apply_closure_semantics(driver, group_id="my_lane")

Dry-run preview (returns counts without writing)::

    result = await apply_closure_semantics(
        driver, group_id="my_lane", dry_run=True
    )
    print(result)  # ClosureResult(...)

Offline script::

    python scripts/apply_closure_semantics.py --group-id my_lane --dry-run
    python scripts/apply_closure_semantics.py --group-id my_lane --apply
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical closure edge names (SCREAMING_SNAKE_CASE, already normalized)
# ---------------------------------------------------------------------------

#: Relation types that trigger closure / invalidation of target entity's facts.
CLOSURE_EDGE_NAMES: frozenset[str] = frozenset({'RESOLVES', 'SUPERSEDES'})


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class ClosureResult:
    """Summary of a closure semantics pass."""

    closure_edges_found: int = 0
    """Number of RESOLVES/SUPERSEDES edges discovered in the graph."""

    facts_invalidated: int = 0
    """Number of EntityEdge facts that were (or would be) marked invalid_at."""

    dry_run: bool = True
    """Whether this was a dry-run (no writes performed)."""

    detail: list[dict] = field(default_factory=list)
    """Per-closure-edge breakdown (for debugging / audit log)."""

    def __repr__(self) -> str:
        mode = 'DRY-RUN' if self.dry_run else 'APPLIED'
        return (
            f'ClosureResult({mode}, closure_edges={self.closure_edges_found}, '
            f'facts_invalidated={self.facts_invalidated})'
        )


# ---------------------------------------------------------------------------
# Cypher queries (Neo4j / FalkorDB compatible)
# ---------------------------------------------------------------------------

# Find all RESOLVES/SUPERSEDES edges with their timestamps and endpoints.
# Returns: closure edge uuid, source node uuid, target node uuid, invalidation time.
_FIND_CLOSURE_EDGES_QUERY = """
MATCH (src:Entity)-[ce:RELATES_TO]->(tgt:Entity)
WHERE ce.name IN $closure_names
  AND ($group_id IS NULL OR ce.group_id = $group_id)
RETURN
    ce.uuid          AS closure_uuid,
    ce.name          AS closure_name,
    src.uuid         AS source_uuid,
    src.name         AS source_name,
    tgt.uuid         AS target_uuid,
    tgt.name         AS target_name,
    ce.valid_at      AS valid_at,
    ce.created_at    AS created_at
"""

# Find all currently-active (invalid_at IS NULL) facts on the target entity.
_FIND_ACTIVE_FACTS_QUERY = """
MATCH (src:Entity)-[e:RELATES_TO]->(tgt:Entity)
WHERE src.uuid = $target_uuid
  AND e.invalid_at IS NULL
  AND e.uuid <> $closure_uuid
  AND ($group_id IS NULL OR e.group_id = $group_id)
RETURN e.uuid AS fact_uuid, e.name AS fact_name
UNION
MATCH (src:Entity)-[e:RELATES_TO]->(tgt:Entity)
WHERE tgt.uuid = $target_uuid
  AND e.invalid_at IS NULL
  AND e.uuid <> $closure_uuid
  AND ($group_id IS NULL OR e.group_id = $group_id)
RETURN e.uuid AS fact_uuid, e.name AS fact_name
"""

# Mark a batch of facts as invalid at the given timestamp.
_INVALIDATE_FACTS_QUERY = """
UNWIND $uuids AS fact_uuid
MATCH ()-[e:RELATES_TO {uuid: fact_uuid}]-()
WHERE e.invalid_at IS NULL
SET e.invalid_at = $invalid_at
"""


# ---------------------------------------------------------------------------
# Core pass
# ---------------------------------------------------------------------------

async def apply_closure_semantics(
    driver,
    *,
    group_id: Optional[str] = None,
    dry_run: bool = False,
) -> ClosureResult:
    """Run a single closure semantics pass over the graph.

    Parameters
    ----------
    driver:
        An instance of ``graphiti_core.driver.driver.GraphDriver`` (or any
        object that exposes an async ``execute_query`` method with the same
        signature).
    group_id:
        Restrict the pass to a single lane/group.  ``None`` scans all groups.
    dry_run:
        If ``True`` (default), collect what *would* be invalidated but do not
        write any changes.  Set to ``False`` to apply.

    Returns
    -------
    ClosureResult
        Summary of what was found and (optionally) changed.
    """
    result = ClosureResult(dry_run=dry_run)

    # --- 1. Find all closure edges ----------------------------------------
    records, _, _ = await driver.execute_query(
        _FIND_CLOSURE_EDGES_QUERY,
        closure_names=list(CLOSURE_EDGE_NAMES),
        group_id=group_id,
        routing_='r',
    )

    if not records:
        logger.info('closure_semantics: no RESOLVES/SUPERSEDES edges found (group_id=%r)', group_id)
        return result

    result.closure_edges_found = len(records)
    logger.info(
        'closure_semantics: found %d closure edge(s) (group_id=%r)',
        result.closure_edges_found,
        group_id,
    )

    # --- 2. Collect active facts for each closure edge target ---------------

    # Pre-compute invalidation timestamps so we can sort deterministically.
    # Sorting by (invalid_at ascending, closure_uuid) gives two guarantees:
    #   a) Deterministic processing order — independent of Neo4j scan order.
    #   b) Min semantics — when multiple closures target the same fact the
    #      earliest timestamp is written first.  Because _INVALIDATE_FACTS_QUERY
    #      guards with WHERE e.invalid_at IS NULL, subsequent (later) closures
    #      for the same fact are no-ops, so the earliest timestamp always wins.
    def _resolve_invalid_at(rec: dict) -> datetime:
        if rec.get('valid_at'):
            return _to_utc(rec['valid_at'])
        if rec.get('created_at'):
            return _to_utc(rec['created_at'])
        return datetime.now(tz=timezone.utc)

    records_with_ts = [
        (rec, _resolve_invalid_at(rec)) for rec in records
    ]
    records_with_ts.sort(
        key=lambda pair: (pair[1], str(pair[0].get('closure_uuid', '')))
    )

    all_uuids_to_invalidate: list[str] = []

    for rec, invalid_at in records_with_ts:
        closure_uuid = rec['closure_uuid']
        closure_name = rec['closure_name']
        target_uuid = rec['target_uuid']
        target_name = rec['target_name']
        source_name = rec['source_name']

        # Find active facts on the target, excluding the closure edge itself
        fact_records, _, _ = await driver.execute_query(
            _FIND_ACTIVE_FACTS_QUERY,
            target_uuid=target_uuid,
            closure_uuid=closure_uuid,
            group_id=group_id,
            routing_='r',
        )
        fact_uuids = [r['fact_uuid'] for r in fact_records]

        edge_detail = {
            'closure_uuid': closure_uuid,
            'closure_name': closure_name,
            'source_name': source_name,
            'target_name': target_name,
            'target_uuid': target_uuid,
            'invalid_at': invalid_at.isoformat() if invalid_at else None,
            'facts_to_invalidate': len(fact_uuids),
        }
        result.detail.append(edge_detail)

        logger.debug(
            'closure_semantics: %s %r --%s--> %r (%d active facts)',
            'would invalidate' if dry_run else 'invalidating',
            source_name,
            closure_name,
            target_name,
            len(fact_uuids),
        )

        if fact_uuids and not dry_run:
            await driver.execute_query(
                _INVALIDATE_FACTS_QUERY,
                uuids=fact_uuids,
                invalid_at=invalid_at,
                routing_='w',
            )

        all_uuids_to_invalidate.extend(fact_uuids)

    result.facts_invalidated = len(all_uuids_to_invalidate)

    if dry_run:
        logger.info(
            'closure_semantics: DRY RUN — would invalidate %d fact(s) across %d closure edge(s)',
            result.facts_invalidated,
            result.closure_edges_found,
        )
    else:
        logger.info(
            'closure_semantics: applied — invalidated %d fact(s) across %d closure edge(s)',
            result.facts_invalidated,
            result.closure_edges_found,
        )

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_utc(ts) -> datetime:
    """Convert a Neo4j DateTime or Python datetime to a UTC-aware datetime.

    Handles three input shapes:

    * ``datetime`` — Python stdlib datetime (aware or naive).
    * Neo4j ``DateTime`` — exposes ``.to_native()``.
    * Anything else — stringified and parsed with ``fromisoformat``.

    The key correctness rule: **always use** ``astimezone(timezone.utc)``
    when the input already carries a UTC-offset.  Using
    ``.replace(tzinfo=timezone.utc)`` on an offset-aware datetime silently
    keeps the wall-clock values while re-labelling the zone as UTC
    (e.g. "10:00+05:30" would become "10:00Z" instead of "04:30Z").
    """
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(timezone.utc)
    # Neo4j DateTime object — convert to Python datetime
    try:
        dt = ts.to_native()
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except AttributeError:
        # Fallback: parse ISO string.  Must preserve and convert any embedded
        # offset rather than stamping UTC unconditionally via .replace().
        dt = datetime.fromisoformat(str(ts))
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
