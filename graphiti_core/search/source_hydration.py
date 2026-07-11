"""Source hydration (redevops fork, G1).

Graphiti extracts an LLM ``fact`` string per ``EntityEdge`` and keeps the source episode UUIDs on
``EntityEdge.episodes``. Search returns the extracted facts, which are *lossy* relative to the raw
turns they came from — great for point-in-time reasoning over a large, revised history, but weaker
than plain document retrieval when the task is really "find the session that says this".

Source hydration closes that gap: given result edges, batch-fetch their source episodes once and
attach the raw content back onto each edge (``edge.source_episodes``), so a caller can retrieve the
original turns instead of only the summarized fact — without a second round-trip per edge.

The collection/attachment logic here is pure (no DB), so it is unit-testable in isolation; the async
fetch lives on ``Graphiti.hydrate_edge_sources``.
"""

from __future__ import annotations

from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EpisodicNode


def collect_episode_uuids(edges: list[EntityEdge]) -> list[str]:
    """The de-duplicated, order-preserving set of source episode UUIDs referenced by ``edges``."""
    seen: set[str] = set()
    ordered: list[str] = []
    for edge in edges:
        for uuid in edge.episodes or []:
            if uuid not in seen:
                seen.add(uuid)
                ordered.append(uuid)
    return ordered


def attach_source_episodes(
    edges: list[EntityEdge],
    episodes_by_uuid: dict[str, EpisodicNode],
) -> list[EntityEdge]:
    """Attach fetched source episodes onto each edge (order-preserving, missing UUIDs skipped).

    Pure — the caller supplies the ``uuid -> EpisodicNode`` map (from the DB or a test fixture).
    Returns the same edge objects, now with ``source_episodes`` populated, for convenient chaining.
    """
    for edge in edges:
        edge.source_episodes = [
            episodes_by_uuid[uuid] for uuid in (edge.episodes or []) if uuid in episodes_by_uuid
        ]
    return edges


def deduped_source_episodes(edges: list[EntityEdge]) -> list[EpisodicNode]:
    """The distinct hydrated source episodes across ``edges``, in first-seen order.

    Useful to surface the raw sessions/turns alongside a fact-level result set (e.g. to fill
    ``SearchResults.episodes``). Requires the edges to have been hydrated first.
    """
    seen: set[str] = set()
    ordered: list[EpisodicNode] = []
    for edge in edges:
        for episode in edge.source_episodes or []:
            if episode.uuid not in seen:
                seen.add(episode.uuid)
                ordered.append(episode)
    return ordered
