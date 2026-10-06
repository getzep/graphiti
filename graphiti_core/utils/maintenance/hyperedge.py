"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

from graphiti_core.edges import EntityEdge
from graphiti_core.utils.datetime_utils import ensure_utc, utc_now


@dataclass
class Hyperedge:
    """The edges sharing one hyperedge_uuid: one atomic fact seen as several pairwise
    edges. Its properties are the shared fact and window; apply() writes them onto every
    member."""

    uuid: str
    members: list[EntityEdge]

    @property
    def representative(self) -> EntityEdge:
        """The first member seen — used to date the group with a single LLM call."""
        return self.members[0]

    @property
    def is_undated(self) -> bool:
        """True when no member has a date yet, so the group still needs one."""
        return all(m.valid_at is None and m.invalid_at is None for m in self.members)

    @property
    def fact(self) -> str:
        """The canonical fact, from the representative. Live members are listed before
        retired ones, so superseded wording is never donated back to a live member."""
        return self.representative.fact

    @property
    def invalid_at(self) -> datetime | None:
        """Latest end across members, or None if none has ended — the widest window.

        If any member has an end, the whole fact ends at the latest one.
        """
        ends = [end for m in self.members if (end := ensure_utc(m.invalid_at)) is not None]
        return max(ends, default=None)

    @property
    def valid_at(self) -> datetime | None:
        """Earliest start across members (widest window), never after invalid_at.

        An end that precedes every start still ends the fact (retraction wins), so the
        start is clamped down to that end rather than leaving an inverted window.
        """
        starts = [start for m in self.members if (start := ensure_utc(m.valid_at)) is not None]
        valid_at = min(starts, default=None)
        invalid_at = self.invalid_at
        if valid_at is not None and invalid_at is not None and valid_at > invalid_at:
            return invalid_at
        return valid_at

    @property
    def expired_at(self) -> datetime | None:
        """Latest non-null expired_at across members, or None if none has one.

        Matches invalid_at (widest window). A later member may inherit an expired_at
        earlier than its own created_at; that is coherent for one atomic fact.
        """
        expiries = [exp for m in self.members if (exp := ensure_utc(m.expired_at)) is not None]
        return max(expiries, default=None)

    def apply(self) -> None:
        """Write the shared fact and window onto every member.

        Clear a member's fact_embedding when its fact changes so it is re-embedded on
        save. expired_at is not written here — apply() runs in the pre-contradiction
        dating pass, and contradiction still gates on per-member expired_at is None.
        Share expiry in apply_expiry() when the group is finalized.

        reference_time stays per member: it records the episode a projection came from.
        """
        fact = self.fact
        valid_at = self.valid_at
        invalid_at = self.invalid_at
        for member in self.members:
            if member.fact != fact:
                member.fact = fact
                member.fact_embedding = None
            member.valid_at = valid_at
            member.invalid_at = invalid_at

    def apply_expiry(self, now: datetime) -> None:
        """Write one shared expired_at onto every member.

        Separate from apply() because _extract_hyperedge_timestamps calls apply()
        before contradiction, which still gates on expired_at is None. Sharing
        expiry there would let a retired sibling suppress supersession for a live
        member. Run this only when finalizing a group (resolve_hyperedges).

        Canonical value is the latest non-null member expired_at. If none has one
        but the group has ended (invalid_at is set), stamp now. Otherwise leave
        all as None.
        """
        canonical = self.expired_at
        if canonical is not None:
            for member in self.members:
                member.expired_at = canonical
        elif self.invalid_at is not None:
            for member in self.members:
                member.expired_at = now


def group_hyperedges(edges: list[EntityEdge]) -> list[Hyperedge]:
    """Group edges that share a hyperedge_uuid, in first-seen order.

    Untagged edges are skipped, so every returned group is a real hyperedge. Members are
    de-duplicated by uuid, so an edge that appears in more than one input list is counted
    once.
    """
    groups: dict[str, Hyperedge] = {}
    seen: set[str] = set()
    for edge in edges:
        if not edge.hyperedge_uuid or edge.uuid in seen:
            continue
        seen.add(edge.uuid)
        group = groups.get(edge.hyperedge_uuid)
        if group is None:
            group = Hyperedge(uuid=edge.hyperedge_uuid, members=[])
            groups[edge.hyperedge_uuid] = group
        group.members.append(edge)
    return list(groups.values())


def is_hyperedge(edges: Iterable[EntityEdge]) -> bool:
    """True when these edges are one hyperedge: two or more edges over three or more nodes.

    Edges are counted by uuid, because dedupe can map two members onto one stored edge.
    Nodes are read from the endpoints the edges carry now.
    """
    members = {edge.uuid: edge for edge in edges}
    if len(members) < 2:
        return False
    endpoints = {
        uuid for edge in members.values() for uuid in (edge.source_node_uuid, edge.target_node_uuid)
    }
    return len(endpoints) >= 3


def untag_invalid_groups(edges: list[EntityEdge], minted_uuids: set[str]) -> None:
    """Clear a uuid minted this batch whose group is no longer a hyperedge.

    Extraction minted it before node resolution, so re-check it here: dedupe can drop it
    below two edges, and resolve_edge_pointers can drop it below three nodes.

    Only uuids minted this batch are cleared. A stored uuid keeps its siblings elsewhere,
    so this cannot judge it.
    """
    invalid = {
        group.uuid
        for group in group_hyperedges(edges)
        if group.uuid in minted_uuids and not is_hyperedge(group.members)
    }
    # Sweep the list, not the members: group_hyperedges keeps one instance per edge uuid.
    for edge in edges:
        if edge.hyperedge_uuid in invalid:
            edge.hyperedge_uuid = None


def minted_hyperedge_uuids(edges: Iterable[EntityEdge]) -> set[str]:
    """The hyperedge uuids stamped on these edges.

    Call this on the edges that came out of extraction: every stamp they carry was
    minted during this extraction, which is what makes them eligible for untagging.
    """
    return {edge.hyperedge_uuid for edge in edges if edge.hyperedge_uuid}


def absorb_into_hyperedge(group_member: EntityEdge, target: EntityEdge) -> None:
    """Give ``target`` the group ``group_member`` belongs to, and that group's fact.

    Dedupe keeps one edge and replaces the other. When the kept edge is ungrouped the
    group silently loses this member, so the kept edge takes the group's uuid to stay
    counted, and the group's fact so a sibling that was not deduped keeps describing its
    own endpoints. Without the fact the kept edge can become the representative and have
    its own wording rewritten onto every sibling.

    A target already in a group keeps it, so two groups never merge. Episodes are left to
    the caller.
    """
    if not group_member.hyperedge_uuid or target.hyperedge_uuid:
        return
    target.hyperedge_uuid = group_member.hyperedge_uuid
    if target.fact != group_member.fact:
        target.fact = group_member.fact
        target.fact_embedding = None


def absorb_canonical_edges(uuid_map: dict[str, str], edges_by_uuid: dict[str, EntityEdge]) -> None:
    """Absorb each canonical edge into the group of the duplicate it replaces.

    ``uuid_map`` maps every uuid in a duplicate set to that set's canonical uuid. The
    canonical edge can be ungrouped, and it replaces the grouped edge before any
    stored-edge dedupe runs, so the group would silently lose that member.

    Call this before the canonical edge replaces its duplicates. A canonical edge already
    in a group keeps it, so two groups never merge.
    """
    for uuid, canonical_uuid in uuid_map.items():
        if uuid == canonical_uuid:
            continue
        duplicate = edges_by_uuid.get(uuid)
        canonical = edges_by_uuid.get(canonical_uuid)
        if duplicate is not None and canonical is not None:
            absorb_into_hyperedge(duplicate, canonical)


def resolve_hyperedges(
    resolved_edges: list[EntityEdge],
    invalidated_edges: list[EntityEdge],
    minted_uuids: set[str] | None = None,
) -> None:
    """Finalize hyperedges after a batch resolves, updating the edges in place. Two steps:

    1. Clear the uuid of a group minted this batch that is no longer a hyperedge (pass
       minted_uuids to enable this; a stored uuid is never cleared). Build the set with
       minted_hyperedge_uuids() over the extracted edges.
    2. Give every group one shared fact and the widest [min valid_at, max invalid_at]
       window across all its members (live and retired), then share one expired_at
       (latest non-null, or now if the group has ended).

    Step 1 needs the uuids minted by extraction, not the uuids of the edges that stayed
    new. Deriving them from the new edges would miss the case where every member of a new
    group deduped onto the same stored edge: nothing is new, so nothing would be cleared,
    and that stored edge would persist as a one-member hyperedge.

    Only the edges passed in are updated. A stored sibling that is not passed in keeps
    its old fact and window and can diverge from the group; a caller that needs
    whole-group consistency must load every member (GetEdgesByHyperedgeIds) and pass
    them all in.
    """
    if minted_uuids:
        untag_invalid_groups(resolved_edges, minted_uuids)
    now = utc_now()
    for hyperedge in group_hyperedges([*resolved_edges, *invalidated_edges]):
        hyperedge.apply()
        hyperedge.apply_expiry(now)
