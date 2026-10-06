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

from datetime import datetime, timezone
from uuid import uuid4

from graphiti_core.edges import EntityEdge
from graphiti_core.utils.maintenance.hyperedge import (
    Hyperedge,
    absorb_canonical_edges,
    group_hyperedges,
    is_hyperedge,
    minted_hyperedge_uuids,
    resolve_hyperedges,
    untag_invalid_groups,
)

UTC = timezone.utc


def _dt(year: int, month: int = 1, day: int = 1) -> datetime:
    return datetime(year, month, day, tzinfo=UTC)


def _edge(
    *,
    source: str,
    target: str,
    fact: str,
    episodes: list[str] | None = None,
    hyperedge_uuid: str | None = None,
    valid_at: datetime | None = None,
    invalid_at: datetime | None = None,
    expired_at: datetime | None = None,
) -> EntityEdge:
    return EntityEdge(
        source_node_uuid=source,
        target_node_uuid=target,
        name='RELATES',
        group_id='g',
        fact=fact,
        episodes=episodes or ['ep-1'],
        created_at=datetime.now(timezone.utc),
        hyperedge_uuid=hyperedge_uuid,
        valid_at=valid_at,
        invalid_at=invalid_at,
        expired_at=expired_at,
    )


# --- is_hyperedge ----------------------------------------------------------


def test_is_hyperedge_needs_two_edges_over_three_nodes():
    a, b, c = (str(uuid4()) for _ in range(3))
    pair = _edge(source=a, target=b, fact='f')
    reverse = _edge(source=b, target=a, fact='f')
    other_relation = _edge(source=a, target=b, fact='f')
    third = _edge(source=a, target=c, fact='f')

    assert is_hyperedge([]) is False
    assert is_hyperedge([pair]) is False
    # Two edges over one pair of nodes say nothing a single edge does not.
    assert is_hyperedge([pair, reverse]) is False
    assert is_hyperedge([pair, other_relation]) is False
    assert is_hyperedge([pair, third]) is True


# --- untag_invalid_groups --------------------------------------------------


def test_untag_invalid_groups_clears_only_a_collapsed_mint():
    """A minted uuid left with one edge is cleared; a stored uuid and a real pair stay."""
    minted, stored, kept = str(uuid4()), str(uuid4()), str(uuid4())
    a, b, c, d, e = (str(uuid4()) for _ in range(5))
    collapsed = _edge(source=a, target=b, fact='f', hyperedge_uuid=minted)
    stored_single = _edge(source=a, target=c, fact='f', hyperedge_uuid=stored)
    pair_1 = _edge(source=a, target=d, fact='f', hyperedge_uuid=kept)
    pair_2 = _edge(source=a, target=e, fact='f', hyperedge_uuid=kept)

    untag_invalid_groups([collapsed, stored_single, pair_1, pair_2], {minted, kept})

    assert collapsed.hyperedge_uuid is None  # minted + single -> cleared
    assert stored_single.hyperedge_uuid == stored  # not minted this batch -> kept
    assert pair_1.hyperedge_uuid == kept  # minted but still a pair -> kept
    assert pair_2.hyperedge_uuid == kept


def test_resolve_hyperedges_without_minted_uuids_keeps_a_stored_single_member_uuid():
    """Re-resolving one stored member must not clear its uuid — only minted uuids are
    eligible, and none are passed here."""
    stored = str(uuid4())
    single = _edge(
        source=str(uuid4()),
        target=str(uuid4()),
        fact='f',
        hyperedge_uuid=stored,
        valid_at=_dt(2020),
    )

    resolve_hyperedges([single], [])

    assert single.hyperedge_uuid == stored


def test_resolve_hyperedges_untags_a_group_that_wholly_deduped_onto_one_stored_edge():
    """Every member of a minted group can dedupe onto the same stored edge.

    Nothing is new, so a minted set derived from the new edges would be empty and the
    stored edge would keep the stamp as a one-member hyperedge. The minted set comes
    from the extracted edges instead, so the stamp is cleared.
    """
    minted = str(uuid4())
    alice, bob, acme = str(uuid4()), str(uuid4()), str(uuid4())
    fact = 'Alice and Bob co-founded Acme.'
    # Two projections of one atomic fact, both stamped at extraction.
    extracted_1 = _edge(source=alice, target=acme, fact=fact, hyperedge_uuid=minted)
    extracted_2 = _edge(source=bob, target=acme, fact=fact, hyperedge_uuid=minted)
    # Dedupe resolved both onto one stored edge, which inherited the stamp.
    stored = _edge(source=alice, target=acme, fact=fact, hyperedge_uuid=minted, valid_at=_dt(2019))

    resolve_hyperedges(
        [stored, stored], [], minted_uuids=minted_hyperedge_uuids([extracted_1, extracted_2])
    )

    assert stored.hyperedge_uuid is None


def test_resolve_hyperedges_keeps_a_mint_left_beside_an_edge_from_another_group():
    """Section 6.2 case 4: one member deduped onto an edge already in another group.

    That edge keeps its stored uuid, and the members left over keep the mint as long as
    two distinct edges still carry it. The leftovers are projections of the fact this
    extraction read, so untagging them would leave identical fact text on unlinked edges.
    """
    minted, foreign = str(uuid4()), str(uuid4())
    alice, bob, dave, erin = (str(uuid4()) for _ in range(4))
    fact = 'Alice introduced Bob to Dave and Erin.'
    # Alice->Bob deduped onto a stored edge that already belongs to `foreign`.
    deduped = _edge(
        source=alice,
        target=bob,
        fact='Alice introduced Bob to Carol.',
        hyperedge_uuid=foreign,
        valid_at=_dt(2019),
    )
    left_1 = _edge(source=alice, target=dave, fact=fact, hyperedge_uuid=minted)
    left_2 = _edge(source=alice, target=erin, fact=fact, hyperedge_uuid=minted)

    resolve_hyperedges([deduped, left_1, left_2], [], minted_uuids={minted})

    assert deduped.hyperedge_uuid == foreign  # never merged, never reassigned
    assert left_1.hyperedge_uuid == minted
    assert left_2.hyperedge_uuid == minted


# --- group_hyperedges ------------------------------------------------------


def test_group_hyperedges_groups_tagged_and_ignores_untagged():
    h = str(uuid4())
    a, b, c, d = (str(uuid4()) for _ in range(4))
    m1 = _edge(source=a, target=b, fact='f', hyperedge_uuid=h)
    m2 = _edge(source=a, target=c, fact='f', hyperedge_uuid=h)
    untagged = _edge(source=a, target=d, fact='u')
    empty = _edge(source=b, target=c, fact='e', hyperedge_uuid='')

    groups = group_hyperedges([m1, m2, untagged, empty])

    assert len(groups) == 1
    assert groups[0].uuid == h
    assert {edge.uuid for edge in groups[0].members} == {m1.uuid, m2.uuid}


def test_group_hyperedges_dedupes_a_member_that_appears_twice():
    h = str(uuid4())
    a, b, c = str(uuid4()), str(uuid4()), str(uuid4())
    m1 = _edge(source=a, target=b, fact='f', hyperedge_uuid=h)
    m2 = _edge(source=a, target=c, fact='f', hyperedge_uuid=h)

    groups = group_hyperedges([m1, m2, m1])

    assert len(groups) == 1
    assert [edge.uuid for edge in groups[0].members] == [m1.uuid, m2.uuid]


# --- Hyperedge properties --------------------------------------------------


def test_hyperedge_properties_are_order_independent():
    h = str(uuid4())
    a = str(uuid4())
    early = _edge(
        source=a,
        target=str(uuid4()),
        fact='early',
        hyperedge_uuid=h,
        valid_at=_dt(2019),
        invalid_at=_dt(2020),
    )
    late = _edge(
        source=a,
        target=str(uuid4()),
        fact='late',
        hyperedge_uuid=h,
        valid_at=_dt(2021),
        invalid_at=_dt(2022),
    )
    hyperedge = Hyperedge(uuid=h, members=[late, early])

    assert hyperedge.representative is late  # first member seen
    assert hyperedge.fact == 'late'  # the representative's, not the earliest member's
    assert hyperedge.valid_at == _dt(2019)  # widest (earliest) start
    assert hyperedge.invalid_at == _dt(2022)  # widest (latest) end
    assert hyperedge.expired_at is None
    assert not hyperedge.is_undated

    early.expired_at = _dt(2020)
    late.expired_at = _dt(2023)
    assert hyperedge.expired_at == _dt(2023)  # latest non-null, order-independent


def test_hyperedge_is_undated_and_falls_back_to_first_member():
    h = str(uuid4())
    a = str(uuid4())
    m1 = _edge(source=a, target=str(uuid4()), fact='f', hyperedge_uuid=h)
    m2 = _edge(source=a, target=str(uuid4()), fact='f', hyperedge_uuid=h)
    hyperedge = Hyperedge(uuid=h, members=[m1, m2])

    assert hyperedge.is_undated
    assert hyperedge.valid_at is None
    assert hyperedge.invalid_at is None
    assert hyperedge.representative is m1


# --- normalization: widest window, shared fact -----------------------------


def test_resolve_hyperedges_gives_the_min_valid_at_to_members_with_none_or_later():
    """min valid_at wins: a member with no start and a member with a later start both
    take the earliest start (mirror of the max invalid_at rule)."""
    h = str(uuid4())
    a = str(uuid4())
    fact = 'Alice introduced Bob to Carol.'
    earliest = _edge(source=a, target=str(uuid4()), fact=fact, hyperedge_uuid=h, valid_at=_dt(2019))
    no_start = _edge(source=a, target=str(uuid4()), fact=fact, hyperedge_uuid=h)  # valid_at None
    later = _edge(source=a, target=str(uuid4()), fact=fact, hyperedge_uuid=h, valid_at=_dt(2021))

    resolve_hyperedges([earliest, no_start, later], [])

    for member in (earliest, no_start, later):
        assert member.valid_at == _dt(2019)
        assert member.invalid_at is None  # no member ended


def test_resolve_hyperedges_widens_invalid_at_to_the_latest_end():
    """Every member is pulled forward to the latest end across the group."""
    h = str(uuid4())
    a, b, c = str(uuid4()), str(uuid4()), str(uuid4())
    fact = 'Alice introduced Bob to Carol.'
    start = _dt(2019)
    m_start = _edge(source=a, target=b, fact=fact, hyperedge_uuid=h, valid_at=start)
    m_early = _edge(
        source=a, target=c, fact=fact, hyperedge_uuid=h, valid_at=start, invalid_at=_dt(2020)
    )
    m_late = _edge(
        source=a,
        target=str(uuid4()),
        fact=fact,
        hyperedge_uuid=h,
        valid_at=start,
        invalid_at=_dt(2021),
    )
    members = [m_start, m_early, m_late]

    resolve_hyperedges(members, [])

    assert {member.valid_at for member in members} == {start}
    assert {member.invalid_at for member in members} == {_dt(2021)}


def test_resolve_hyperedges_keeps_the_live_wording_when_a_retired_member_is_folded_in():
    """All members (live and retired) end identical, on the live wording. A retired
    member has an earlier start but its wording is superseded, so it must not donate."""
    h = str(uuid4())
    alice, bob, carol = str(uuid4()), str(uuid4()), str(uuid4())
    retired = _edge(
        source=alice,
        target=bob,
        fact='original wording',
        hyperedge_uuid=h,
        valid_at=_dt(2019),
        invalid_at=_dt(2021),
        expired_at=_dt(2021),
    )
    live = _edge(
        source=alice, target=carol, fact='live wording', hyperedge_uuid=h, valid_at=_dt(2020)
    )

    resolve_hyperedges([live], [retired])

    # The live member is the representative, so its wording wins; the start is still
    # the widest (the retired member's 2019).
    assert live.fact == 'live wording'
    assert retired.fact == 'live wording'
    assert live.valid_at == _dt(2019) and retired.valid_at == _dt(2019)
    assert live.invalid_at == _dt(2021) and retired.invalid_at == _dt(2021)
    # Shared expiry is the latest non-null (retired's 2021), not a fresh now.
    assert retired.expired_at == _dt(2021)
    assert live.expired_at == _dt(2021)


def test_apply_leaves_each_member_reference_time_alone():
    """reference_time records the episode a projection came from, so it is per member.
    The fact and the window are still shared."""
    h = str(uuid4())
    a = str(uuid4())
    source = _edge(
        source=a, target=str(uuid4()), fact='canonical', hyperedge_uuid=h, valid_at=_dt(2019)
    )
    source.reference_time = _dt(2019, 6, 1)
    other = _edge(source=a, target=str(uuid4()), fact='other', hyperedge_uuid=h, valid_at=_dt(2020))
    other.reference_time = _dt(2020, 6, 1)

    Hyperedge(uuid=h, members=[source, other]).apply()

    # Each member keeps the episode it was extracted from.
    assert source.reference_time == _dt(2019, 6, 1)
    assert other.reference_time == _dt(2020, 6, 1)
    # The fact and the window are still the group's.
    assert other.fact == 'canonical'
    assert source.valid_at == other.valid_at == _dt(2019)


def test_apply_nulls_fact_embedding_only_when_the_fact_changes():
    h = str(uuid4())
    a, b, c = str(uuid4()), str(uuid4()), str(uuid4())
    source = _edge(source=a, target=b, fact='canonical', hyperedge_uuid=h, valid_at=_dt(2020))
    source.fact_embedding = [0.1, 0.2]
    changed = _edge(source=a, target=c, fact='different', hyperedge_uuid=h, valid_at=_dt(2021))
    changed.fact_embedding = [0.3, 0.4]

    Hyperedge(uuid=h, members=[source, changed]).apply()

    # source is the representative, so it keeps its fact and its embedding
    assert source.fact == 'canonical'
    assert source.fact_embedding == [0.1, 0.2]
    # changed adopts the canonical fact, so its stale embedding is cleared
    assert changed.fact == 'canonical'
    assert changed.fact_embedding is None


def test_widest_window_handles_naive_and_aware_datetimes():
    """min/max over mixed tz-naive and tz-aware dates works and yields aware results."""
    h = str(uuid4())
    a, b = str(uuid4()), str(uuid4())
    fact = 'Alice introduced Bob to Carol.'
    naive_early = _edge(
        source=a,
        target=b,
        fact=fact,
        hyperedge_uuid=h,
        valid_at=datetime(2019, 1, 1),  # tz-naive
        invalid_at=_dt(2020),  # tz-aware
    )
    aware_late = _edge(
        source=a,
        target=str(uuid4()),
        fact=fact,
        hyperedge_uuid=h,
        valid_at=_dt(2021),
        invalid_at=datetime(2022, 1, 1),  # tz-naive
    )

    resolve_hyperedges([naive_early, aware_late], [])

    for member in (naive_early, aware_late):
        assert member.valid_at == _dt(2019)  # earliest start, normalized to UTC
        assert member.invalid_at == _dt(2022)  # latest end, normalized to UTC
        assert member.valid_at.tzinfo is not None
        assert member.invalid_at.tzinfo is not None


def test_resolve_hyperedges_retracts_a_start_only_member_when_a_sibling_ended():
    """Retraction wins: an end-only member ends the fact for a start-only sibling. The
    sibling is expired at the group end, and the window never inverts (no impossible
    valid_at > invalid_at)."""
    h = str(uuid4())
    a = str(uuid4())
    end_only = _edge(
        source=a, target=str(uuid4()), fact='f', hyperedge_uuid=h, invalid_at=_dt(2020)
    )
    start_only = _edge(
        source=a, target=str(uuid4()), fact='f', hyperedge_uuid=h, valid_at=_dt(2024)
    )

    resolve_hyperedges([start_only], [end_only])

    for member in (end_only, start_only):
        assert member.invalid_at == _dt(2020)  # retraction propagates
        assert member.valid_at == _dt(2020)  # clamped, so valid_at <= invalid_at
        assert member.valid_at <= member.invalid_at
    # The active sibling is retracted, not left open, and shares the group's expiry.
    assert start_only.expired_at is not None
    assert start_only.expired_at == end_only.expired_at


def test_apply_does_not_expire_but_apply_expiry_does():
    """apply() aligns fact/window only; apply_expiry stamps expired_at separately."""
    h = str(uuid4())
    a, b = str(uuid4()), str(uuid4())
    ended = _edge(
        source=a, target=b, fact='f', hyperedge_uuid=h, valid_at=_dt(2020), invalid_at=_dt(2021)
    )
    sibling = _edge(source=a, target=str(uuid4()), fact='f', hyperedge_uuid=h, valid_at=_dt(2020))
    group = Hyperedge(uuid=h, members=[ended, sibling])

    group.apply()

    # Both members now carry the end, but apply() must not have expired either of them.
    assert ended.invalid_at == _dt(2021)
    assert sibling.invalid_at == _dt(2021)
    assert ended.expired_at is None
    assert sibling.expired_at is None

    now = _dt(2026)
    group.apply_expiry(now)

    assert ended.expired_at == now
    assert sibling.expired_at == now


def test_apply_expiry_stamps_one_shared_now_when_the_group_ends_for_the_first_time():
    """A group that has ended but has no expired_at yet gets one shared now."""
    h = str(uuid4())
    a = str(uuid4())
    m1 = _edge(
        source=a,
        target=str(uuid4()),
        fact='f',
        hyperedge_uuid=h,
        valid_at=_dt(2020),
        invalid_at=_dt(2021),
    )
    m2 = _edge(source=a, target=str(uuid4()), fact='f', hyperedge_uuid=h, valid_at=_dt(2020))

    resolve_hyperedges([m1, m2], [])

    assert m1.expired_at is not None
    assert m1.expired_at == m2.expired_at


# --- expiry cascade --------------------------------------------------------


def test_resolve_hyperedges_leaves_an_all_open_group_open():
    h = str(uuid4())
    a, b, c = str(uuid4()), str(uuid4()), str(uuid4())
    fact = 'Alice introduced Bob to Carol.'
    m1 = _edge(source=a, target=b, fact=fact, hyperedge_uuid=h, valid_at=_dt(2020))
    m2 = _edge(source=a, target=c, fact=fact, hyperedge_uuid=h, valid_at=_dt(2021))
    members = [m1, m2]

    resolve_hyperedges(members, [])

    assert {member.valid_at for member in members} == {_dt(2020)}
    assert {member.invalid_at for member in members} == {None}
    assert {member.expired_at for member in members} == {None}


# --- isolation between groups and from untagged edges ----------------------


def test_resolve_hyperedges_keeps_two_groups_independent():
    """Two facts ending in one batch must not share a window or a fact."""
    first, second = str(uuid4()), str(uuid4())
    a = str(uuid4())
    f1 = _edge(
        source=a,
        target=str(uuid4()),
        fact='F',
        hyperedge_uuid=first,
        valid_at=_dt(2019),
        invalid_at=_dt(2020),
    )
    f2 = _edge(source=a, target=str(uuid4()), fact='F', hyperedge_uuid=first, valid_at=_dt(2019))
    s1 = _edge(
        source=a,
        target=str(uuid4()),
        fact='S',
        hyperedge_uuid=second,
        valid_at=_dt(2019),
        invalid_at=_dt(2021),
    )
    s2 = _edge(source=a, target=str(uuid4()), fact='S', hyperedge_uuid=second, valid_at=_dt(2019))

    resolve_hyperedges([f1, f2, s1, s2], [])

    assert f2.invalid_at == _dt(2020)
    assert s2.invalid_at == _dt(2021)
    assert f2.fact == 'F'
    assert s2.fact == 'S'


def test_resolve_hyperedges_leaves_untagged_edges_alone():
    h = str(uuid4())
    a, b, c, d = (str(uuid4()) for _ in range(4))
    fact = 'Alice introduced Bob to Carol.'
    m1 = _edge(source=a, target=b, fact=fact, hyperedge_uuid=h, valid_at=_dt(2020))
    m2 = _edge(source=a, target=c, fact=fact, hyperedge_uuid=h, valid_at=_dt(2021))
    untagged = _edge(source=a, target=d, fact='Alice works at Acme.', valid_at=_dt(2019))
    prior_graph_edge = _edge(
        source=b, target=c, fact='old', invalid_at=_dt(2020), expired_at=_dt(2020)
    )

    resolved_edges = [m1, m2, untagged]
    invalidated_edges = [prior_graph_edge]

    assert resolve_hyperedges(resolved_edges, invalidated_edges) is None

    assert untagged.fact == 'Alice works at Acme.'
    assert untagged.valid_at == _dt(2019)
    assert untagged.invalid_at is None
    assert prior_graph_edge.fact == 'old'
    assert resolved_edges == [m1, m2, untagged]
    assert invalidated_edges == [prior_graph_edge]


def _with_uuid(edge: EntityEdge, uuid: str) -> EntityEdge:
    """Pin an edge uuid so a duplicate set's canonical is the one the test wants."""
    edge.uuid = uuid
    return edge


def test_absorb_canonical_edges_stamps_an_ungrouped_canonical():
    """The lowest uuid wins dedupe; it must still carry the group it displaced."""
    group_uuid = str(uuid4())
    canonical = _with_uuid(
        _edge(source='alice', target='bob', fact='Alice introduced Bob'), 'edge-a'
    )
    member = _with_uuid(
        _edge(
            source='alice',
            target='bob',
            fact='Alice introduced Bob to Carol',
            hyperedge_uuid=group_uuid,
        ),
        'edge-b',
    )
    sibling = _with_uuid(
        _edge(
            source='alice',
            target='carol',
            fact='Alice introduced Bob to Carol',
            hyperedge_uuid=group_uuid,
        ),
        'edge-c',
    )
    canonical.fact_embedding = [0.1, 0.2]

    absorb_canonical_edges(
        {'edge-a': 'edge-a', 'edge-b': 'edge-a'},
        {'edge-a': canonical, 'edge-b': member, 'edge-c': sibling},
    )

    assert canonical.hyperedge_uuid == group_uuid
    assert canonical.fact == 'Alice introduced Bob to Carol'
    assert canonical.fact_embedding is None

    # The group now has two edges over three nodes, so it survives resolution.
    resolve_hyperedges([canonical, sibling], [], {group_uuid})
    assert canonical.hyperedge_uuid == group_uuid
    assert sibling.hyperedge_uuid == group_uuid


def test_absorb_canonical_edges_is_inert_without_stamps():
    """Accounts without hyperedges see no change at all."""
    canonical = _with_uuid(_edge(source='alice', target='bob', fact='kept'), 'edge-a')
    member = _with_uuid(_edge(source='alice', target='bob', fact='dropped'), 'edge-b')
    canonical.fact_embedding = [0.1, 0.2]

    absorb_canonical_edges(
        {'edge-a': 'edge-a', 'edge-b': 'edge-a'},
        {'edge-a': canonical, 'edge-b': member},
    )

    assert canonical.hyperedge_uuid is None
    assert canonical.fact == 'kept'
    assert canonical.fact_embedding == [0.1, 0.2]
