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

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field

from graphiti_core.errors import StaleCommitError, UnitExpiredError

STATUS_ADMITTING = 'admitting'
STATUS_ADMITTED = 'admitted'
STATUS_ANALYZED = 'analyzed'
STATUS_APPLYING = 'applying'
STATUS_COMMITTED = 'committed'
STATUS_FAILED = 'failed'
STATUS_APPLY_FAILED = 'apply_failed'

# For the order of commits, these statuses release the position of a unit.
TERMINAL_STATUSES = frozenset({STATUS_COMMITTED, STATUS_FAILED, STATUS_APPLY_FAILED})
FAILED_STATUSES = frozenset({STATUS_FAILED, STATUS_APPLY_FAILED})

REASON_APPLY_FAILED = 'apply_failed'
REASON_EXPIRED = 'expired'


@dataclass(frozen=True)
class Admission:
    """The position of a unit.

    `existing` is True when the store already had a live unit for the `unit_id`. The
    caller then has no admit lease, and it must not write the episodes again.
    `previous_seq` is the position of an earlier unit of the `unit_id` that failed.
    """

    seq: int
    existing: bool = False
    previous_seq: int | None = None


@dataclass(frozen=True)
class AnalysisStart:
    snapshot_seq: int
    started_at: float


@dataclass(frozen=True)
class CommitTurn:
    """The unit has the commit turn. `final_payload` has a value when a final change set is stored."""

    fence: int
    final_payload: bytes | None = None


@dataclass(frozen=True)
class CommitWait:
    """An earlier unit is in progress."""


@dataclass(frozen=True)
class CommitRecover:
    """An earlier unit stopped after it stored its final change set. The caller applies it."""

    seq: int
    fence: int
    final_payload: bytes


@dataclass(frozen=True)
class DeltaUnit:
    seq: int
    status: str
    payload: bytes


def is_released(status: str, deadline: float, now: float) -> bool:
    """Return True when a unit does not keep its `unit_id` for a retry of the admission."""
    if status in FAILED_STATUSES:
        return True
    if status in (STATUS_APPLYING, STATUS_COMMITTED):
        return False
    return deadline <= now


def is_stale_heartbeat(status: str, unit_fence: int, fence: int | None) -> bool:
    """Return True when a heartbeat with `fence` must not set the deadline of the unit."""
    if fence is None:
        return status == STATUS_APPLYING
    return fence != unit_fence


class IngestionStateStore(ABC):
    """The state of concurrent ingestion: the sequence, the change sets, and the enrichment order.

    Each method is atomic. A store for more than one host must compare deadlines with
    the clock of the store. The store does not read a payload.
    """

    @abstractmethod
    async def begin_admit(self, group_id: str, unit_id: str, ttl: float) -> Admission | None:
        """Give the admit lease and the next sequence number to a unit.

        For a `unit_id` that has a live unit, the call returns the `seq` of that unit
        with `existing=True`. It gives no lease, and it changes no state. The caller
        must then write nothing and call no method that changes the state of that
        unit: a different attempt has the unit. The Admit phase raises
        `UnitInProgressError` for this result. The host can call `begin_admit` again
        after the lease time.

        A unit is not live when its status is `failed` or `apply_failed`, or when its
        deadline is in the past and it has no stored final change set. The call marks
        such an expired unit `failed`. The `unit_id` then gets a new sequence number,
        and `previous_seq` is the `seq` of the unit that is not live.

        Returns None when a different unit has the lease and its deadline is in the
        future.
        """

    @abstractmethod
    async def end_admit(self, group_id: str, seq: int) -> None:
        """Set the status `admitted` and release the admit lease."""

    @abstractmethod
    async def start_analysis(self, group_id: str, seq: int) -> AnalysisStart:
        """Store and return the watermark and the current time."""

    @abstractmethod
    async def save_change_set(self, group_id: str, seq: int, payload: bytes) -> None:
        """Store the payload of the analysis and set the status `analyzed`.

        The call is permitted only for a unit with the status `admitted`. For each other
        status it raises `StaleCommitError`, and the stored payload stays.
        """

    @abstractmethod
    async def heartbeat(
        self, group_id: str, seq: int, ttl: float, fence: int | None = None
    ) -> None:
        """Set the deadline of a unit to the current time plus `ttl`.

        `fence` is the fence of the commit turn of the caller, or None for a caller
        with no turn. A `fence` that is not the fence of the unit raises
        `StaleCommitError`. For a unit with the status `applying`, only the attempt
        with the turn can set the deadline: a call with no `fence` also raises
        `StaleCommitError`.
        """

    @abstractmethod
    async def begin_commit(
        self, group_id: str, seq: int, ttl: float
    ) -> CommitTurn | CommitWait | CommitRecover:
        """Give the commit turn to a unit when each earlier unit is committed or failed.

        Returns `CommitWait` also when the unit has the status `applying` and a deadline
        in the future. A different attempt then applies its final change set.
        """

    @abstractmethod
    async def load_delta(
        self, group_id: str, after_seq: int, before_seq: int, since: float
    ) -> list[DeltaUnit]:
        """Return the final change sets of the `committed` and `apply_failed` units.

        A unit is in the result when its `seq` is below `before_seq`, and its `seq` is
        above `after_seq` or its end time is at or after `since`. The order is `seq` order.
        """

    @abstractmethod
    async def save_final(self, group_id: str, seq: int, fence: int, payload: bytes) -> None:
        """Replace the payload with the final change set and set the status `applying`."""

    @abstractmethod
    async def finish_commit(
        self,
        group_id: str,
        seq: int,
        fence: int,
        node_uuids: list[str],
        retain_seconds: float,
    ) -> None:
        """Set the status `committed`, register the nodes for enrichment, move the watermark."""

    @abstractmethod
    async def fail(self, group_id: str, seq: int, reason: str, fence: int | None = None) -> None:
        """Release the position of a unit.

        The call is not permitted for the status `applying`, with one exception: the
        reason `apply_failed`, which needs the current fence and keeps the payload.
        """

    @abstractmethod
    async def ready_enrich(self, group_id: str, seq: int) -> list[str]:
        """Return the nodes of the unit for which no earlier unit has an enrichment entry."""

    @abstractmethod
    async def finish_enrich(self, group_id: str, seq: int, node_uuids: list[str]) -> None:
        """Delete the enrichment entries of the unit for the node UUIDs."""

    @abstractmethod
    async def close(self) -> None:
        """Release the resources of the store."""


@dataclass
class _Unit:
    seq: int
    unit_id: str
    status: str
    deadline: float
    snapshot_seq: int | None = None
    started_at: float | None = None
    ended_at: float | None = None
    fence: int = 0
    fail_reason: str | None = None
    payload: bytes | None = None


@dataclass
class _Group:
    next_seq: int = 1
    watermark: int = 0
    admit_seq: int | None = None
    units: dict[int, _Unit] = field(default_factory=dict)
    unit_ids: dict[str, int] = field(default_factory=dict)
    # seq -> node UUIDs with an enrichment entry, in registration order
    enrich: dict[int, list[str]] = field(default_factory=dict)


class InMemoryIngestionStateStore(IngestionStateStore):
    """A state store in process memory.

    No lock is necessary, because each method has no `await` between its read and its
    write. The store is correct only when one process ingests into the graph.
    """

    def __init__(self, clock: Callable[[], float] = time.time):
        self._clock = clock
        self._groups: dict[str, _Group] = {}

    def _group(self, group_id: str) -> _Group:
        group = self._groups.get(group_id)
        if group is None:
            group = _Group()
            self._groups[group_id] = group
        return group

    def _live_unit(self, group_id: str, seq: int, *, applying_exempt: bool = False) -> _Unit:
        unit = self._group(group_id).units.get(seq)
        if unit is None or unit.status in FAILED_STATUSES:
            raise UnitExpiredError(group_id, seq)
        if unit.deadline <= self._clock() and not (
            applying_exempt and unit.status == STATUS_APPLYING
        ):
            raise UnitExpiredError(group_id, seq)
        return unit

    def _advance_watermark(self, group: _Group) -> None:
        while group.watermark + 1 < group.next_seq:
            unit = group.units.get(group.watermark + 1)
            if unit is not None and unit.status not in TERMINAL_STATUSES:
                break
            group.watermark += 1

    def _fail_unit(self, group: _Group, unit: _Unit, reason: str, *, keep_payload: bool) -> None:
        unit.status = STATUS_APPLY_FAILED if keep_payload else STATUS_FAILED
        unit.fail_reason = reason
        unit.ended_at = self._clock()
        if not keep_payload:
            unit.payload = None
        group.enrich.pop(unit.seq, None)
        if group.admit_seq == unit.seq:
            group.admit_seq = None
        self._advance_watermark(group)

    def _delete_old_units(self, group: _Group, retain_seconds: float) -> None:
        now = self._clock()
        limit = group.watermark
        for unit in group.units.values():
            if unit.status not in TERMINAL_STATUSES and unit.snapshot_seq is not None:
                limit = min(limit, unit.snapshot_seq)
        for seq in [seq for seq in group.units if seq <= limit]:
            unit = group.units[seq]
            if (
                unit.status in TERMINAL_STATUSES
                and unit.ended_at is not None
                and unit.ended_at < now - retain_seconds
                and not group.enrich.get(seq)
            ):
                del group.units[seq]
                if group.unit_ids.get(unit.unit_id) == seq:
                    del group.unit_ids[unit.unit_id]
                group.enrich.pop(seq, None)

    async def begin_admit(self, group_id: str, unit_id: str, ttl: float) -> Admission | None:
        group = self._group(group_id)
        now = self._clock()
        previous_seq = group.unit_ids.get(unit_id)
        previous = group.units.get(previous_seq) if previous_seq is not None else None
        if previous is not None:
            if not is_released(previous.status, previous.deadline, now):
                return Admission(seq=previous.seq, existing=True)
            if previous.status not in TERMINAL_STATUSES:
                self._fail_unit(group, previous, REASON_EXPIRED, keep_payload=False)

        if group.admit_seq is not None:
            holder = group.units.get(group.admit_seq)
            if holder is not None and holder.deadline > now:
                return None
            if holder is not None:
                self._fail_unit(group, holder, REASON_EXPIRED, keep_payload=False)
            group.admit_seq = None

        seq = group.next_seq
        group.next_seq += 1
        group.units[seq] = _Unit(
            seq=seq, unit_id=unit_id, status=STATUS_ADMITTING, deadline=now + ttl
        )
        group.unit_ids[unit_id] = seq
        group.admit_seq = seq
        return Admission(seq=seq, previous_seq=previous.seq if previous is not None else None)

    async def end_admit(self, group_id: str, seq: int) -> None:
        group = self._group(group_id)
        unit = self._live_unit(group_id, seq)
        if unit.status == STATUS_ADMITTING:
            unit.status = STATUS_ADMITTED
        if group.admit_seq == seq:
            group.admit_seq = None

    async def start_analysis(self, group_id: str, seq: int) -> AnalysisStart:
        group = self._group(group_id)
        unit = self._live_unit(group_id, seq)
        unit.snapshot_seq = group.watermark
        unit.started_at = self._clock()
        return AnalysisStart(snapshot_seq=unit.snapshot_seq, started_at=unit.started_at)

    async def save_change_set(self, group_id: str, seq: int, payload: bytes) -> None:
        unit = self._live_unit(group_id, seq, applying_exempt=True)
        if unit.status != STATUS_ADMITTED:
            # A second analysis does not replace a stored payload.
            raise StaleCommitError(group_id, seq)
        unit.payload = payload
        unit.status = STATUS_ANALYZED

    async def heartbeat(
        self, group_id: str, seq: int, ttl: float, fence: int | None = None
    ) -> None:
        unit = self._live_unit(group_id, seq, applying_exempt=True)
        if is_stale_heartbeat(unit.status, unit.fence, fence):
            raise StaleCommitError(group_id, seq)
        unit.deadline = self._clock() + ttl

    async def begin_commit(
        self, group_id: str, seq: int, ttl: float
    ) -> CommitTurn | CommitWait | CommitRecover:
        group = self._group(group_id)
        unit = self._live_unit(group_id, seq, applying_exempt=True)
        if unit.status == STATUS_COMMITTED:
            raise StaleCommitError(group_id, seq)

        now = self._clock()
        for earlier_seq in range(group.watermark + 1, seq):
            earlier = group.units.get(earlier_seq)
            if earlier is None or earlier.status in TERMINAL_STATUSES:
                continue
            if earlier.deadline > now:
                return CommitWait()
            if earlier.status == STATUS_APPLYING and earlier.payload is not None:
                earlier.fence += 1
                earlier.deadline = now + ttl
                return CommitRecover(
                    seq=earlier.seq, fence=earlier.fence, final_payload=earlier.payload
                )
            self._fail_unit(group, earlier, REASON_EXPIRED, keep_payload=False)

        if unit.status == STATUS_APPLYING and unit.deadline > now:
            # A different attempt applies the final change set of the unit.
            return CommitWait()
        unit.fence += 1
        unit.deadline = now + ttl
        final_payload = unit.payload if unit.status == STATUS_APPLYING else None
        return CommitTurn(fence=unit.fence, final_payload=final_payload)

    async def load_delta(
        self, group_id: str, after_seq: int, before_seq: int, since: float
    ) -> list[DeltaUnit]:
        group = self._group(group_id)
        result: list[DeltaUnit] = []
        for seq in sorted(group.units):
            unit = group.units[seq]
            if seq >= before_seq or unit.payload is None:
                continue
            if unit.status not in (STATUS_COMMITTED, STATUS_APPLY_FAILED):
                continue
            if seq > after_seq or (unit.ended_at is not None and unit.ended_at >= since):
                result.append(DeltaUnit(seq=seq, status=unit.status, payload=unit.payload))
        return result

    def _commit_unit(self, group_id: str, seq: int, fence: int) -> _Unit:
        unit = self._group(group_id).units.get(seq)
        if unit is None or unit.status in FAILED_STATUSES:
            raise UnitExpiredError(group_id, seq)
        if unit.status == STATUS_COMMITTED or unit.fence != fence:
            raise StaleCommitError(group_id, seq)
        if unit.deadline <= self._clock():
            raise UnitExpiredError(group_id, seq)
        return unit

    async def save_final(self, group_id: str, seq: int, fence: int, payload: bytes) -> None:
        unit = self._commit_unit(group_id, seq, fence)
        unit.payload = payload
        unit.status = STATUS_APPLYING

    async def finish_commit(
        self,
        group_id: str,
        seq: int,
        fence: int,
        node_uuids: list[str],
        retain_seconds: float,
    ) -> None:
        group = self._group(group_id)
        unit = self._commit_unit(group_id, seq, fence)
        unit.status = STATUS_COMMITTED
        unit.ended_at = self._clock()
        if node_uuids:
            group.enrich[seq] = list(dict.fromkeys(node_uuids))
        self._advance_watermark(group)
        self._delete_old_units(group, retain_seconds)

    async def fail(self, group_id: str, seq: int, reason: str, fence: int | None = None) -> None:
        group = self._group(group_id)
        unit = group.units.get(seq)
        if unit is None or unit.status in TERMINAL_STATUSES:
            return
        if unit.status == STATUS_APPLYING:
            if reason != REASON_APPLY_FAILED:
                raise ValueError(
                    'fail is not permitted for a unit with a stored final change set, '
                    f'with the exception of the reason {REASON_APPLY_FAILED!r}'
                )
            if fence is None or fence != unit.fence:
                raise StaleCommitError(group_id, seq)
            self._fail_unit(group, unit, reason, keep_payload=True)
            return
        self._fail_unit(group, unit, reason, keep_payload=False)

    async def ready_enrich(self, group_id: str, seq: int) -> list[str]:
        group = self._group(group_id)
        self._live_unit(group_id, seq)
        now = self._clock()
        for entry_seq in list(group.enrich):
            entry_unit = group.units.get(entry_seq)
            if entry_unit is None or entry_unit.deadline <= now:
                del group.enrich[entry_seq]

        blocked: set[str] = set()
        for entry_seq, node_uuids in group.enrich.items():
            if entry_seq < seq:
                blocked.update(node_uuids)
        return [uuid for uuid in group.enrich.get(seq, []) if uuid not in blocked]

    async def finish_enrich(self, group_id: str, seq: int, node_uuids: list[str]) -> None:
        group = self._group(group_id)
        remaining = [uuid for uuid in group.enrich.get(seq, []) if uuid not in set(node_uuids)]
        if remaining:
            group.enrich[seq] = remaining
        else:
            group.enrich.pop(seq, None)

    async def close(self) -> None:
        self._groups = {}
