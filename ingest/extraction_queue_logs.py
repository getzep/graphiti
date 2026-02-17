#!/usr/bin/env python3
"""Parse Graphiti MCP queue-service log lines into structured lifecycle events."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Optional


_QUEUE_SERVICE_LINE_RE = re.compile(
    r"^(?:[^|]+\|\s*)?"
    r"(?P<ts>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})(?:[.,]\d+)?"
    r"\s+-\s+services\.queue_service\s+-\s+(?P<level>[A-Z]+)\s+-\s+(?P<msg>.+)$"
)

_PROCESSING_RE = re.compile(
    r"^Processing episode (?P<episode>[^\s]+) for group (?P<group>[^\s]+)$"
)
_SUCCESS_RE = re.compile(
    r"^Successfully processed episode (?P<episode>[^\s]+) for group (?P<group>[^\s]+)$"
)
_FAILED_RE = re.compile(
    r"^Failed to process episode (?P<episode>[^\s]+) for group (?P<group>[^\s]+): (?P<reason>.+)$"
)


@dataclass(frozen=True)
class QueueServiceEvent:
    """Parsed extraction event from a Graphiti queue-service log line."""

    timestamp: str  # ISO-8601 Z, second precision
    event_type: str  # processing | succeeded | failed
    group_id: str
    episode_uuid: Optional[str]
    failure_reason: Optional[str]
    raw_line: str


def _normalize_ts(ts_text: str) -> str:
    raw = str(ts_text or "").strip().replace(" ", "T")
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _normalize_episode(token: Optional[str]) -> Optional[str]:
    t = str(token or "").strip()
    if not t:
        return None
    if t.lower() in {"none", "null", "nil"}:
        return None
    return t


def parse_queue_service_line(line: str) -> Optional[QueueServiceEvent]:
    """Parse one queue-service line.

    Returns None for non-queue-service lines or unsupported message formats.
    """

    text = str(line or "").rstrip("\n")
    m = _QUEUE_SERVICE_LINE_RE.match(text)
    if not m:
        return None

    ts = _normalize_ts(m.group("ts"))
    msg = m.group("msg").strip()

    m_proc = _PROCESSING_RE.match(msg)
    if m_proc:
        return QueueServiceEvent(
            timestamp=ts,
            event_type="processing",
            group_id=m_proc.group("group"),
            episode_uuid=_normalize_episode(m_proc.group("episode")),
            failure_reason=None,
            raw_line=text,
        )

    m_ok = _SUCCESS_RE.match(msg)
    if m_ok:
        return QueueServiceEvent(
            timestamp=ts,
            event_type="succeeded",
            group_id=m_ok.group("group"),
            episode_uuid=_normalize_episode(m_ok.group("episode")),
            failure_reason=None,
            raw_line=text,
        )

    m_fail = _FAILED_RE.match(msg)
    if m_fail:
        return QueueServiceEvent(
            timestamp=ts,
            event_type="failed",
            group_id=m_fail.group("group"),
            episode_uuid=_normalize_episode(m_fail.group("episode")),
            failure_reason=(m_fail.group("reason") or "").strip() or None,
            raw_line=text,
        )

    return None


def parse_queue_service_lines(lines: Iterable[str]) -> list[QueueServiceEvent]:
    """Parse many lines, preserving input order for recognized events."""

    out: list[QueueServiceEvent] = []
    for line in lines:
        ev = parse_queue_service_line(line)
        if ev is not None:
            out.append(ev)
    return out
