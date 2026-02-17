#!/usr/bin/env python3
"""Parse OpenClaw session transcripts (*.jsonl) into v1 evidence chunks.

Implements deterministic, larger chunking per:
- tools/graphiti/prd/EVIDENCE-SCHEMA-CHUNKING-SPEC.md
- tools/graphiti/prd/IDENTITY-SCOPE-SPEC.md

Key behaviors:
- Chunk by *message-events* (not turns): window=200, overlap=50 (step=150)
- Include only user + assistant messages
- Exclude assistant "thinking" blocks and clawdbot delivery-mirror duplicates
- Emit stable: source_key, chunk_key, start_id/end_id
- Emit scope (private|group) and stable speaker_ids when possible (tg:<id>)

Output (default): tools/graphiti/evidence/sessions_v1/<agent_id>/all_evidence.json

Usage:
  cd tools/graphiti
  python3 ingest/parse_sessions_v1.py --agent main
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from common import extract_message_id, write_evidence_batch


WINDOW_EVENTS = 200
OVERLAP_EVENTS = 50
STEP_EVENTS = WINDOW_EVENTS - OVERLAP_EVENTS

INGESTER_VERSION = "sessions_v1@2026-02-03"
CHUNKER_VERSION = f"events_win{WINDOW_EVENTS}_step{STEP_EVENTS}_overlap{OVERLAP_EVENTS}"


def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _iso_or_none(dt: Optional[datetime]) -> Optional[str]:
    if not dt:
        return None
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_iso(ts: str | None) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def read_jsonl(file_path: Path) -> Iterator[dict[str, Any]]:
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def extract_text_content(content: list[dict] | str | None, *, include_thinking: bool = False) -> str:
    """Extract text from message.content.

    v1 rules:
    - exclude thinking by default
    - include text only (ignore other types)
    """

    if not content:
        return ""

    if isinstance(content, str):
        return content

    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        t = item.get("type")
        if t == "text":
            txt = item.get("text") or ""
            if txt:
                parts.append(txt)
        elif include_thinking and t == "thinking":
            thinking = item.get("thinking") or ""
            if thinking:
                parts.append(f"[thinking] {thinking}")

    return "\n".join(parts).strip()


# Telegram header examples:
#   [Telegram Yuan Han Li | Personal (@yuan_han_li) id:1439681712 +8h 2026-01-24 22:52 UTC] hey
#   [Telegram Yuan Han & YHL, Archibald Lite id:-5169861041 +5s 2026-01-26 22:19 UTC] Yuan Han Li | Personal (1439681712): ...
_TELEGRAM_HEADER_RE = re.compile(
    r"^\[Telegram\s+(?P<title>.+?)\s*(?:\|\s*(?P<chat_type>[^\(\]]+?)\s*(?:\((?P<username>@\w+)\))?\s*)?id:(?P<chat_id>-?\d+)[^\]]*\]\s*(?P<body>.*)$",
    re.DOTALL,
)

# Inside group chats, user lines often look like:
#   Yuan Han Li | Personal (1439681712): @archibaldbutlerlitebot ...
_GROUP_SPEAKER_LINE_RE = re.compile(
    r"^(?P<name>.+?)\s*\|\s*(?P<label>[^\(\n]+?)\s*\((?P<user_id>-?\d+)\)\s*:\s*(?P<msg>.*)$",
    re.DOTALL,
)

# Timestamp embedded in Telegram header:
#   ... 2026-01-26 22:19 UTC]
_TELEGRAM_UTC_TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2})\s+UTC")


@dataclass(frozen=True)
class MsgEvent:
    event_id: str
    timestamp: Optional[datetime]
    role: str  # user|assistant
    speaker_id: str  # tg:<id>|assistant:archibald|unknown
    speaker_label: str
    text: str
    tg_message_id: Optional[str] = None
    telegram_chat_id: Optional[int] = None


def _split_telegram_blocks(text: str) -> list[str]:
    """Split a user event that may contain multiple telegram messages (queued batches)."""
    if "[Telegram" not in text:
        return [text]

    # Keep only segments that start with [Telegram ...]
    segs = [s.strip() for s in re.split(r"(?=\[Telegram\s)", text) if s.strip()]
    return [s for s in segs if s.lstrip().startswith("[Telegram")]


def _parse_telegram_header(block: str) -> tuple[dict[str, Any], str]:
    """Parse [Telegram ...] header, returning (meta, body)."""

    m = _TELEGRAM_HEADER_RE.match(block.strip())
    if not m:
        return {}, block.strip()

    chat_id = int(m.group("chat_id"))
    chat_type = (m.group("chat_type") or "").strip() or None
    username = (m.group("username") or "").strip() or None
    title = (m.group("title") or "").strip()
    body = (m.group("body") or "").strip()

    ts_match = _TELEGRAM_UTC_TS_RE.search(block)
    header_ts: Optional[datetime] = None
    if ts_match:
        try:
            header_ts = datetime.strptime(
                f"{ts_match.group(1)} {ts_match.group(2)}",
                "%Y-%m-%d %H:%M",
            ).replace(tzinfo=timezone.utc)
        except Exception:
            header_ts = None

    return (
        {
            "chat_id": chat_id,
            "chat_type": chat_type,
            "username": username,
            "title": title,
            "header_timestamp": header_ts,
        },
        body,
    )


def _infer_scope_from_chat(chat_id: Optional[int], chat_type: Optional[str]) -> Optional[str]:
    if chat_id is None and not chat_type:
        return None

    if chat_id is not None and chat_id < 0:
        return "group"

    if chat_type:
        s = chat_type.strip().lower()
        if "group" in s or "supergroup" in s:
            return "group"
        if "personal" in s or "private" in s or "dm" in s:
            return "private"

    if chat_id is not None and chat_id > 0:
        return "private"

    return None


def _parse_user_block_as_telegram(
    block: str,
) -> list[tuple[str, str, Optional[int], Optional[str], str, Optional[datetime]]]:
    """Parse a single Telegram message block starting with a [Telegram ...] header.

    Returns list of:
      (speaker_id, speaker_label, telegram_chat_id, telegram_chat_type, clean_text, ts)
    """

    meta, body = _parse_telegram_header(block)
    chat_id: Optional[int] = meta.get("chat_id")
    chat_type: Optional[str] = meta.get("chat_type")
    ts: Optional[datetime] = meta.get("header_timestamp")

    # Remove trailing Telegram message_id marker, if present.
    body = re.sub(r"\s*\[message_id:\s*\d+\]\s*$", "", body).strip()

    gm = _GROUP_SPEAKER_LINE_RE.match(body)
    if gm:
        uid = gm.group("user_id")
        name = (gm.group("name") or "").strip()
        msg = (gm.group("msg") or "").strip()
        speaker_id = f"tg:{uid}" if uid else "unknown"
        speaker_label = name or speaker_id
        return [(speaker_id, speaker_label, chat_id, chat_type, msg, ts)]

    if chat_id is not None and chat_id > 0:
        label = (meta.get("title") or "User").strip() if isinstance(meta, dict) else "User"
        return [(f"tg:{chat_id}", label or "User", chat_id, chat_type, body, ts)]

    return [("unknown", "User", chat_id, chat_type, body, ts)]


def _build_msg_events(entries: list[dict[str, Any]]) -> tuple[list[MsgEvent], str]:
    events: list[MsgEvent] = []
    inferred_scopes: set[str] = set()

    for entry in entries:
        if entry.get("type") != "message":
            continue

        msg = entry.get("message") or {}
        role = msg.get("role")
        if role not in ("user", "assistant"):
            continue

        # Exclude clawdbot delivery-mirror duplicates/system banners.
        provider = msg.get("provider") or entry.get("provider")
        model = msg.get("model") or entry.get("model")
        if role == "assistant" and (provider == "clawdbot" or model == "delivery-mirror"):
            continue

        entry_id = str(entry.get("id") or "")
        entry_ts = _parse_iso(entry.get("timestamp"))

        if role == "assistant":
            txt = extract_text_content(msg.get("content"), include_thinking=False)
            if not txt.strip():
                continue
            events.append(
                MsgEvent(
                    event_id=entry_id,
                    timestamp=entry_ts,
                    role=role,
                    speaker_id="assistant:archibald",
                    speaker_label="Archibald",
                    text=txt.strip(),
                )
            )
            continue

        # role == user
        raw = extract_text_content(msg.get("content"), include_thinking=False)
        if not raw.strip():
            continue

        blocks = _split_telegram_blocks(raw)
        if blocks and blocks[0].lstrip().startswith("[Telegram"):
            for bi, block in enumerate(blocks):
                tg_mid = extract_message_id(block) or extract_message_id(raw)
                parsed = _parse_user_block_as_telegram(block)
                for pi, (speaker_id, speaker_label, chat_id, chat_type, clean_text, header_ts) in enumerate(parsed):
                    if not clean_text.strip():
                        continue

                    scope = _infer_scope_from_chat(chat_id, chat_type)
                    if scope:
                        inferred_scopes.add(scope)

                    seg_id = entry_id if (len(blocks) == 1 and len(parsed) == 1) else f"{entry_id}#{bi}.{pi}"
                    ts = header_ts or entry_ts
                    events.append(
                        MsgEvent(
                            event_id=seg_id,
                            timestamp=ts,
                            role=role,
                            speaker_id=speaker_id,
                            speaker_label=speaker_label,
                            text=clean_text.strip(),
                            tg_message_id=tg_mid,
                            telegram_chat_id=chat_id,
                        )
                    )
            continue

        # Generic non-telegram user message
        tg_mid = extract_message_id(raw)
        events.append(
            MsgEvent(
                event_id=entry_id,
                timestamp=entry_ts,
                role=role,
                speaker_id="unknown",
                speaker_label="User",
                text=raw.strip(),
                tg_message_id=tg_mid,
            )
        )

    scope = "group" if "group" in inferred_scopes else "private"

    # Canonical order: by timestamp, then file order.
    def _sort_key(i_ev: tuple[int, MsgEvent]):
        idx, ev = i_ev
        ts = ev.timestamp.timestamp() if ev.timestamp else float("inf")
        return (ts, idx)

    events = [ev for _, ev in sorted(list(enumerate(events)), key=_sort_key)]
    return events, scope


def _format_event(ev: MsgEvent) -> str:
    ts = _iso_or_none(ev.timestamp) or ""
    mid = f" tg_mid={ev.tg_message_id}" if ev.tg_message_id else ""
    return f"[{ts}] {ev.speaker_label} ({ev.speaker_id}) id={ev.event_id}{mid}: {ev.text}".strip()


def _extract_topic_suffix(file_path: Path) -> str:
    """Extract topic suffix from filename like 'uuid-topic-18.jsonl' -> ':topic-18'."""
    m = re.search(r"-topic-(\d+)(?:\.jsonl)?$", file_path.name)
    return f":topic-{m.group(1)}" if m else ""


def parse_session_file_v1(file_path: Path, agent_id: str) -> list[dict[str, Any]]:
    entries = list(read_jsonl(file_path))

    session_meta = next((e for e in entries if e.get("type") == "session"), {})
    session_uuid = (session_meta.get("id") or file_path.stem)

    events, scope = _build_msg_events(entries)
    if not events:
        return []

    # Append topic suffix to disambiguate topic files that share the same session UUID.
    topic_suffix = _extract_topic_suffix(file_path)
    source_key = f"session:{agent_id}:{session_uuid}{topic_suffix}"

    # Extract topic_id for source metadata (if present).
    topic_id = topic_suffix.replace(":topic-", "") if topic_suffix else None

    chunks: list[dict[str, Any]] = []
    start = 0
    chunk_index = 0

    while start < len(events):
        end = min(start + WINDOW_EVENTS, len(events))
        chunk_events = events[start:end]
        if not chunk_events:
            break

        start_id = chunk_events[0].event_id
        end_id = chunk_events[-1].event_id

        sess_ts = _parse_iso(session_meta.get("timestamp"))
        ts_start = chunk_events[0].timestamp or sess_ts
        ts_end = chunk_events[-1].timestamp or sess_ts

        chunk_key = f"{source_key}:c{chunk_index}"

        content = "\n\n".join([_format_event(ev) for ev in chunk_events]).strip() + "\n"
        content_hash = _sha256_hex(content)
        evidence_id = _sha256_hex(f"{chunk_key}:{content_hash}")[:32]

        evidence: dict[str, Any] = {
            "evidence_id": evidence_id,
            "source_key": source_key,
            "chunk_key": chunk_key,
            "source": {
                "type": "session",
                "path": str(file_path),
                "agent_id": agent_id,
                "session_id": session_uuid,
                "topic_id": topic_id,
                "session_version": session_meta.get("version"),
            },
            "scope": scope,
            "timestamp_range": {"start": _iso_or_none(ts_start), "end": _iso_or_none(ts_end)},
            "start_id": start_id,
            "end_id": end_id,
            "speaker_ids": sorted({ev.speaker_id for ev in chunk_events if ev.speaker_id}),
            "content": content,
            "content_hash": content_hash,
            "content_type": "conversation",
            "message_ids": [ev.event_id for ev in chunk_events],
            "ingester_version": INGESTER_VERSION,
            "chunker_version": CHUNKER_VERSION,
        }

        # Drop nulls for tidiness
        evidence = {k: v for k, v in evidence.items() if v is not None}
        if isinstance(evidence.get("source"), dict):
            evidence["source"] = {k: v for k, v in evidence["source"].items() if v is not None}

        chunks.append(evidence)

        chunk_index += 1
        start = start + STEP_EVENTS

    return chunks


def find_session_dirs(clawdbot_dir: Path) -> list[tuple[str, Path]]:
    agents_dir = clawdbot_dir / "agents"
    if not agents_dir.exists():
        return []

    results: list[tuple[str, Path]] = []
    for agent_dir in agents_dir.iterdir():
        if not agent_dir.is_dir():
            continue
        sessions_dir = agent_dir / "sessions"
        if sessions_dir.exists():
            results.append((agent_dir.name, sessions_dir))
    return results


def parse_all_sessions_v1(
    sessions_base: Path,
    output_dir: Path,
    agent_filter: str | None = None,
) -> dict[str, Any]:
    agent_dirs = find_session_dirs(sessions_base)

    if agent_filter:
        agent_dirs = [(a, p) for a, p in agent_dirs if a == agent_filter]

    stats: dict[str, Any] = {
        "agents_processed": 0,
        "sessions_processed": 0,
        "evidence_created": 0,
        "errors": [],
        "window_events": WINDOW_EVENTS,
        "overlap_events": OVERLAP_EVENTS,
    }

    if agent_filter and not agent_dirs:
        out_path = output_dir / "sessions_v1" / agent_filter / "all_evidence.json"
        if not out_path.exists():
            write_evidence_batch([], out_path)
        return stats

    for agent_id, sessions_dir in agent_dirs:
        session_files = sorted(sessions_dir.glob("*.jsonl"))
        agent_evidence: list[dict[str, Any]] = []

        for file_path in session_files:
            try:
                chunks = parse_session_file_v1(file_path, agent_id)
                agent_evidence.extend(chunks)
                stats["sessions_processed"] += 1
                stats["evidence_created"] += len(chunks)
            except Exception as e:
                stats["errors"].append({"file": str(file_path), "error": str(e)})

        out_path = output_dir / "sessions_v1" / agent_id / "all_evidence.json"
        if agent_evidence:
            write_evidence_batch(agent_evidence, out_path)
        elif not out_path.exists():
            write_evidence_batch([], out_path)

        stats["agents_processed"] += 1

    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Parse session transcripts into v1 evidence chunks")
    ap.add_argument(
        "--sessions-dir",
        type=Path,
        default=Path.home() / ".clawdbot",
        help="Base path to .clawdbot directory (expects agents/*/sessions/*.jsonl)",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent / "evidence",
        help="Output directory for evidence files (default: tools/graphiti/evidence)",
    )
    ap.add_argument(
        "--agent",
        type=str,
        default=None,
        help="Filter to specific agent (e.g., 'main', 'group')",
    )
    args = ap.parse_args()

    if not args.sessions_dir.exists():
        raise SystemExit(f"Sessions directory not found: {args.sessions_dir}")

    stats = parse_all_sessions_v1(args.sessions_dir, args.output, args.agent)

    print(json.dumps({k: stats[k] for k in ["agents_processed", "sessions_processed", "evidence_created"]}, indent=2))
    if stats.get("errors"):
        print(f"Errors: {len(stats['errors'])}")
        for err in stats["errors"][:5]:
            print(f"  - {err['file']}: {err['error']}")

    print(f"\nOutput: {args.output / 'sessions_v1'}")


if __name__ == "__main__":
    main()
