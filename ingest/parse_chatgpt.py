#!/usr/bin/env python3
"""Parse ChatGPT export (conversations.json) into evidence documents.

We keep the evidence format consistent with memory + sessions:
- deterministic evidence ids
- chunked conversation text

Excluding Constantine:
- Find the conversation titled "Leadership: Follower Focus"; treat it as the most recent
  Constantine conversation (per Yuan).
- Exclude ALL conversations with create_time <= that cutoff, except explicit allowlisted titles:
  - Enhancing Blockchain Capital Bio
  - Plan for Tomorrow's Conversation
  - Tax Status Change Confirmation

Exclusions are persisted to the ingest registry so future exports keep filtering
even if the original cutoff conversation is no longer present.

Outputs:
- evidence/chatgpt/all_evidence.json
- evidence/chatgpt/filter_report.json (allow/deny lists + counts)

Usage:
  cd tools/graphiti
  python ingest/parse_chatgpt.py
  python ingest/parse_chatgpt.py --persist-exclusions  # Save exclusions to registry
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from common import generate_evidence_id, write_evidence_batch
from registry import get_registry


@dataclass
class Message:
    role: str
    text: str
    create_time: float | None


def _norm_title(s: str) -> str:
    return (s or "").strip().lower().replace("’", "'")


def _extract_message_text(msg: dict) -> str:
    content = (msg.get("content") or {})
    parts = content.get("parts") or []
    if isinstance(parts, list):
        # join text parts; ignore non-strings
        return "\n".join([p for p in parts if isinstance(p, str)]).strip()
    if isinstance(parts, str):
        return parts.strip()
    return ""


def _linearize_messages(conversation: dict) -> list[Message]:
    """Flatten ChatGPT mapping tree into a time-ordered list of messages.

    We take all mapping nodes that have a `message` and sort by message.create_time.
    This is not a perfect reconstruction of branching, but works well for typical exports.
    """

    mapping = conversation.get("mapping") or {}
    msgs: list[Message] = []

    for node in mapping.values():
        m = node.get("message")
        if not m:
            continue

        author = (m.get("author") or {}).get("role")
        if author not in ("user", "assistant"):
            continue

        text = _extract_message_text(m)
        if not text:
            continue

        msgs.append(
            Message(
                role=author,
                text=text,
                create_time=m.get("create_time"),
            )
        )

    msgs.sort(key=lambda x: x.create_time or 0)
    return msgs


def _turns(messages: list[Message]) -> list[list[Message]]:
    turns: list[list[Message]] = []
    cur: list[Message] = []
    for m in messages:
        cur.append(m)
        if m.role == "assistant":
            turns.append(cur)
            cur = []
    if cur:
        turns.append(cur)
    return turns


def _chunk_turns(turns: list[list[Message]], max_turns: int = 5, overlap: int = 1) -> list[list[Message]]:
    chunks: list[list[Message]] = []
    start = 0
    while start < len(turns):
        end = min(start + max_turns, len(turns))
        chunk_msgs = [m for t in turns[start:end] for m in t]
        if chunk_msgs:
            chunks.append(chunk_msgs)
        start = end - overlap if end < len(turns) else len(turns)
    return chunks


def parse_chatgpt(
    conversations_path: Path,
    output_dir: Path,
    persist_exclusions: bool = False,
    use_registry_exclusions: bool = True,
) -> dict:
    """Parse ChatGPT export with Constantine filtering.
    
    Args:
        conversations_path: Path to conversations.json
        output_dir: Output directory for evidence files
        persist_exclusions: If True, save new exclusions to registry
        use_registry_exclusions: If True, also exclude IDs from registry
    """
    conversations = json.loads(conversations_path.read_text(encoding="utf-8"))
    registry = get_registry()

    # Load existing exclusions from registry
    registry_exclusions = registry.get_excluded_ids() if use_registry_exclusions else set()

    # Determine cutoff from conversation data
    cutoff_title = "leadership: follower focus"
    cutoff_matches = [c for c in conversations if _norm_title(c.get("title", "")) == cutoff_title]
    
    # If cutoff conversation not found, check if we have registry exclusions to use
    cutoff_conv = None
    cutoff_time = 0.0
    
    if cutoff_matches:
        # If duplicates, use the most recent by create_time
        cutoff_conv = sorted(cutoff_matches, key=lambda c: c.get("create_time") or 0)[-1]
        cutoff_time = float(cutoff_conv.get("create_time") or 0)
    elif not registry_exclusions:
        raise RuntimeError(
            f"Could not find cutoff conversation with title: {cutoff_title!r}\n"
            "And no existing exclusions in registry. Run with original export first."
        )
    else:
        print(f"⚠️  Cutoff conversation not found, using {len(registry_exclusions)} exclusions from registry")

    allow_titles = {
        _norm_title("Enhancing Blockchain Capital Bio"),
        _norm_title("Plan for Tomorrow's Conversation"),
        _norm_title("Tax Status Change Confirmation"),
    }

    allow_ids: list[str] = []
    deny_ids: list[str] = []
    new_exclusions: list[dict] = []
    evidences: list[dict[str, Any]] = []

    for conv in conversations:
        title = conv.get("title") or ""
        ntitle = _norm_title(title)
        ctime = float(conv.get("create_time") or 0)
        conv_id = conv.get("conversation_id") or conv.get("id") or ""

        # Check if already excluded via registry
        if conv_id in registry_exclusions:
            deny_ids.append(conv_id)
            continue

        # Apply Constantine rule: exclude if before cutoff (unless allowlisted)
        is_old = cutoff_time > 0 and ctime <= cutoff_time
        is_exception = ntitle in allow_titles

        if is_old and not is_exception:
            deny_ids.append(conv_id)
            new_exclusions.append({
                "conversation_id": conv_id,
                "reason": "constantine_cutoff",
                "title": title,
                "create_time": ctime,
            })
            continue

        allow_ids.append(conv_id)

        msgs = _linearize_messages(conv)
        if not msgs:
            continue

        turns = _turns(msgs)
        chunks = _chunk_turns(turns, max_turns=5, overlap=1)

        for chunk_idx, chunk_msgs in enumerate(chunks):
            lines = [f"Title: {title}", f"Conversation ID: {conv_id}", ""]
            for m in chunk_msgs:
                who = "User" if m.role == "user" else "ChatGPT"
                lines.append(f"**{who}:** {m.text}")

            body = "\n\n".join(lines).strip()

            evidence = {
                "id": generate_evidence_id(str(conversations_path), body, chunk_idx),
                "source": {
                    "type": "chatgpt",
                    "path": str(conversations_path),
                    "conversationId": conv_id,
                    "title": title,
                },
                "timestamp": datetime.fromtimestamp(ctime, tz=timezone.utc).isoformat() if ctime else None,
                "content": body,
                "contentType": "conversation",
                "chunkIndex": chunk_idx,
                "chunkTotal": len(chunks),
            }
            evidence = {k: v for k, v in evidence.items() if v is not None}
            evidences.append(evidence)

    # Persist new exclusions to registry if requested
    if persist_exclusions and new_exclusions:
        registry.add_exclusions_batch(new_exclusions)
        print(f"📝 Persisted {len(new_exclusions)} new exclusions to registry")

    out_dir = output_dir / "chatgpt"
    write_evidence_batch(evidences, out_dir / "all_evidence.json")

    report = {
        "total_conversations": len(conversations),
        "cutoff": {
            "title": cutoff_conv.get("title") if cutoff_conv else "(from registry)",
            "conversationId": (cutoff_conv.get("conversation_id") or cutoff_conv.get("id")) if cutoff_conv else None,
            "create_time": cutoff_time if cutoff_time else None,
            "create_time_iso": datetime.fromtimestamp(cutoff_time, tz=timezone.utc).isoformat() if cutoff_time else None,
        },
        "allow_titles_exceptions": sorted(list(allow_titles)),
        "allowed_conversation_ids": allow_ids,
        "denied_conversation_ids": deny_ids,
        "allowed_count": len(allow_ids),
        "denied_count": len(deny_ids),
        "registry_exclusions_used": len(registry_exclusions),
        "new_exclusions_added": len(new_exclusions) if persist_exclusions else 0,
        "evidence_created": len(evidences),
    }

    (out_dir / "filter_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report


def main():
    ap = argparse.ArgumentParser(description="Parse ChatGPT export into Graphiti evidence")
    ap.add_argument(
        "--conversations",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "imports" / "chatgpt_export" / "conversations.json",
        help="Path to ChatGPT conversations.json",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent / "evidence",
        help="Output directory for evidence files",
    )
    ap.add_argument(
        "--persist-exclusions",
        action="store_true",
        help="Save Constantine exclusions to registry for future runs",
    )
    ap.add_argument(
        "--no-registry",
        action="store_true",
        help="Don't use existing exclusions from registry",
    )
    args = ap.parse_args()

    if not args.conversations.exists():
        raise SystemExit(f"ChatGPT conversations.json not found: {args.conversations}")

    report = parse_chatgpt(
        args.conversations,
        args.output,
        persist_exclusions=args.persist_exclusions,
        use_registry_exclusions=not args.no_registry,
    )
    summary_keys = ["total_conversations", "allowed_count", "denied_count", "evidence_created"]
    if report.get("registry_exclusions_used"):
        summary_keys.append("registry_exclusions_used")
    if report.get("new_exclusions_added"):
        summary_keys.append("new_exclusions_added")
    print(json.dumps({k: report[k] for k in summary_keys}, indent=2))
    print(f"\nOutput: {args.output / 'chatgpt'}")


if __name__ == "__main__":
    main()
