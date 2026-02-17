#!/usr/bin/env python3
"""
Parse session transcripts (*.jsonl) into normalized evidence documents.

Usage:
    python parse_sessions.py [--output DIR] [--sessions-dir DIR] [--agent AGENT]
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from common import (
    generate_evidence_id,
    parse_telegram_context,
    extract_message_id,
    clean_message_content,
    write_evidence_batch,
)


def read_jsonl(file_path: Path) -> Iterator[dict]:
    """Read JSONL file, yielding parsed objects."""
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def extract_text_content(content: list[dict] | str) -> str:
    """Extract text from message content (handles various formats)."""
    if isinstance(content, str):
        return content
    
    texts = []
    for item in content:
        if isinstance(item, dict):
            if item.get("type") == "text":
                texts.append(item.get("text", ""))
            elif item.get("type") == "thinking":
                # Include thinking for context but mark it
                thinking = item.get("thinking", "")
                if thinking:
                    texts.append(f"[thinking: {thinking[:200]}...]" if len(thinking) > 200 else f"[thinking: {thinking}]")
        elif isinstance(item, str):
            texts.append(item)
    
    return "\n".join(texts)


def parse_session_file(file_path: Path, agent_id: str) -> list[dict]:
    """Parse a single session file into evidence documents."""
    session_id = file_path.stem
    messages = []
    session_meta = {}
    
    # First pass: collect messages and metadata
    for entry in read_jsonl(file_path):
        entry_type = entry.get("type")
        
        if entry_type == "session":
            session_meta = {
                "sessionId": entry.get("id"),
                "timestamp": entry.get("timestamp"),
                "cwd": entry.get("cwd"),
            }
        elif entry_type == "message":
            msg = entry.get("message", {})
            if msg.get("role") in ("user", "assistant"):
                messages.append({
                    "id": entry.get("id"),
                    "timestamp": entry.get("timestamp"),
                    "role": msg.get("role"),
                    "content": extract_text_content(msg.get("content", [])),
                    "raw_content": msg.get("content"),
                })
    
    if not messages:
        return []
    
    # Group into conversation turns (user + assistant pairs)
    turns = []
    current_turn = []
    
    for msg in messages:
        current_turn.append(msg)
        if msg["role"] == "assistant":
            turns.append(current_turn)
            current_turn = []
    
    # Don't forget trailing messages without assistant response
    if current_turn:
        turns.append(current_turn)
    
    # Chunk turns (max 5 per chunk, 1 overlap)
    MAX_TURNS_PER_CHUNK = 5
    OVERLAP = 1
    
    evidences = []
    total_chunks = max(1, (len(turns) - 1) // (MAX_TURNS_PER_CHUNK - OVERLAP) + 1)
    
    chunk_idx = 0
    start = 0
    
    while start < len(turns):
        end = min(start + MAX_TURNS_PER_CHUNK, len(turns))
        chunk_turns = turns[start:end]
        
        # Flatten messages in this chunk
        chunk_messages = [msg for turn in chunk_turns for msg in turn]
        
        if not chunk_messages:
            start = end - OVERLAP if end < len(turns) else len(turns)
            continue
        
        # Extract metadata from first user message
        first_user = next((m for m in chunk_messages if m["role"] == "user"), None)
        context = {}
        participants = set()
        
        if first_user:
            context = parse_telegram_context(first_user["content"])
            if context.get("username"):
                participants.add(context["username"])
            if context.get("displayName"):
                participants.add(context["displayName"])
        
        # Always add "Archibald" as assistant participant
        participants.add("Archibald")
        
        # Build conversation content
        content_lines = []
        message_ids = []
        
        for msg in chunk_messages:
            role_label = "User" if msg["role"] == "user" else "Archibald"
            clean_content = clean_message_content(msg["content"]) if msg["role"] == "user" else msg["content"]
            content_lines.append(f"**{role_label}:** {clean_content}")
            
            # Extract message_id if present
            if msg["role"] == "user":
                msg_id = extract_message_id(msg["content"])
                if msg_id:
                    message_ids.append(msg_id)
            if msg.get("id"):
                message_ids.append(msg["id"])
        
        # Get timestamp range
        timestamps = [m["timestamp"] for m in chunk_messages if m.get("timestamp")]
        
        evidence = {
            "id": generate_evidence_id(str(file_path), "\n".join(content_lines), chunk_idx),
            "source": {
                "type": "session",
                "path": str(file_path),
                "agent": agent_id,
                "sessionId": session_id,
            },
            "timestamp": timestamps[0] if timestamps else session_meta.get("timestamp"),
            "timestampRange": {
                "start": timestamps[0] if timestamps else None,
                "end": timestamps[-1] if timestamps else None,
            } if len(timestamps) > 1 else None,
            "messageIds": message_ids if message_ids else None,
            "channel": context.get("channel"),
            "participants": list(participants) if participants else None,
            "content": "\n\n".join(content_lines),
            "contentType": "conversation",
            "chunkIndex": chunk_idx,
            "chunkTotal": total_chunks,
        }
        
        # Remove None values
        evidence = {k: v for k, v in evidence.items() if v is not None}
        
        evidences.append(evidence)
        
        chunk_idx += 1
        start = end - OVERLAP if end < len(turns) else len(turns)
    
    return evidences


def find_session_dirs(base_dir: Path) -> list[tuple[str, Path]]:
    """Find all agent session directories."""
    agents_dir = base_dir / "agents"
    if not agents_dir.exists():
        return []
    
    results = []
    for agent_dir in agents_dir.iterdir():
        if agent_dir.is_dir():
            sessions_dir = agent_dir / "sessions"
            if sessions_dir.exists():
                results.append((agent_dir.name, sessions_dir))
    
    return results


def parse_all_sessions(
    sessions_base: Path,
    output_dir: Path,
    agent_filter: str | None = None,
) -> dict:
    """Parse all session files and write evidence documents."""
    agent_dirs = find_session_dirs(sessions_base)
    
    if agent_filter:
        agent_dirs = [(a, p) for a, p in agent_dirs if a == agent_filter]
    
    stats = {
        "agents_processed": 0,
        "sessions_processed": 0,
        "evidence_created": 0,
        "errors": [],
    }
    
    for agent_id, sessions_dir in agent_dirs:
        print(f"\n📁 Agent: {agent_id}")
        session_files = sorted(sessions_dir.glob("*.jsonl"))
        
        agent_evidence = []
        
        for file_path in session_files:
            try:
                evidences = parse_session_file(file_path, agent_id)
                agent_evidence.extend(evidences)
                stats["sessions_processed"] += 1
                stats["evidence_created"] += len(evidences)
                if evidences:
                    print(f"  ✓ {file_path.name}: {len(evidences)} chunks")
            except Exception as e:
                stats["errors"].append({"file": str(file_path), "error": str(e)})
                print(f"  ✗ {file_path.name}: {e}", file=sys.stderr)
        
        # Write agent-level evidence file
        if agent_evidence:
            agent_output = output_dir / "sessions" / agent_id / "all_evidence.json"
            write_evidence_batch(agent_evidence, agent_output)
        
        stats["agents_processed"] += 1
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Parse session transcripts into evidence")
    parser.add_argument(
        "--sessions-dir",
        type=Path,
        default=Path.home() / ".clawdbot",
        help="Base path to .clawdbot directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent / "evidence",
        help="Output directory for evidence files",
    )
    parser.add_argument(
        "--agent",
        type=str,
        default=None,
        help="Filter to specific agent (e.g., 'main', 'group')",
    )
    args = parser.parse_args()
    
    if not args.sessions_dir.exists():
        print(f"Error: Sessions directory not found: {args.sessions_dir}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Parsing sessions from: {args.sessions_dir}")
    print(f"Output directory: {args.output}")
    if args.agent:
        print(f"Filtering to agent: {args.agent}")
    
    stats = parse_all_sessions(args.sessions_dir, args.output, args.agent)
    
    print()
    print("=" * 50)
    print(f"Agents processed: {stats['agents_processed']}")
    print(f"Sessions processed: {stats['sessions_processed']}")
    print(f"Evidence created: {stats['evidence_created']}")
    if stats["errors"]:
        print(f"Errors: {len(stats['errors'])}")
        for err in stats["errors"][:5]:
            print(f"  - {err['file']}: {err['error']}")
        if len(stats["errors"]) > 5:
            print(f"  ... and {len(stats['errors']) - 5} more")
    
    print(f"\nOutput: {args.output / 'sessions'}")


if __name__ == "__main__":
    main()
