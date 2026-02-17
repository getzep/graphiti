"""
Common utilities for evidence ingestion.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def generate_evidence_id(source_path: str, content: str, chunk_index: int = 0) -> str:
    """Generate deterministic evidence ID from source and content."""
    data = f"{source_path}:{chunk_index}:{content[:500]}"
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def parse_date_from_filename(filename: str) -> datetime | None:
    """Extract date from filename like 2026-01-26.md or 2026-01-26-topic.md."""
    match = re.match(r"(\d{4}-\d{2}-\d{2})", filename)
    if match:
        try:
            return datetime.strptime(match.group(1), "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def extract_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """Extract YAML frontmatter from markdown content."""
    if not content.startswith("---"):
        return {}, content
    
    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}, content
    
    try:
        import yaml
        frontmatter = yaml.safe_load(parts[1])
        return frontmatter or {}, parts[2].strip()
    except Exception:
        return {}, content


def split_markdown_by_h2(content: str) -> list[tuple[str, str]]:
    """Split markdown by H2 headers. Returns list of (header, content) tuples."""
    sections = []
    current_header = ""
    current_content = []
    
    for line in content.split("\n"):
        if line.startswith("## "):
            # Save previous section
            if current_content:
                sections.append((current_header, "\n".join(current_content).strip()))
            current_header = line[3:].strip()
            current_content = []
        else:
            current_content.append(line)
    
    # Don't forget the last section
    if current_content:
        sections.append((current_header, "\n".join(current_content).strip()))
    
    return sections


def extract_h1_title(content: str) -> str:
    """Extract the H1 title from markdown content."""
    for line in content.split("\n"):
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def chunk_by_tokens(text: str, max_tokens: int = 1000, overlap: int = 100) -> list[str]:
    """
    Simple token-based chunking (approximates tokens as words * 1.3).
    Returns list of text chunks.
    """
    words = text.split()
    tokens_per_word = 1.3
    words_per_chunk = int(max_tokens / tokens_per_word)
    overlap_words = int(overlap / tokens_per_word)
    
    chunks = []
    start = 0
    
    while start < len(words):
        end = start + words_per_chunk
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap_words if end < len(words) else len(words)
    
    return chunks


def write_evidence(evidence: dict, output_path: Path) -> None:
    """Write evidence document to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(evidence, f, indent=2, default=str)


def write_evidence_batch(evidences: list[dict], output_path: Path) -> None:
    """Write multiple evidence documents to a single JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(evidences, f, indent=2, default=str)


def parse_telegram_context(text: str) -> dict[str, Any]:
    """Parse Telegram message context from format:
    [Telegram Yuan Han Li | Personal (@yuan_han_li) id:1439681712 +8h 2026-01-24 22:52 UTC]

    ⚠️  PII Notice: This function extracts personally-identifiable information
    (display name, username, user ID).  Callers MUST treat returned values as
    **sensitive PII** — do not log at INFO/DEBUG, do not include in public
    exports, and ensure storage complies with data-retention policies.
    """
    pattern = r"\[Telegram\s+([^|]+)\s*\|\s*([^(]+)\s*\((@\w+)\)\s*id:(\d+)\s+[^\]]+\]"
    match = re.search(pattern, text)
    
    if match:
        return {
            "channel": "telegram",
            "displayName": match.group(1).strip(),   # PII: real name
            "chatType": match.group(2).strip(),
            "username": match.group(3),               # PII: Telegram handle
            "userId": match.group(4),                 # PII: numeric Telegram ID
        }
    return {}


def extract_message_id(text: str) -> str | None:
    """Extract message_id from text like [message_id: 1540]."""
    match = re.search(r"\[message_id:\s*(\d+)\]", text)
    return match.group(1) if match else None


def clean_message_content(text: str) -> str:
    """Remove metadata markers from message content."""
    # Remove [Telegram ...] prefix
    text = re.sub(r"^\[Telegram[^\]]+\]\s*", "", text)
    # Remove [message_id: ...] suffix
    text = re.sub(r"\s*\[message_id:\s*\d+\]$", "", text)
    return text.strip()


_UUID_LINE_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def sanitize_for_graphiti(text: str) -> str:
    """Sanitize episode text before sending into Graphiti.

    Graphiti builds FalkorDB RediSearch full-text queries internally. Certain punctuation-heavy
    tokens (markdown tables with `|`, backticks, file paths, URLs) can trigger RediSearch
    syntax errors and cause retry loops.

    Policy: keep semantics, sacrifice exact punctuation.

    Transform (conservative but effective):
    - Drop bare UUID-only lines.
    - Remove markdown table pipe formatting and backticks.
    - Replace common punctuation that tends to become standalone tokens (|, /, \\, :, @, ~, *)
      with whitespace.
    - Replace any remaining non "word-ish" characters with whitespace.
    - Collapse excessive whitespace (preserve newlines).
    """

    if not text:
        return ""

    # Normalize some common Unicode punctuation.
    text = (
        text.replace("→", "->")
        .replace("•", "-")
        .replace("—", "-")
        .replace("–", "-")
    )

    # Remove backticks (often used in markdown/code, can create odd tokens).
    text = text.replace("`", "")

    # Remove/neutralize pipe-heavy markdown tables.
    lines: list[str] = []
    for line in text.splitlines():
        s = line.strip()

        # Drop UUID-only lines
        if _UUID_LINE_RE.match(s):
            continue

        # If it's a markdown table row, strip leading/trailing pipes and replace internal pipes.
        if s.startswith("|") and s.endswith("|") and "|" in s[1:-1]:
            s = s.strip("|").replace("|", " ")
            line = s
        else:
            line = line.replace("|", " ")

        lines.append(line)

    text = "\n".join(lines)

    # Replace punctuation that tends to become standalone query tokens.
    text = re.sub(r"[\\/:@~*]", " ", text)

    # Replace anything that's not alnum/underscore/space/newline/basic punctuation with space.
    text = re.sub(r"[^0-9A-Za-z_\n\r\t \.,()\-]", " ", text)

    # Collapse repeated spaces/tabs but keep newlines.
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip() + "\n"
