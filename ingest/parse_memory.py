#!/usr/bin/env python3
"""
Parse memory/*.md files into normalized evidence documents.

Usage:
    python parse_memory.py [--output DIR] [--memory-dir DIR]
"""

import argparse
import json
import sys
from datetime import timezone
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from common import (
    generate_evidence_id,
    parse_date_from_filename,
    extract_frontmatter,
    split_markdown_by_h2,
    extract_h1_title,
    chunk_by_tokens,
    write_evidence_batch,
)


def parse_memory_file(file_path: Path) -> list[dict]:
    """Parse a single memory file into evidence documents."""
    content = file_path.read_text(encoding="utf-8")
    filename = file_path.name
    
    # Extract metadata
    file_date = parse_date_from_filename(filename)
    frontmatter, body = extract_frontmatter(content)
    h1_title = extract_h1_title(body)
    
    # Split by H2 sections
    sections = split_markdown_by_h2(body)
    
    # If no H2 sections, chunk the whole file
    if not sections or (len(sections) == 1 and not sections[0][0]):
        chunks = chunk_by_tokens(body, max_tokens=1000)
        sections = [(h1_title or filename, chunk) for chunk in chunks]
    
    evidences = []
    total_chunks = len(sections)
    
    for idx, (section_header, section_content) in enumerate(sections):
        if not section_content.strip():
            continue
        
        # Build content with context
        content_parts = []
        if h1_title and section_header != h1_title:
            content_parts.append(f"# {h1_title}")
        if section_header:
            content_parts.append(f"## {section_header}")
        content_parts.append(section_content)
        
        full_content = "\n\n".join(content_parts)
        
        evidence = {
            "id": generate_evidence_id(str(file_path), section_content, idx),
            "source": {
                "type": "memory",
                "path": str(file_path),
            },
            "timestamp": (file_date or frontmatter.get("date", "")).isoformat() if file_date else None,
            "section": section_header or None,
            "content": full_content,
            "contentType": "markdown",
            "chunkIndex": idx,
            "chunkTotal": total_chunks,
            "tags": frontmatter.get("tags", []),
        }
        
        # Add frontmatter metadata if present
        if frontmatter:
            evidence["frontmatter"] = frontmatter
        
        evidences.append(evidence)
    
    return evidences


def parse_all_memory(memory_dir: Path, output_dir: Path) -> dict:
    """Parse all memory files and write evidence documents."""
    memory_files = sorted(memory_dir.glob("*.md"))
    
    stats = {
        "files_processed": 0,
        "evidence_created": 0,
        "errors": [],
    }
    
    all_evidence = []
    
    for file_path in memory_files:
        try:
            evidences = parse_memory_file(file_path)
            all_evidence.extend(evidences)
            stats["files_processed"] += 1
            stats["evidence_created"] += len(evidences)
            print(f"✓ {file_path.name}: {len(evidences)} chunks")
        except Exception as e:
            stats["errors"].append({"file": str(file_path), "error": str(e)})
            print(f"✗ {file_path.name}: {e}", file=sys.stderr)
    
    # Write all evidence to single file (easier for batch ingestion)
    output_file = output_dir / "memory" / "all_memory_evidence.json"
    write_evidence_batch(all_evidence, output_file)
    
    # Also write per-date files for incremental updates
    by_date = {}
    for ev in all_evidence:
        date_key = ev.get("timestamp", "unknown")[:10] if ev.get("timestamp") else "unknown"
        by_date.setdefault(date_key, []).append(ev)
    
    for date_key, evidences in by_date.items():
        date_file = output_dir / "memory" / "by_date" / f"{date_key}.json"
        write_evidence_batch(evidences, date_file)
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Parse memory files into evidence")
    parser.add_argument(
        "--memory-dir",
        type=Path,
        default=Path(__file__).parent.parent.parent.parent / "memory",
        help="Path to memory directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent / "evidence",
        help="Output directory for evidence files",
    )
    args = parser.parse_args()
    
    if not args.memory_dir.exists():
        print(f"Error: Memory directory not found: {args.memory_dir}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Parsing memory files from: {args.memory_dir}")
    print(f"Output directory: {args.output}")
    print()
    
    stats = parse_all_memory(args.memory_dir, args.output)
    
    print()
    print("=" * 50)
    print(f"Files processed: {stats['files_processed']}")
    print(f"Evidence created: {stats['evidence_created']}")
    if stats["errors"]:
        print(f"Errors: {len(stats['errors'])}")
        for err in stats["errors"]:
            print(f"  - {err['file']}: {err['error']}")
    
    print(f"\nOutput: {args.output / 'memory'}")


if __name__ == "__main__":
    main()
