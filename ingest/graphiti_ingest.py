#!/usr/bin/env python3
"""
Ingest evidence documents into Graphiti.

This is a template/example showing how to feed evidence into Graphiti.
Requires Graphiti and Neo4j to be set up separately.

Usage:
    python graphiti_ingest.py --evidence-dir ./evidence [--dry-run]
"""

import argparse
import asyncio
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from common import sanitize_for_graphiti
except ImportError:
    from ingest.common import sanitize_for_graphiti

# Graphiti import (will fail if not installed - that's OK for Phase 1)
try:
    from graphiti_core import Graphiti
    from graphiti_core.nodes import EpisodeType
    GRAPHITI_AVAILABLE = True
except ImportError:
    GRAPHITI_AVAILABLE = False
    print("Note: graphiti-core not installed. Running in dry-run mode.")


def _sanitize_metadata_str(value: Any, field_name: str, max_len: int = 256) -> str:
    """Validate and sanitize a metadata string field.

    Returns a safe string (stripped, truncated, control-chars removed).
    Falls back to 'unknown' for missing/empty values.
    """
    if value is None:
        return "unknown"
    s = str(value).strip()
    if not s:
        return "unknown"
    # Remove control characters (keep printable + basic whitespace)
    s = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", s)
    return s[:max_len]


async def ingest_evidence(
    graphiti: Any,
    evidence: dict,
    dry_run: bool = False,
) -> dict:
    """Ingest a single evidence document into Graphiti."""
    
    source = evidence.get("source", {})
    source_type = _sanitize_metadata_str(source.get("type"), "source.type")
    source_path = _sanitize_metadata_str(source.get("path"), "source.path")
    evidence_id = _sanitize_metadata_str(evidence.get("id"), "evidence.id")
    
    # Build episode name
    if source_type == "memory":
        episode_name = f"memory:{evidence_id[:8]}"
        source_desc = f"Daily memory note: {Path(source_path).name}"
    else:
        agent = _sanitize_metadata_str(source.get("agent"), "source.agent")
        session_id = _sanitize_metadata_str(source.get("sessionId"), "source.sessionId")[:8]
        episode_name = f"session:{agent}:{evidence_id[:8]}"
        source_desc = f"Session transcript: {agent}/{session_id}"
    
    # Parse timestamp
    timestamp = evidence.get("timestamp")
    if timestamp:
        try:
            ref_time = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except ValueError:
            ref_time = datetime.now()
    else:
        ref_time = datetime.now()
    
    # Prepare episode data
    episode_body = sanitize_for_graphiti(str(evidence.get("content", "") or ""))
    episode_data = {
        "name": episode_name,
        "episode_body": episode_body,
        "source_description": source_desc,
        "reference_time": ref_time,
        "source": EpisodeType.text if GRAPHITI_AVAILABLE else "text",
    }
    
    if dry_run:
        return {
            "status": "dry_run",
            "episode": episode_name,
            "content_preview": evidence.get("content", "")[:100] + "...",
        }
    
    if not GRAPHITI_AVAILABLE:
        return {"status": "skipped", "reason": "graphiti not installed"}
    
    try:
        await graphiti.add_episode(**episode_data)
        return {"status": "success", "episode": episode_name}
    except Exception as e:
        return {"status": "error", "episode": episode_name, "error": str(e)}


async def ingest_all(evidence_dir: Path, dry_run: bool = False) -> dict:
    """Ingest all evidence files from directory."""
    
    stats = {
        "total": 0,
        "success": 0,
        "errors": [],
        "dry_run": dry_run,
    }
    
    # Initialize Graphiti (if available and not dry run)
    graphiti = None
    if GRAPHITI_AVAILABLE and not dry_run:
        neo4j_password = os.environ.get("NEO4J_PASSWORD")
        if not neo4j_password:
            raise SystemExit("Missing NEO4J_PASSWORD env var. Set it before ingesting or use --dry-run.")
        try:
            # Local Neo4j connection (password must come from environment).
            graphiti = Graphiti(
                uri=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
                user=os.environ.get("NEO4J_USER", "neo4j"),
                password=neo4j_password,
            )
            await graphiti.build_indices()
        except Exception as e:
            print(f"Warning: Could not connect to Neo4j: {e}")
            print("Running in dry-run mode.")
            dry_run = True
    
    # Find all evidence JSON files
    evidence_files = list(evidence_dir.glob("**/*.json"))
    print(f"Found {len(evidence_files)} evidence files")
    
    for file_path in evidence_files:
        print(f"\nProcessing: {file_path.relative_to(evidence_dir)}")
        
        try:
            with open(file_path) as f:
                data = json.load(f)
            
            # Handle both single evidence and arrays
            evidences = data if isinstance(data, list) else [data]
            
            for evidence in evidences:
                stats["total"] += 1
                result = await ingest_evidence(graphiti, evidence, dry_run)
                
                if result.get("status") == "success":
                    stats["success"] += 1
                elif result.get("status") == "error":
                    stats["errors"].append(result)
                
                if stats["total"] % 50 == 0:
                    print(f"  Processed {stats['total']} evidence documents...")
                    
        except Exception as e:
            print(f"  Error processing {file_path}: {e}")
            stats["errors"].append({"file": str(file_path), "error": str(e)})
    
    # Close Graphiti connection
    if graphiti:
        await graphiti.close()
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Ingest evidence into Graphiti")
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=Path(__file__).parent.parent / "evidence",
        help="Directory containing evidence JSON files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't actually ingest, just preview",
    )
    args = parser.parse_args()
    
    if not args.evidence_dir.exists():
        print(f"Error: Evidence directory not found: {args.evidence_dir}")
        sys.exit(1)
    
    print(f"Evidence directory: {args.evidence_dir}")
    print(f"Dry run: {args.dry_run}")
    print()
    
    stats = asyncio.run(ingest_all(args.evidence_dir, args.dry_run))
    
    print()
    print("=" * 50)
    print(f"Total processed: {stats['total']}")
    print(f"Successful: {stats['success']}")
    print(f"Errors: {len(stats['errors'])}")
    
    if stats["errors"]:
        print("\nFirst 5 errors:")
        for err in stats["errors"][:5]:
            print(f"  - {err}")


if __name__ == "__main__":
    main()
