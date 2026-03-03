#!/usr/bin/env python3
"""Backfill OM lane group_id tags on legacy OM graph objects.

Sets group_id on existing OM path nodes/edges that predate lane tagging:
- (:Message)
- (:Episode)
- (:OMNode)
- (:OMExtractionEvent)
- [:HAS_MESSAGE]
- [:EVIDENCE_FOR]
- any relationship between (:OMNode)-[]->(:OMNode)

Usage:
  python3 scripts/om_backfill_group_id.py --group-id s1_observational_memory --apply
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from neo4j import GraphDatabase

DEFAULT_OM_GROUP_ID = "s1_observational_memory"
NEO4J_ENV_FALLBACK_FILE = Path.home() / ".clawdbot" / "credentials" / "neo4j.env"


def _load_env_fallback() -> None:
    if not NEO4J_ENV_FALLBACK_FILE.exists():
        return
    for line in NEO4J_ENV_FALLBACK_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _driver() -> GraphDatabase.driver:
    _load_env_fallback()
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USERNAME") or os.environ.get("NEO4J_USER", "neo4j")
    pw = os.environ.get("NEO4J_PASSWORD")
    if not pw:
        raise RuntimeError("NEO4J_PASSWORD is required")
    return GraphDatabase.driver(uri, auth=(user, pw))


def _count(tx, query: str, params: dict[str, object]) -> int:
    row = tx.run(query, params).single()
    return int((row or {}).get("c") or 0)


def _set(tx, query: str, params: dict[str, object]) -> int:
    row = tx.run(query, params).single()
    return int((row or {}).get("updated") or 0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group-id", default=DEFAULT_OM_GROUP_ID)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    group_id = str(args.group_id).strip() or DEFAULT_OM_GROUP_ID
    if not all(ch.isalnum() or ch in {"_", "-", "."} for ch in group_id):
        raise RuntimeError(f"invalid --group-id: {group_id!r}")

    params = {"group_id": group_id}

    with _driver() as drv:
        with drv.session(database=os.environ.get("NEO4J_DATABASE", "neo4j")) as session:
            before = {
                "messages_null": _count(session, "MATCH (m:Message) WHERE m.group_id IS NULL RETURN count(m) AS c", params),
                "episodes_null": _count(session, "MATCH (e:Episode) WHERE e.group_id IS NULL RETURN count(e) AS c", params),
                "omnodes_null": _count(session, "MATCH (n:OMNode) WHERE n.group_id IS NULL RETURN count(n) AS c", params),
                "events_null": _count(session, "MATCH (e:OMExtractionEvent) WHERE e.group_id IS NULL RETURN count(e) AS c", params),
                "has_message_null": _count(
                    session,
                    "MATCH (:Episode)-[r:HAS_MESSAGE]->(:Message) WHERE r.group_id IS NULL RETURN count(r) AS c",
                    params,
                ),
                "evidence_for_null": _count(
                    session,
                    "MATCH (:Message)-[r:EVIDENCE_FOR]->(:OMNode) WHERE r.group_id IS NULL RETURN count(r) AS c",
                    params,
                ),
                "om_to_om_rels_null": _count(
                    session,
                    "MATCH (:OMNode)-[r]->(:OMNode) WHERE r.group_id IS NULL RETURN count(r) AS c",
                    params,
                ),
            }

            updated = {
                "messages": 0,
                "episodes": 0,
                "omnodes": 0,
                "events": 0,
                "has_message": 0,
                "evidence_for": 0,
                "om_to_om_rels": 0,
            }

            if args.apply:
                updated["messages"] = _set(
                    session,
                    "MATCH (m:Message) WHERE m.group_id IS NULL SET m.group_id = $group_id RETURN count(m) AS updated",
                    params,
                )
                updated["episodes"] = _set(
                    session,
                    "MATCH (e:Episode) WHERE e.group_id IS NULL SET e.group_id = $group_id RETURN count(e) AS updated",
                    params,
                )
                updated["omnodes"] = _set(
                    session,
                    "MATCH (n:OMNode) WHERE n.group_id IS NULL SET n.group_id = $group_id RETURN count(n) AS updated",
                    params,
                )
                updated["events"] = _set(
                    session,
                    "MATCH (e:OMExtractionEvent) WHERE e.group_id IS NULL SET e.group_id = $group_id RETURN count(e) AS updated",
                    params,
                )
                updated["has_message"] = _set(
                    session,
                    "MATCH (:Episode)-[r:HAS_MESSAGE]->(:Message) WHERE r.group_id IS NULL SET r.group_id = $group_id RETURN count(r) AS updated",
                    params,
                )
                updated["evidence_for"] = _set(
                    session,
                    "MATCH (:Message)-[r:EVIDENCE_FOR]->(:OMNode) WHERE r.group_id IS NULL SET r.group_id = $group_id RETURN count(r) AS updated",
                    params,
                )
                updated["om_to_om_rels"] = _set(
                    session,
                    "MATCH (:OMNode)-[r]->(:OMNode) WHERE r.group_id IS NULL SET r.group_id = $group_id RETURN count(r) AS updated",
                    params,
                )

            print(
                json.dumps(
                    {
                        "group_id": group_id,
                        "apply": bool(args.apply),
                        "before": before,
                        "updated": updated,
                    },
                    indent=2,
                )
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
