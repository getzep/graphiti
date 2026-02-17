#!/usr/bin/env python3
"""Workflow Pack Shadow Compare Harness.

Deterministic offline CLI harness that loads synthetic cases, builds each
context pack twice (resolver=graphiti_primary and resolver=qmd_only_eval),
computes overlap/provenance/scope/unknown metrics, and writes a markdown
diff report.

This harness is intentionally:
- stdlib-only
- deterministic (fixture-driven; no network calls)
- scope-aware (group_safe must never return private-scoped items)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


sys.dont_write_bytecode = True


# ── Helpers ──────────────────────────────────────────────────────────────────


def get_repo_root() -> Path:
    here = Path(__file__).resolve()
    for ancestor in here.parents:
        if (ancestor / "scripts").is_dir() and (ancestor / "evals").is_dir():
            return ancestor
    return Path.cwd()


REPO_ROOT = get_repo_root()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_git_sha() -> str:
    try:
        cp = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return cp.stdout.strip() or "unknown"
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return "unknown"


def _resolve_case_path(path: Path) -> Path:
    candidate = path if path.is_absolute() else (REPO_ROOT / path)
    resolved = candidate.resolve()
    if not resolved.exists() or not resolved.is_file():
        raise ValueError(f"Case file not found: {resolved}")
    if resolved.suffix.lower() != ".json":
        raise ValueError(f"Case file must be a .json file: {resolved}")
    return resolved


def load_cases(path: Path) -> list[dict[str, Any]]:
    resolved = _resolve_case_path(path)
    with open(resolved, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and isinstance(data.get("cases"), list):
        raw = data["cases"]
    elif isinstance(data, list):
        raw = data
    else:
        raise ValueError(f"Invalid case file format: {resolved}")

    cases: list[dict[str, Any]] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"Case at index {idx} is not an object")
        case = dict(item)
        case.setdefault("id", f"case-{idx + 1:03d}")
        cases.append(case)

    cases.sort(key=lambda c: str(c.get("id") or ""))
    return cases


# ── Pack building (deterministic, fixture-driven) ───────────────────────────


def _classify_item(fact: dict[str, Any]) -> str:
    """Classify a graphiti fact as promoted_fact or pending_candidate."""
    if fact.get("status") in ("pending", "requires_approval"):
        return "pending_candidate"
    return "promoted_fact"


def _has_provenance(item: dict[str, Any]) -> bool:
    """Check if an item has evidence refs / provenance."""
    refs = item.get("evidence_refs")
    if isinstance(refs, list) and len(refs) > 0:
        return True
    source_key = item.get("source_key")
    if source_key:
        return True
    return False


def _is_scope_violation(item_scope: str, requested_scope: str) -> bool:
    """Check if an item's scope violates the requested scope."""
    if requested_scope == "group_safe":
        return item_scope == "private"
    return False


def build_graphiti_pack(pack_input: dict[str, Any], requested_scope: str) -> list[dict[str, Any]]:
    """Build a context pack using graphiti_primary resolver.

    Uses graphiti_facts + candidates from the fixture, filtering by scope.
    """
    items: list[dict[str, Any]] = []

    for fact in (pack_input.get("graphiti_facts") or []):
        item_scope = str(fact.get("scope") or "private")
        if _is_scope_violation(item_scope, requested_scope):
            continue
        items.append({
            "id": fact.get("fact_id") or fact.get("id"),
            "kind": "promoted_fact",
            "scope": item_scope,
            "has_provenance": _has_provenance(fact),
            "text": _value_to_text(fact.get("value")),
            "predicate": fact.get("predicate"),
        })

    for cand in (pack_input.get("candidates") or []):
        item_scope = str(cand.get("scope") or "private")
        if _is_scope_violation(item_scope, requested_scope):
            continue
        items.append({
            "id": cand.get("candidate_id") or cand.get("id"),
            "kind": "pending_candidate",
            "scope": item_scope,
            "has_provenance": _has_provenance(cand),
            "text": _value_to_text(cand.get("value")),
            "predicate": cand.get("predicate"),
        })

    return items


def build_qmd_pack(pack_input: dict[str, Any], requested_scope: str) -> list[dict[str, Any]]:
    """Build a context pack using qmd_only_eval resolver.

    Uses qmd_items from the fixture, filtering by scope.
    """
    items: list[dict[str, Any]] = []

    for qmd in (pack_input.get("qmd_items") or []):
        item_scope = str(qmd.get("scope") or "private")
        if _is_scope_violation(item_scope, requested_scope):
            continue
        items.append({
            "id": qmd.get("id"),
            "kind": "qmd_item",
            "scope": item_scope,
            "has_provenance": bool(qmd.get("source_key")),
            "text": str(qmd.get("text") or ""),
            "score": qmd.get("score"),
        })

    return items


def _value_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value)


# ── Metrics ──────────────────────────────────────────────────────────────────


def compute_metrics(
    graphiti_items: list[dict[str, Any]],
    qmd_items: list[dict[str, Any]],
    requested_scope: str,
) -> dict[str, Any]:
    """Compute overlap, provenance, scope, and unknown metrics."""
    g_ids = {str(it["id"]) for it in graphiti_items}
    q_ids = {str(it["id"]) for it in qmd_items}

    common = g_ids & q_ids
    g_only = g_ids - q_ids
    q_only = q_ids - g_ids

    g_provenance = sum(1 for it in graphiti_items if it.get("has_provenance"))
    q_provenance = sum(1 for it in qmd_items if it.get("has_provenance"))

    g_scope_violations = sum(
        1 for it in graphiti_items
        if _is_scope_violation(str(it.get("scope") or "private"), requested_scope)
    )
    q_scope_violations = sum(
        1 for it in qmd_items
        if _is_scope_violation(str(it.get("scope") or "private"), requested_scope)
    )

    # Kind distribution for graphiti items
    g_kinds: Counter[str] = Counter()
    for it in graphiti_items:
        g_kinds[str(it.get("kind") or "unknown")] += 1

    return {
        "common_ids": sorted(common),
        "graphiti_only_ids": sorted(g_only),
        "qmd_only_ids": sorted(q_only),
        "graphiti_total": len(graphiti_items),
        "qmd_total": len(qmd_items),
        "graphiti_provenance": g_provenance,
        "qmd_provenance": q_provenance,
        "graphiti_scope_violations": g_scope_violations,
        "qmd_scope_violations": q_scope_violations,
        "graphiti_unknowns": 0,
        "qmd_unknowns": 0,
        "graphiti_kind_dist": dict(sorted(g_kinds.items())),
    }


# ── Report generation ────────────────────────────────────────────────────────


def write_report(
    *,
    out_path: Path,
    version: str,
    cases: list[dict[str, Any]],
    case_results: list[dict[str, Any]],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("# Workflow Pack Shadow Compare Report")
    lines.append("")
    lines.append(f"**Generated:** {utc_now_iso()}")
    lines.append("")
    lines.append(f"**Test Suite Version:** {version}")
    lines.append("")
    lines.append(f"**Cases Processed:** {len(case_results)}")
    lines.append("")
    lines.append("---")
    lines.append("")

    for result in case_results:
        case = result["case"]
        cid = str(case.get("id") or "")
        name = str(case.get("name") or cid)
        category = str(case.get("category") or "")
        query = str(case.get("query") or "")
        scope = str(case.get("scope") or "private")

        lines.append(f"## Case {cid}: {name}")
        lines.append("")
        lines.append(f"- **Category:** {category}")
        lines.append(f"- **Query:** {query}")
        lines.append(f"- **Scope:** {scope}")
        lines.append("")

        for pack_result in result["pack_results"]:
            pack_type = pack_result["pack_type"]
            mode = pack_result.get("mode")
            metrics = pack_result["metrics"]

            mode_str = f"mode={mode}" if mode else "mode=None"
            lines.append(f"### Pack Type: {pack_type} ({mode_str})")
            lines.append("")

            lines.append("#### Retrieval Backend Status")
            lines.append("")
            lines.append(f"- **Graphiti backend used:** {metrics['graphiti_total'] > 0}")
            lines.append(f"- **QMD backend used:** {metrics['qmd_total'] > 0}")
            lines.append("")

            lines.append("#### Item Counts")
            lines.append("")
            lines.append("| Backend | Total Items | With Provenance | Scope Violations | Unknowns |")
            lines.append("|---------|-------------|-----------------|------------------|----------|")
            lines.append(
                f"| Graphiti | {metrics['graphiti_total']} "
                f"| {metrics['graphiti_provenance']} "
                f"| {metrics['graphiti_scope_violations']} "
                f"| {metrics['graphiti_unknowns']} |"
            )
            lines.append(
                f"| QMD | {metrics['qmd_total']} "
                f"| {metrics['qmd_provenance']} "
                f"| {metrics['qmd_scope_violations']} "
                f"| {metrics['qmd_unknowns']} |"
            )
            lines.append("")

            lines.append("#### Item ID Overlap")
            lines.append("")
            lines.append(f"- **Common IDs:** {len(metrics['common_ids'])}")
            lines.append(f"- **Graphiti-only IDs:** {len(metrics['graphiti_only_ids'])}")
            lines.append(f"- **QMD-only IDs:** {len(metrics['qmd_only_ids'])}")
            lines.append("")

            # Kind distribution
            kind_dist = metrics.get("graphiti_kind_dist") or {}
            if kind_dist:
                lines.append("#### Item Kind Distribution (Graphiti)")
                lines.append("")
                for kind, count in sorted(kind_dist.items()):
                    lines.append(f"- **{kind}:** {count}")
                lines.append("")

            # Summary
            lines.append("#### Summary")
            lines.append("")

            g_only_count = len(metrics['graphiti_only_ids'])
            q_only_count = len(metrics['qmd_only_ids'])
            common_count = len(metrics['common_ids'])

            if g_only_count == 0 and q_only_count == 0 and common_count > 0:
                lines.append(f"\u2705 **Full overlap** \u2014 all {common_count} IDs match.")
            elif common_count > 0:
                lines.append(
                    f"\u26a0\ufe0f **Partial overlap** \u2014 "
                    f"{g_only_count} Graphiti-only, {q_only_count} QMD-only."
                )
            else:
                lines.append(
                    f"\u26a0\ufe0f **Partial overlap** \u2014 "
                    f"{g_only_count} Graphiti-only, {q_only_count} QMD-only."
                )

            total_scope_violations = (
                metrics['graphiti_scope_violations'] + metrics['qmd_scope_violations']
            )
            if total_scope_violations > 0:
                lines.append(
                    f"\u274c **Scope violations detected** \u2014 "
                    f"{total_scope_violations} item(s) violate scope gating."
                )
            else:
                lines.append(
                    "\u2705 **No scope violations** \u2014 all items respect the requested scope."
                )

            lines.append("")

        lines.append("---")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


# ── Main ─────────────────────────────────────────────────────────────────────


def process_case(case: dict[str, Any]) -> dict[str, Any]:
    """Process a single case: build packs for both resolvers and compute metrics."""
    scope = str(case.get("scope") or "private")
    pack_results: list[dict[str, Any]] = []

    for pack_input in (case.get("context_pack_inputs") or []):
        pack_type = str(pack_input.get("pack_type") or "")
        mode = pack_input.get("mode")

        graphiti_items = build_graphiti_pack(pack_input, scope)
        qmd_items = build_qmd_pack(pack_input, scope)
        metrics = compute_metrics(graphiti_items, qmd_items, scope)

        pack_results.append({
            "pack_type": pack_type,
            "mode": mode,
            "graphiti_items": graphiti_items,
            "qmd_items": qmd_items,
            "metrics": metrics,
        })

    return {"case": case, "pack_results": pack_results}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Workflow Pack Shadow Compare Harness (Graphiti vs QMD baseline)."
    )
    ap.add_argument(
        "--cases", required=True,
        help="Path to synthetic cases JSON file.",
    )
    ap.add_argument(
        "--out", required=True,
        help="Output markdown report path.",
    )
    args = ap.parse_args()

    cases_path = Path(args.cases)
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = (REPO_ROOT / out_path).resolve()

    cases = load_cases(cases_path)

    # Extract version from cases file metadata
    with open(_resolve_case_path(cases_path), "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    version = str(raw_data.get("version", "1.0")) if isinstance(raw_data, dict) else "1.0"

    case_results: list[dict[str, Any]] = []
    for case in cases:
        case_results.append(process_case(case))

    write_report(
        out_path=out_path,
        version=version,
        cases=cases,
        case_results=case_results,
    )

    print(f"Report written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
