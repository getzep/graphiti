#!/usr/bin/env python3
"""Report current runtime memory backend mode + guardrail state.

Exits non-zero if config is contradictory/unsafe or if --expect check fails.

Stdlib only.  Deterministic JSON output.

Usage:
    # Report current status
    python3 scripts/runtime_memory_backend_status.py

    # Assert expected profile (exits non-zero if mismatch)
    python3 scripts/runtime_memory_backend_status.py --expect qmd_primary
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True


REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILES_PATH = REPO_ROOT / "config" / "runtime_memory_backend_profiles.json"
STATE_PATH = REPO_ROOT / "config" / ".runtime_memory_backend_state.json"


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def load_profiles() -> dict[str, Any]:
    if not PROFILES_PATH.exists():
        return {}
    with open(PROFILES_PATH) as f:
        return json.load(f)


def load_state() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return {}
    try:
        with open(STATE_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def get_status() -> dict[str, Any]:
    """Build status report."""
    profiles_data = load_profiles()
    state = load_state()
    issues: list[str] = []

    profiles = profiles_data.get("profiles", {})
    guardrails = profiles_data.get("guardrails", {})
    default_profile = str(profiles_data.get("default_profile", "qmd_primary"))

    # Determine active profile
    if state.get("active_profile"):
        active = str(state["active_profile"])
        source = "state_file"
    else:
        active = default_profile
        source = "default"

    # Validate active profile exists
    if active not in profiles:
        issues.append(f"active_profile '{active}' not found in profiles")

    profile = profiles.get(active, {})

    # Check guardrails
    group_safe_ok = profile.get("group_safe_gating", False)
    shadow_compare_ok = profile.get("shadow_compare", False)

    if guardrails.get("group_safe_gating_required") and not group_safe_ok:
        issues.append("group_safe_gating is required but not enabled in active profile")

    if guardrails.get("shadow_compare_recommended") and not shadow_compare_ok:
        issues.append("shadow_compare is recommended but not enabled in active profile")

    # Check for instant revert capability
    previous = state.get("previous_profile")
    revert_available = bool(previous and previous in profiles and previous != active)
    if guardrails.get("instant_revert_required") and not revert_available and source == "state_file":
        issues.append("instant_revert required but no valid previous_profile recorded")

    status: dict[str, Any] = {
        "active_profile": active,
        "profile_source": source,
        "default_profile": default_profile,
        "previous_profile": previous,
        "revert_available": revert_available,
        "profile_details": {
            "label": profile.get("label", ""),
            "memory_search_backend": profile.get("memory_search_backend", ""),
            "memory_get_backend": profile.get("memory_get_backend", ""),
            "resolver_mode": profile.get("resolver_mode", ""),
            "shadow_compare": shadow_compare_ok,
            "shadow_backend": profile.get("shadow_backend"),
            "failover_backend": profile.get("failover_backend"),
            "group_safe_gating": group_safe_ok,
        },
        "guardrails": guardrails,
        "guardrails_satisfied": {
            "group_safe_gating": group_safe_ok,
            "shadow_compare": shadow_compare_ok,
            "instant_revert": revert_available or source == "default",
        },
        "issues": issues,
        "safe": len(issues) == 0,
        "state_file_exists": STATE_PATH.exists(),
        "switched_at": state.get("switched_at"),
    }

    return status


def main() -> int:
    p = argparse.ArgumentParser(
        description="Report runtime memory backend status + guardrails."
    )
    p.add_argument("--expect", help="Assert expected active profile (exit non-zero if mismatch)")
    p.add_argument("--quiet", "-q", action="store_true", help="Suppress JSON output (exit code only)")
    args = p.parse_args()

    status = get_status()

    if args.expect:
        expected = str(args.expect).strip()
        actual = status["active_profile"]
        if actual != expected:
            if not args.quiet:
                result = {
                    "check": "expect",
                    "expected": expected,
                    "actual": actual,
                    "match": False,
                    "status": status,
                }
                print(_canonical_json(result))
            return 1
        else:
            if not args.quiet:
                result = {
                    "check": "expect",
                    "expected": expected,
                    "actual": actual,
                    "match": True,
                    "status": status,
                }
                print(_canonical_json(result))
            return 0

    if not args.quiet:
        print(_canonical_json(status))

    return 0 if status["safe"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
