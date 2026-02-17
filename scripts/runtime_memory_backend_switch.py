#!/usr/bin/env python3
"""Operator switch layer for runtime memory backend profiles.

Deterministic flip/revert between canonical backend profiles
(qmd_primary ↔ graphiti_primary).

Modes:
  --dry-run:  Print exact patch payload + expected guardrails (no changes).
  --execute:  Apply the profile switch (writes state file).
  --revert:   Switch back to previous profile.

Stdlib only.  Deterministic JSON output.

Usage:
    # Dry-run: see what would change
    python3 scripts/runtime_memory_backend_switch.py --target graphiti_primary --dry-run

    # Execute the flip
    python3 scripts/runtime_memory_backend_switch.py --target graphiti_primary --execute

    # Revert to previous
    python3 scripts/runtime_memory_backend_switch.py --revert --dry-run
    python3 scripts/runtime_memory_backend_switch.py --revert --execute
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

sys.dont_write_bytecode = True


REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILES_PATH = REPO_ROOT / "config" / "runtime_memory_backend_profiles.json"
STATE_PATH = REPO_ROOT / "config" / ".runtime_memory_backend_state.json"


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_profiles() -> dict[str, Any]:
    """Load canonical backend profiles."""
    if not PROFILES_PATH.exists():
        raise FileNotFoundError(f"Profiles file not found: {PROFILES_PATH}")
    with open(PROFILES_PATH) as f:
        return json.load(f)


def load_state() -> dict[str, Any]:
    """Load current backend state. Returns empty dict if no state file."""
    if not STATE_PATH.exists():
        return {}
    try:
        with open(STATE_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(state: dict[str, Any]) -> None:
    """Atomically write state file."""
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(str(tmp), str(STATE_PATH))


def current_profile_name(profiles_data: dict[str, Any], state: dict[str, Any]) -> str:
    """Return active profile name."""
    if state.get("active_profile"):
        return str(state["active_profile"])
    return str(profiles_data.get("default_profile", "qmd_primary"))


def generate_switch_payload(
    *,
    profiles_data: dict[str, Any],
    current_state: dict[str, Any],
    target_profile: str,
) -> dict[str, Any]:
    """Generate deterministic switch payload (for dry-run or execute)."""
    profiles = profiles_data.get("profiles", {})
    guardrails = profiles_data.get("guardrails", {})

    current = current_profile_name(profiles_data, current_state)

    if target_profile not in profiles:
        return {
            "action": "switch",
            "status": "error",
            "error": f"Unknown target profile: {target_profile}",
            "available_profiles": sorted(profiles.keys()),
        }

    if target_profile == current:
        return {
            "action": "switch",
            "status": "noop",
            "reason": f"Already on profile: {target_profile}",
            "current_profile": current,
            "target_profile": target_profile,
        }

    from_profile = profiles.get(current, {})
    to_profile = profiles.get(target_profile, {})

    payload = {
        "action": "switch",
        "status": "ready",
        "timestamp": _utc_now_iso(),
        "from_profile": current,
        "to_profile": target_profile,
        "patch": {
            "active_profile": target_profile,
            "previous_profile": current,
            "state_file": str(STATE_PATH),
        },
        "diff": {
            "memory_search_backend": {
                "from": from_profile.get("memory_search_backend"),
                "to": to_profile.get("memory_search_backend"),
            },
            "memory_get_backend": {
                "from": from_profile.get("memory_get_backend"),
                "to": to_profile.get("memory_get_backend"),
            },
            "resolver_mode": {
                "from": from_profile.get("resolver_mode"),
                "to": to_profile.get("resolver_mode"),
            },
            "shadow_compare": {
                "from": from_profile.get("shadow_compare"),
                "to": to_profile.get("shadow_compare"),
            },
            "failover_backend": {
                "from": from_profile.get("failover_backend"),
                "to": to_profile.get("failover_backend"),
            },
        },
        "guardrails": guardrails,
        "guardrails_satisfied": {
            "group_safe_gating": to_profile.get("group_safe_gating", False),
            "shadow_compare_enabled": to_profile.get("shadow_compare", False),
            "instant_revert_available": True,
        },
    }

    return payload


def execute_switch(
    *,
    profiles_data: dict[str, Any],
    current_state: dict[str, Any],
    target_profile: str,
) -> dict[str, Any]:
    """Execute a backend switch and write state file."""
    payload = generate_switch_payload(
        profiles_data=profiles_data,
        current_state=current_state,
        target_profile=target_profile,
    )

    if payload.get("status") != "ready":
        return payload

    current = current_profile_name(profiles_data, current_state)

    new_state = {
        "active_profile": target_profile,
        "previous_profile": current,
        "switched_at": _utc_now_iso(),
        "switch_history": current_state.get("switch_history", []) + [
            {
                "from": current,
                "to": target_profile,
                "at": _utc_now_iso(),
            }
        ],
    }

    save_state(new_state)
    payload["status"] = "applied"
    payload["state_written"] = str(STATE_PATH)
    return payload


def main() -> int:
    p = argparse.ArgumentParser(
        description="Runtime memory backend switch (flip/revert between profiles)."
    )
    p.add_argument("--target", help="Target profile name (e.g. graphiti_primary, qmd_primary)")
    p.add_argument("--revert", action="store_true", help="Revert to previous profile")
    p.add_argument("--dry-run", action="store_true", help="Print patch payload without applying")
    p.add_argument("--execute", action="store_true", help="Apply the switch")
    args = p.parse_args()

    if not args.dry_run and not args.execute:
        p.error("Must specify --dry-run or --execute")

    if args.revert and args.target:
        p.error("Cannot specify both --target and --revert")

    if not args.revert and not args.target:
        p.error("Must specify --target or --revert")

    try:
        profiles_data = load_profiles()
    except FileNotFoundError as e:
        print(_canonical_json({"error": str(e)}), file=sys.stderr)
        return 1

    current_state = load_state()

    # Determine target
    if args.revert:
        previous = current_state.get("previous_profile")
        if not previous:
            # Default revert target when no state: revert to default
            default_profile = str(profiles_data.get("default_profile", "qmd_primary"))
            current = current_profile_name(profiles_data, current_state)
            if current == default_profile:
                # Already at default, nothing to revert to
                result = {
                    "action": "revert",
                    "status": "noop",
                    "reason": "No previous profile recorded and already at default",
                    "current_profile": current,
                }
                print(_canonical_json(result))
                return 0
            previous = default_profile
        target = previous
    else:
        target = args.target

    if args.dry_run:
        payload = generate_switch_payload(
            profiles_data=profiles_data,
            current_state=current_state,
            target_profile=target,
        )
        payload["dry_run"] = True
        print(_canonical_json(payload))
        return 0 if payload.get("status") in ("ready", "noop") else 1
    else:
        result = execute_switch(
            profiles_data=profiles_data,
            current_state=current_state,
            target_profile=target,
        )
        print(_canonical_json(result))
        return 0 if result.get("status") in ("applied", "noop") else 1


if __name__ == "__main__":
    raise SystemExit(main())
