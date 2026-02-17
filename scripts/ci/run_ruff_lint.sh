#!/usr/bin/env bash
set -euo pipefail

collect_python_targets() {
  local base_ref=""

  # pull_request events expose the target branch as GITHUB_BASE_REF.
  if [[ -n "${GITHUB_BASE_REF:-}" ]]; then
    git fetch --no-tags --depth=1 origin "${GITHUB_BASE_REF}" >/dev/null 2>&1 || true
    base_ref="FETCH_HEAD"
  # push events expose the pre-push commit as GITHUB_EVENT_BEFORE.
  elif [[ -n "${GITHUB_EVENT_BEFORE:-}" && "${GITHUB_EVENT_BEFORE}" != "0000000000000000000000000000000000000000" ]]; then
    base_ref="${GITHUB_EVENT_BEFORE}"
  elif git rev-parse --verify HEAD~1 >/dev/null 2>&1; then
    base_ref="HEAD~1"
  fi

  if [[ -z "${base_ref}" ]]; then
    echo "No base ref detected; linting tracked Python files." >&2
    git ls-files '*.py'
    return
  fi

  diff_output=""
  if diff_output=$(git diff --name-only "${base_ref}...HEAD" -- '*.py' 2>/dev/null); then
    :
  else
    # Fallback for shallow or unrelated-history contexts.
    diff_output=$(git show --name-only --pretty='' HEAD -- '*.py' 2>/dev/null || true)
  fi

  printf '%s
' "${diff_output}" | while IFS= read -r file; do
    [[ -f "${file}" ]] && echo "${file}"
  done
}

py_targets=()
while IFS= read -r target; do
  [[ -n "${target}" ]] && py_targets+=("${target}")
done < <(collect_python_targets | sort -u)

if (( ${#py_targets[@]} > 0 )); then
  echo "Running Ruff on ${#py_targets[@]} changed Python files"
  uvx ruff check --output-format=github "${py_targets[@]}"
else
  echo "No changed Python files detected; skipping Ruff check."
fi

python3 scripts/delta_tool.py boundary-lint --   --manifest config/public_export_allowlist.yaml   --denylist config/public_export_denylist.yaml
python3 scripts/delta_tool.py contracts-check --   --policy config/migration_sync_policy.json   --state-manifest config/state_migration_manifest.json   --contract-policy config/delta_contract_policy.json   --extensions-dir extensions   --strict
