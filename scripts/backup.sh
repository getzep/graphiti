#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/backup.sh precutover [common-args]
  scripts/backup.sh snapshot [common-args] [--] [snapshot-create-args]
  scripts/backup.sh restore-test [common-args] [--] [restore-test-args]

Common args:
  --repo <path>          Repo root/subdir (default: .)
  --snapshot-dir <path>  Snapshot directory (default: backup/snapshots)
  --manifest <path>      Snapshot manifest path (default: backup/snapshots/latest-manifest.json)
  --export-dir <path>    Migration export temp dir (default: /tmp/graphiti-state-export-clean)

Examples:
  scripts/backup.sh precutover
  scripts/backup.sh snapshot -- --include state --include exports --force
  scripts/backup.sh restore-test -- --keep-temp
EOF
}

parse_common_args() {
  REPO_PATH='.'
  SNAPSHOT_DIR='backup/snapshots'
  MANIFEST_PATH='backup/snapshots/latest-manifest.json'
  EXPORT_DIR='/tmp/graphiti-state-export-clean'

  POSITIONAL_ARGS=()

  while (($#)); do
    case "$1" in
      --repo)
        REPO_PATH="${2:-}"
        shift 2
        ;;
      --snapshot-dir)
        SNAPSHOT_DIR="${2:-}"
        shift 2
        ;;
      --manifest)
        MANIFEST_PATH="${2:-}"
        shift 2
        ;;
      --export-dir)
        EXPORT_DIR="${2:-}"
        shift 2
        ;;
      --)
        shift
        POSITIONAL_ARGS=("$@")
        return 0
        ;;
      *)
        POSITIONAL_ARGS+=("$1")
        shift
        ;;
    esac
  done

  return 0
}

repo_root_from_arg() {
  git -C "$1" rev-parse --show-toplevel
}

cmd_precutover() {
  if ! parse_common_args "$@"; then
    return 1
  fi

  if [[ ${#POSITIONAL_ARGS[@]} -ne 0 ]]; then
    echo "ERROR: unexpected args for precutover: ${POSITIONAL_ARGS[*]}" >&2
    usage >&2
    return 2
  fi

  local repo_root
  repo_root="$(repo_root_from_arg "$REPO_PATH")"

  rm -rf "$EXPORT_DIR"

  python3 "$repo_root/scripts/state_migration_export.py" --dry-run --out "$EXPORT_DIR"
  python3 "$repo_root/scripts/state_migration_check.py" --package "$EXPORT_DIR" --dry-run
  python3 "$repo_root/scripts/state_migration_import.py" --dry-run --allow-overwrite --in "$EXPORT_DIR"

  python3 "$repo_root/scripts/snapshot_create.py" \
    --repo "$repo_root" \
    --snapshot-dir "$SNAPSHOT_DIR" \
    --manifest "$MANIFEST_PATH" \
    --force

  python3 "$repo_root/scripts/snapshot_restore_test.py" \
    --repo "$repo_root" \
    --snapshot-dir "$SNAPSHOT_DIR" \
    --manifest "$MANIFEST_PATH"
}

cmd_snapshot() {
  if ! parse_common_args "$@"; then
    return 1
  fi

  local repo_root
  repo_root="$(repo_root_from_arg "$REPO_PATH")"

  python3 "$repo_root/scripts/snapshot_create.py" \
    --repo "$repo_root" \
    --snapshot-dir "$SNAPSHOT_DIR" \
    --manifest "$MANIFEST_PATH" \
    "${POSITIONAL_ARGS[@]}"
}

cmd_restore_test() {
  if ! parse_common_args "$@"; then
    return 1
  fi

  local repo_root
  repo_root="$(repo_root_from_arg "$REPO_PATH")"

  python3 "$repo_root/scripts/snapshot_restore_test.py" \
    --repo "$repo_root" \
    --snapshot-dir "$SNAPSHOT_DIR" \
    --manifest "$MANIFEST_PATH" \
    "${POSITIONAL_ARGS[@]}"
}

main() {
  local command="${1:-}"
  if [[ -z "$command" ]]; then
    usage >&2
    return 2
  fi
  shift || true

  case "$command" in
    precutover)
      cmd_precutover "$@"
      ;;
    snapshot)
      cmd_snapshot "$@"
      ;;
    restore-test)
      cmd_restore_test "$@"
      ;;
    -h|--help|help)
      usage
      ;;
    *)
      echo "ERROR: unknown command: $command" >&2
      usage >&2
      return 2
      ;;
  esac
}

main "$@"
