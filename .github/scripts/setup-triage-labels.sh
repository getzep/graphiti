#!/usr/bin/env bash
# One-time setup script for PR triage labels.
# Run from repo root: bash .github/scripts/setup-triage-labels.sh
#
# Requires: gh CLI authenticated with repo access

set -euo pipefail

REPO="getzep/graphiti"

echo "Creating triage labels for $REPO..."

# Priority labels
gh label create "triage/high"     --repo "$REPO" --color "d73a4a" --description "High priority - needs maintainer attention" --force
gh label create "triage/medium"   --repo "$REPO" --color "fbca04" --description "Medium priority - worth reviewing" --force
gh label create "triage/low"      --repo "$REPO" --color "0e8a16" --description "Low priority - backlog" --force
gh label create "triage/skip"     --repo "$REPO" --color "e4e669" --description "Skip - duplicate, stale, or misaligned" --force

# Signal labels
gh label create "needs-tests"     --repo "$REPO" --color "e4e669" --description "PR lacks adequate test coverage" --force
gh label create "needs-rfc"       --repo "$REPO" --color "e4e669" --description "Large change needs design discussion" --force
gh label create "slop-detected"   --repo "$REPO" --color "b60205" --description "Likely AI-generated low-quality contribution" --force
gh label create "duplicate"       --repo "$REPO" --color "cfd3d7" --description "Duplicate of another open PR" --force
gh label create "recommend-close" --repo "$REPO" --color "b60205" --description "Triage recommends closing" --force

echo "Done. All triage labels created/updated."
