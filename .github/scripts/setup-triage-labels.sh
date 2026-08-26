#!/usr/bin/env bash
# One-time setup for Graphiti issue and pull request triage labels.
# Run from the repository root:
#   bash .github/scripts/setup-triage-labels.sh
#
# Requires the GitHub CLI authenticated with permission to manage labels.
#
# Run this BEFORE merging the issue forms. GitHub drops labels referenced by an
# issue form that do not yet exist in the repository, and it does so without an
# error, so 02-feature.yml would otherwise create issues with no type label.

set -euo pipefail

REPO="${REPO:-getzep/graphiti}"

create_label() {
    local name="$1"
    local color="$2"
    local description="$3"

    gh label create "$name" \
        --repo "$REPO" \
        --color "$color" \
        --description "$description" \
        --force
}

echo "Creating triage labels for $REPO..."

# Type labels
create_label "feature" "a2eeef" "New functionality or an improvement to existing behavior"

# Area labels
create_label "area/core" "1d76db" "Graphiti core library"
create_label "area/mcp" "1d76db" "Graphiti MCP server"
create_label "area/server" "1d76db" "Graphiti REST server"
create_label "area/docs" "1d76db" "Documentation and examples"

# Intake and design process
create_label "intake/needs-info" "fbca04" "Reporter needs to provide information before triage can continue"
create_label "needs-rfc" "e4e669" "Large feature needs design discussion or approval"
create_label "rfc-approved" "0e8a16" "Maintainers approved the large feature design"
create_label "needs-tests" "e4e669" "Pull request lacks adequate test coverage"
create_label "needs-rework" "d876e3" "Contribution needs focused revisions before maintainer review"
create_label "security" "b60205" "Public report may describe a security vulnerability; route it privately"

# Maintainer priority labels
create_label "triage/high" "d73a4a" "High priority - needs maintainer attention"
create_label "triage/medium" "fbca04" "Medium priority - worth reviewing"
create_label "triage/low" "0e8a16" "Low priority - backlog"
create_label "triage/skip" "e4e669" "Skip - duplicate, stale, or misaligned"

# Existing triage signals
create_label "duplicate" "cfd3d7" "Duplicate of another issue or pull request"
create_label "recommend-close" "b60205" "Triage recommends closing"

# enhancement and slop-detected are intentionally left untouched for historical items.
# New triage uses feature and needs-rework instead.

echo "Done. All active triage labels were created or updated."
