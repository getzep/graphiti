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

# Type (GitHub already provides bug, documentation, and question)
create_label "feature" "a2eeef" "New functionality or an improvement to existing behavior"

# Scope — which package the change belongs to (required on new issues and PRs)
create_label "scope:core" "1d76db" "graphiti_core library, including drivers, LLM, search, and ingest"
create_label "scope:mcp" "1d76db" "Graphiti MCP server"
create_label "scope:service" "1d76db" "Graphiti REST server"
create_label "scope:docs" "1d76db" "Documentation and examples"
create_label "scope:ci" "1d76db" "GitHub Actions, Docker, CLA, Makefile, and release plumbing"

# Not ready for review — automation keys off these
create_label "needs-info" "fbca04" "Author owes information before review can continue"
create_label "needs-issue" "fbca04" "Pull request is missing a linked issue"
create_label "needs-rfc" "e4e669" "Large feature needs design discussion or approval"
create_label "rfc-approved" "0e8a16" "Maintainers approved the large feature design"
create_label "needs-tests" "e4e669" "Pull request lacks adequate test coverage"
create_label "needs-rework" "d876e3" "Contribution needs focused revisions before maintainer review"

# Flags (GitHub provides invalid by default, but create it so a deleted/renamed
# default never breaks the bot's writes)
create_label "security" "b60205" "Public report may describe a security vulnerability; route it privately"
create_label "duplicate" "cfd3d7" "Duplicate of another issue or pull request"
create_label "invalid" "e6e6e6" "Spam, hiring post, or an empty or off-topic report"
create_label "stale" "795548" "Inactive and missing required info; scheduled to auto-close"

# enhancement, slop-detected, area/*, intake/needs-info, and triage/* are left
# untouched for historical items. New work uses feature, scope:*, and needs-*.

echo "Done. All active triage labels were created or updated."
