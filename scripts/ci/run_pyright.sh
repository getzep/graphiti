#!/usr/bin/env bash
set -euo pipefail

uv run pyright ./graphiti_core
(
  cd server
  uv sync --all-extras
  uv run pyright .
)
