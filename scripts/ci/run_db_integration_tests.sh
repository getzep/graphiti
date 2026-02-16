#!/usr/bin/env bash
set -euo pipefail

uv run pytest \
  tests/test_graphiti_mock.py \
  tests/test_node_int.py \
  tests/test_edge_int.py \
  tests/cross_encoder/test_bge_reranker_client_int.py \
  tests/driver/test_falkordb_driver.py \
  -m "not integration"
