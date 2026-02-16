#!/usr/bin/env bash
set -euo pipefail

uv run pytest tests/ -m "not integration" \
  --ignore=tests/test_graphiti_int.py \
  --ignore=tests/test_graphiti_mock.py \
  --ignore=tests/test_node_int.py \
  --ignore=tests/test_edge_int.py \
  --ignore=tests/test_entity_exclusion_int.py \
  --ignore=tests/driver/ \
  --ignore=tests/llm_client/test_anthropic_client_int.py \
  --ignore=tests/utils/maintenance/test_temporal_operations_int.py \
  --ignore=tests/cross_encoder/test_bge_reranker_client_int.py \
  --ignore=tests/evals/
