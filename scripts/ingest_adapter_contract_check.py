#!/usr/bin/env python3
"""Contract checker for INGEST_ADAPTER_CONTRACT_V1.

Validates sample fixtures and outputs against the ingest adapter contract.
Runs built-in fixture tests by default; use --strict to exit non-zero on any
violation.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

# Ensure the repo root is importable so ``ingest.contracts`` resolves.
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ingest.contracts import (
    INGEST_ADAPTER_CONTRACT_VERSION,
    validate_chunk,
    validate_determinism,
    validate_record,
)


# ---------------------------------------------------------------------------
# Sample fixtures
# ---------------------------------------------------------------------------

def _make_valid_record() -> dict[str, Any]:
    return {
        'source_type': 'conversation',
        'source_key': 'session-2026-02-26',
        'record_id': 'rec-0001',
        'created_at': '2026-02-26T12:00:00Z',
        'text': 'Hello, world!',
        'provenance': {'origin': 'test-fixture'},
    }


def _make_valid_chunk(*, content: str = 'Hello, world!') -> dict[str, Any]:
    content_hash = hashlib.sha256(content.encode()).hexdigest()
    return {
        'chunk_id': f'chk-{content_hash[:12]}',
        'chunk_key': 'session-2026-02-26/chunk-0',
        'source_key': 'session-2026-02-26',
        'record_ids': ['rec-0001'],
        'content': content,
        'content_hash': content_hash,
        'boundary_reason': 'end-of-record',
        'token_count': 4,
    }


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def _run_tests() -> list[str]:
    """Run built-in fixture tests. Returns a list of failure descriptions."""
    failures: list[str] = []

    # -- Valid record should pass ------------------------------------------------
    valid_record = _make_valid_record()
    errs = validate_record(valid_record)
    if errs:
        failures.append(f'valid record unexpectedly failed: {errs}')

    # -- Valid chunk should pass -------------------------------------------------
    valid_chunk = _make_valid_chunk()
    errs = validate_chunk(valid_chunk)
    if errs:
        failures.append(f'valid chunk unexpectedly failed: {errs}')

    # -- Record missing required field -------------------------------------------
    bad_record = {k: v for k, v in valid_record.items() if k != 'text'}
    errs = validate_record(bad_record)
    if not errs:
        failures.append('record missing "text" should have produced errors')

    # -- Record with wrong type --------------------------------------------------
    bad_type_record = {**valid_record, 'text': 42}
    errs = validate_record(bad_type_record)
    if not errs:
        failures.append('record with int "text" should have produced errors')

    # -- Record with invalid source_type -----------------------------------------
    bad_source_record = {**valid_record, 'source_type': 'unknown'}
    errs = validate_record(bad_source_record)
    if not errs:
        failures.append('record with invalid source_type should have produced errors')

    # -- Chunk missing required field --------------------------------------------
    bad_chunk = {k: v for k, v in valid_chunk.items() if k != 'content'}
    errs = validate_chunk(bad_chunk)
    if not errs:
        failures.append('chunk missing "content" should have produced errors')

    # -- Chunk with empty chunk_id -----------------------------------------------
    bad_chunk_id = {**valid_chunk, 'chunk_id': ''}
    errs = validate_chunk(bad_chunk_id)
    if not errs:
        failures.append('chunk with empty chunk_id should have produced errors')

    # -- Chunk with empty record_ids ---------------------------------------------
    bad_rids = {**valid_chunk, 'record_ids': []}
    errs = validate_chunk(bad_rids)
    if not errs:
        failures.append('chunk with empty record_ids should have produced errors')

    # -- Chunk with non-string record_ids entry ----------------------------------
    bad_rid_type = {**valid_chunk, 'record_ids': [123]}
    errs = validate_chunk(bad_rid_type)
    if not errs:
        failures.append('chunk with non-string record_ids entry should have produced errors')

    # -- Determinism: identical chunks pass --------------------------------------
    chunk_a = _make_valid_chunk()
    chunk_b = _make_valid_chunk()
    errs = validate_determinism([chunk_a], [chunk_b])
    if errs:
        failures.append(f'identical chunks should be deterministic: {errs}')

    # -- Determinism: different chunk counts fail --------------------------------
    errs = validate_determinism([chunk_a], [chunk_a, chunk_b])
    if not errs:
        failures.append('differing chunk counts should produce errors')

    # -- Determinism: different chunk_ids fail -----------------------------------
    chunk_c = {**chunk_a, 'chunk_id': 'chk-different'}
    errs = validate_determinism([chunk_a], [chunk_c])
    if not errs:
        failures.append('differing chunk_ids should produce errors')

    return failures


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Validate fixtures against INGEST_ADAPTER_CONTRACT_V1.',
    )
    parser.add_argument(
        '--strict',
        action='store_true',
        help='Exit non-zero on any violation (default: print warnings, exit 0)',
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    print(f'Ingest adapter contract version: {INGEST_ADAPTER_CONTRACT_VERSION}')
    print()

    failures = _run_tests()

    if failures:
        print(f'{len(failures)} failure(s):', file=sys.stderr)
        for failure in failures:
            print(f'  - {failure}', file=sys.stderr)
        return 1 if args.strict else 0

    print('All fixture tests passed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
