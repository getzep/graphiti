"""INGEST_ADAPTER_CONTRACT_V1 -- normative interface for all ingestion lanes."""

from __future__ import annotations

INGEST_ADAPTER_CONTRACT_VERSION = 'v1'

RECORD_REQUIRED_FIELDS = {
    'source_type': str,
    'source_key': str,
    'record_id': str,
    'created_at': str,
    'text': str,
    'provenance': dict,
}

RECORD_SOURCE_TYPES = {'conversation', 'document', 'structured_rows'}

CHUNK_REQUIRED_FIELDS = {
    'chunk_id': str,
    'chunk_key': str,
    'source_key': str,
    'record_ids': list,
    'content': str,
    'content_hash': str,
    'boundary_reason': str,
    'token_count': int,
}


def validate_record(record: dict) -> list[str]:
    """Validate a record against INGEST_ADAPTER_CONTRACT_V1. Returns list of errors."""
    errors = []
    for field, expected_type in RECORD_REQUIRED_FIELDS.items():
        if field not in record:
            errors.append(f'missing required field: {field}')
        elif not isinstance(record[field], expected_type):
            errors.append(
                f'{field} must be {expected_type.__name__}, '
                f'got {type(record[field]).__name__}'
            )
    if 'source_type' in record and record['source_type'] not in RECORD_SOURCE_TYPES:
        errors.append(
            f'source_type must be one of {RECORD_SOURCE_TYPES}, '
            f'got {record["source_type"]!r}'
        )
    return errors


def validate_chunk(chunk: dict) -> list[str]:
    """Validate a chunk against INGEST_ADAPTER_CONTRACT_V1. Returns list of errors."""
    errors = []
    for field, expected_type in CHUNK_REQUIRED_FIELDS.items():
        if field not in chunk:
            errors.append(f'missing required field: {field}')
        elif not isinstance(chunk[field], expected_type):
            errors.append(
                f'{field} must be {expected_type.__name__}, '
                f'got {type(chunk[field]).__name__}'
            )
    # Determinism: chunk_id must be non-empty
    if 'chunk_id' in chunk and not chunk['chunk_id']:
        errors.append('chunk_id must be non-empty')
    # record_ids must be non-empty list of strings
    if 'record_ids' in chunk and isinstance(chunk['record_ids'], list):
        if not chunk['record_ids']:
            errors.append('record_ids must be non-empty')
        for i, rid in enumerate(chunk['record_ids']):
            if not isinstance(rid, str):
                errors.append(f'record_ids[{i}] must be str')
    return errors


def validate_determinism(chunks_a: list[dict], chunks_b: list[dict]) -> list[str]:
    """Validate that two chunk sets from same input are deterministic."""
    errors = []
    if len(chunks_a) != len(chunks_b):
        errors.append(f'chunk count differs: {len(chunks_a)} vs {len(chunks_b)}')
        return errors
    for i, (a, b) in enumerate(zip(chunks_a, chunks_b)):
        if a.get('chunk_id') != b.get('chunk_id'):
            errors.append(
                f'chunk[{i}] chunk_id differs: {a.get("chunk_id")} vs {b.get("chunk_id")}'
            )
        if a.get('record_ids') != b.get('record_ids'):
            errors.append(f'chunk[{i}] record_ids differ')
    return errors
