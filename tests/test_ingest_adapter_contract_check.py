"""Tests for INGEST_ADAPTER_CONTRACT_V1 and checker script (FR-6)."""

from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ingest.contracts import (
    CHUNK_REQUIRED_FIELDS,
    INGEST_ADAPTER_CONTRACT_VERSION,
    RECORD_REQUIRED_FIELDS,
    RECORD_SOURCE_TYPES,
    validate_chunk,
    validate_determinism,
    validate_record,
)


def _valid_record() -> dict:
    return {
        'source_type': 'conversation',
        'source_key': 'session:main:abc123',
        'record_id': 'msg_001',
        'created_at': '2025-01-01T12:00:00Z',
        'text': 'Hello world',
        'provenance': {'source': 'session', 'agent_id': 'main'},
    }


def _valid_chunk() -> dict:
    return {
        'chunk_id': hashlib.sha256(b'test').hexdigest(),
        'chunk_key': 'session:main:abc123:c0',
        'source_key': 'session:main:abc123',
        'record_ids': ['msg_001', 'msg_002'],
        'content': 'Hello world\n\nGoodbye world',
        'content_hash': hashlib.sha256(b'Hello world\n\nGoodbye world').hexdigest(),
        'boundary_reason': 'semantic_drift',
        'token_count': 10,
    }


class TestContractVersion(unittest.TestCase):
    def test_version_is_v1(self):
        self.assertEqual(INGEST_ADAPTER_CONTRACT_VERSION, 'v1')


class TestValidateRecord(unittest.TestCase):
    def test_valid_record_no_errors(self):
        errors = validate_record(_valid_record())
        self.assertEqual(errors, [])

    def test_missing_field(self):
        for field in RECORD_REQUIRED_FIELDS:
            record = _valid_record()
            del record[field]
            errors = validate_record(record)
            self.assertTrue(
                any(f'missing required field: {field}' in e for e in errors),
                f'Expected error for missing {field}',
            )

    def test_wrong_type(self):
        record = _valid_record()
        record['source_type'] = 123  # should be str
        errors = validate_record(record)
        self.assertTrue(any('source_type' in e for e in errors))

    def test_invalid_source_type(self):
        record = _valid_record()
        record['source_type'] = 'invalid_type'
        errors = validate_record(record)
        self.assertTrue(any('source_type must be one of' in e for e in errors))

    def test_all_source_types_accepted(self):
        for st in RECORD_SOURCE_TYPES:
            record = _valid_record()
            record['source_type'] = st
            errors = validate_record(record)
            self.assertEqual(errors, [], f'source_type={st} should be valid')


class TestValidateChunk(unittest.TestCase):
    def test_valid_chunk_no_errors(self):
        errors = validate_chunk(_valid_chunk())
        self.assertEqual(errors, [])

    def test_missing_field(self):
        for field in CHUNK_REQUIRED_FIELDS:
            chunk = _valid_chunk()
            del chunk[field]
            errors = validate_chunk(chunk)
            self.assertTrue(
                any(f'missing required field: {field}' in e for e in errors),
                f'Expected error for missing {field}',
            )

    def test_empty_chunk_id(self):
        chunk = _valid_chunk()
        chunk['chunk_id'] = ''
        errors = validate_chunk(chunk)
        self.assertTrue(any('chunk_id must be non-empty' in e for e in errors))

    def test_empty_record_ids(self):
        chunk = _valid_chunk()
        chunk['record_ids'] = []
        errors = validate_chunk(chunk)
        self.assertTrue(any('record_ids must be non-empty' in e for e in errors))

    def test_non_string_record_ids(self):
        chunk = _valid_chunk()
        chunk['record_ids'] = ['valid', 123]
        errors = validate_chunk(chunk)
        self.assertTrue(any('record_ids[1] must be str' in e for e in errors))

    def test_wrong_type_token_count(self):
        chunk = _valid_chunk()
        chunk['token_count'] = '10'  # should be int
        errors = validate_chunk(chunk)
        self.assertTrue(any('token_count' in e for e in errors))


class TestValidateDeterminism(unittest.TestCase):
    def test_identical_chunks_pass(self):
        chunks = [_valid_chunk()]
        errors = validate_determinism(chunks, chunks)
        self.assertEqual(errors, [])

    def test_different_count_fails(self):
        chunks_a = [_valid_chunk()]
        chunks_b = [_valid_chunk(), _valid_chunk()]
        errors = validate_determinism(chunks_a, chunks_b)
        self.assertTrue(any('chunk count differs' in e for e in errors))

    def test_different_chunk_id_fails(self):
        c1 = _valid_chunk()
        c2 = _valid_chunk()
        c2['chunk_id'] = 'different_id'
        errors = validate_determinism([c1], [c2])
        self.assertTrue(any('chunk_id differs' in e for e in errors))

    def test_different_record_ids_fails(self):
        c1 = _valid_chunk()
        c2 = _valid_chunk()
        c2['record_ids'] = ['different']
        errors = validate_determinism([c1], [c2])
        self.assertTrue(any('record_ids differ' in e for e in errors))


class TestCheckerScript(unittest.TestCase):
    """Test that the checker script can be imported and run."""

    def test_checker_imports(self):
        import scripts.ingest_adapter_contract_check as checker

        self.assertTrue(hasattr(checker, 'main'))

    def test_checker_strict_mode_passes(self):
        """Strict mode should pass with valid fixtures."""
        import scripts.ingest_adapter_contract_check as checker

        # The checker creates its own fixtures internally
        exit_code = checker.main(['--strict'])
        self.assertEqual(exit_code, 0)


if __name__ == '__main__':
    unittest.main()
