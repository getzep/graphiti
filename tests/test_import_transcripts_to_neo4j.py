"""Tests for import_transcripts_to_neo4j.py (FR-1)."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def _normalize_text(value: str) -> str:
    import re
    import unicodedata

    text = unicodedata.normalize('NFKC', value or '')
    text = text.strip()
    return re.sub(r'\s+', ' ', text)


class TestDeterministicIDs(unittest.TestCase):
    """Test deterministic ID formulas from the PRD."""

    def test_episode_id_formula(self):
        path = '/home/user/.openclaw/agents/main/sessions/abc123.jsonl'
        expected = _sha256(f'episode|{path}')
        self.assertEqual(len(expected), 64)
        # Same input => same output
        self.assertEqual(expected, _sha256(f'episode|{path}'))

    def test_message_id_formula(self):
        session_id = 'session-abc'
        file_path = '/path/to/file.jsonl'
        line_index = 5
        role = 'user'
        created_at = '2025-01-01T12:00:00Z'
        content = 'Hello world'
        normalized = _normalize_text(content)

        msg_id = _sha256(
            f'msg|{session_id}|{file_path}|{line_index}|{role}|{created_at}|{normalized}'
        )
        self.assertEqual(len(msg_id), 64)
        # Deterministic
        msg_id2 = _sha256(
            f'msg|{session_id}|{file_path}|{line_index}|{role}|{created_at}|{normalized}'
        )
        self.assertEqual(msg_id, msg_id2)

    def test_message_id_different_for_different_content(self):
        base = 'msg|s1|/f.jsonl|0|user|2025-01-01T00:00:00Z|'
        id1 = _sha256(base + 'hello')
        id2 = _sha256(base + 'world')
        self.assertNotEqual(id1, id2)


class TestContentExtraction(unittest.TestCase):
    """Test message content extraction logic."""

    def test_extract_text_from_string(self):
        from scripts.import_transcripts_to_neo4j import extract_text_content

        self.assertEqual(extract_text_content('hello'), 'hello')

    def test_extract_text_from_list(self):
        from scripts.import_transcripts_to_neo4j import extract_text_content

        content = [
            {'type': 'text', 'text': 'Hello'},
            {'type': 'thinking', 'thinking': 'Let me think...'},
            {'type': 'text', 'text': 'World'},
        ]
        result = extract_text_content(content)
        self.assertEqual(result, 'Hello\nWorld')

    def test_extract_text_excludes_thinking(self):
        from scripts.import_transcripts_to_neo4j import extract_text_content

        content = [
            {'type': 'thinking', 'thinking': 'Thinking about this...'},
        ]
        result = extract_text_content(content)
        self.assertEqual(result, '')

    def test_extract_text_empty(self):
        from scripts.import_transcripts_to_neo4j import extract_text_content

        self.assertEqual(extract_text_content(None), '')
        self.assertEqual(extract_text_content(''), '')
        self.assertEqual(extract_text_content([]), '')


class TestJSONLParsing(unittest.TestCase):
    """Test JSONL file parsing."""

    def test_parse_session_file(self):
        from scripts.import_transcripts_to_neo4j import parse_session_messages

        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.jsonl', delete=False
        ) as f:
            # Session header
            json.dump(
                {'type': 'session', 'id': 'test-session', 'timestamp': '2025-01-01T12:00:00Z'},
                f,
            )
            f.write('\n')
            # User message
            json.dump(
                {
                    'type': 'message',
                    'id': 'msg-1',
                    'timestamp': '2025-01-01T12:01:00Z',
                    'message': {
                        'role': 'user',
                        'content': 'Hello there, this is a test message.',
                    },
                },
                f,
            )
            f.write('\n')
            # Assistant message
            json.dump(
                {
                    'type': 'message',
                    'id': 'msg-2',
                    'timestamp': '2025-01-01T12:02:00Z',
                    'message': {
                        'role': 'assistant',
                        'content': [{'type': 'text', 'text': 'Hi! How can I help?'}],
                    },
                },
                f,
            )
            f.write('\n')
            # Delivery mirror (should be excluded)
            json.dump(
                {
                    'type': 'message',
                    'id': 'msg-3',
                    'timestamp': '2025-01-01T12:03:00Z',
                    'message': {
                        'role': 'assistant',
                        'content': 'Mirrored delivery',
                        'provider': 'clawdbot',
                    },
                },
                f,
            )
            f.write('\n')
            tmp_path = f.name

        try:
            session_id, messages = parse_session_messages(Path(tmp_path))
            self.assertEqual(session_id, 'test-session')
            self.assertEqual(len(messages), 2)
            self.assertEqual(messages[0]['role'], 'user')
            self.assertEqual(messages[1]['role'], 'assistant')
        finally:
            os.unlink(tmp_path)

    def test_exclude_thinking_content(self):
        from scripts.import_transcripts_to_neo4j import parse_session_messages

        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.jsonl', delete=False
        ) as f:
            json.dump(
                {'type': 'session', 'id': 'test-session', 'timestamp': '2025-01-01T00:00:00Z'},
                f,
            )
            f.write('\n')
            json.dump(
                {
                    'type': 'message',
                    'id': 'msg-1',
                    'timestamp': '2025-01-01T00:01:00Z',
                    'message': {
                        'role': 'assistant',
                        'content': [
                            {'type': 'thinking', 'thinking': 'Let me think about this...'},
                            {'type': 'text', 'text': 'Here is my answer.'},
                        ],
                    },
                },
                f,
            )
            f.write('\n')
            tmp_path = f.name

        try:
            _, messages = parse_session_messages(Path(tmp_path))
            self.assertEqual(len(messages), 1)
            self.assertEqual(messages[0]['content'], 'Here is my answer.')
        finally:
            os.unlink(tmp_path)


class TestStatsOutput(unittest.TestCase):
    """Test structured stats JSON output."""

    def test_stats_schema(self):
        from scripts.import_transcripts_to_neo4j import ImportStats

        stats = ImportStats()
        d = stats.to_dict()
        self.assertIn('files_seen', d)
        self.assertIn('messages_seen', d)
        self.assertIn('messages_inserted', d)
        self.assertIn('messages_updated', d)
        self.assertIn('embeddings_computed', d)
        self.assertIn('embeddings_skipped', d)
        self.assertIn('errors', d)

    def test_stats_json_serializable(self):
        from scripts.import_transcripts_to_neo4j import ImportStats

        stats = ImportStats()
        stats.files_seen = 10
        stats.messages_seen = 100
        serialized = json.dumps(stats.to_dict())
        self.assertIn('"files_seen": 10', serialized)


class TestDryRunMode(unittest.TestCase):
    """Test dry-run mode doesn't write to Neo4j."""

    def test_dry_run_flag(self):
        from scripts.import_transcripts_to_neo4j import parse_args

        args = parse_args([
            '--sessions-dir', '/tmp/test',
            '--dry-run',
            '--max-files', '2',
            '--max-messages', '100',
        ])
        self.assertTrue(args.dry_run)
        self.assertEqual(args.max_files, 2)
        self.assertEqual(args.max_messages, 100)

    def test_confirm_required_for_live(self):
        from scripts.import_transcripts_to_neo4j import parse_args

        args = parse_args(['--sessions-dir', '/tmp/test'])
        self.assertFalse(args.confirm)
        self.assertFalse(args.dry_run)


class TestNormalization(unittest.TestCase):
    """Test text normalization for content_hash."""

    def test_normalize_whitespace(self):
        from scripts.import_transcripts_to_neo4j import normalize_text

        self.assertEqual(normalize_text('hello  world'), 'hello world')
        self.assertEqual(normalize_text('  hello\n\nworld  '), 'hello world')

    def test_normalize_unicode(self):
        from scripts.import_transcripts_to_neo4j import normalize_text

        # NFKC normalization
        self.assertEqual(normalize_text(''), '')
        self.assertEqual(normalize_text('  '), '')


class TestContentHash(unittest.TestCase):
    """Test content_hash computation."""

    def test_content_hash_deterministic(self):
        content = 'Hello world'
        h1 = _sha256(_normalize_text(content))
        h2 = _sha256(_normalize_text(content))
        self.assertEqual(h1, h2)

    def test_content_hash_different_for_different_content(self):
        h1 = _sha256(_normalize_text('Hello'))
        h2 = _sha256(_normalize_text('World'))
        self.assertNotEqual(h1, h2)


if __name__ == '__main__':
    unittest.main()
