from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from scripts import om_compressor


class _FakeCursor:
    def __init__(self, row: dict | None = None):
        self._row = row

    def single(self):
        return self._row

    def consume(self):
        return None


class _FakeTx:
    def __init__(self, *, existing_content: str | None):
        self.existing_content = existing_content
        self.queries: list[str] = []

    def run(self, query: str, params=None):  # noqa: ANN001
        self.queries.append(query)
        normalized = " ".join(query.split())
        if normalized.startswith("MATCH (n:OMNode {node_id:$node_id}) RETURN n.content AS content"):
            if self.existing_content is None:
                return _FakeCursor(None)
            return _FakeCursor({"content": self.existing_content})
        return _FakeCursor()


class OMCompressorEmbeddingContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self._orig_rewrite = os.environ.get('OM_REWRITE_EMBEDDINGS')
        self.message = om_compressor.MessageRow(
            message_id='m1',
            source_session_id='s1',
            content='hello',
            created_at='2026-02-25T00:00:00Z',
            content_embedding=[0.1, 0.2],
            om_extract_attempts=0,
        )
        self.cfg = om_compressor.ExtractorConfig(
            schema_version='1',
            prompt_template='pt',
            model_id='gpt-5.1-codex-mini',
            extractor_version='ev1',
        )

    def tearDown(self) -> None:
        if self._orig_rewrite is None:
            os.environ.pop('OM_REWRITE_EMBEDDINGS', None)
        else:
            os.environ['OM_REWRITE_EMBEDDINGS'] = self._orig_rewrite

    def _extracted(self, content: str) -> om_compressor.ExtractedChunk:
        return om_compressor.ExtractedChunk(
            nodes=[
                om_compressor.ExtractionNode(
                    node_id='node-1',
                    node_type='Friction',
                    semantic_domain='planning',
                    content=content,
                    urgency_score=3,
                    source_session_id='s1',
                    source_message_ids=['m1'],
                )
            ],
            edges=[],
        )

    def test_node_content_mismatch_raises_when_rewrite_disabled(self) -> None:
        tx = _FakeTx(existing_content='old content')
        with (
            patch.object(om_compressor, '_extract_items', return_value=self._extracted('new content')),
            patch.object(om_compressor, '_embedding_config', return_value=('embeddinggemma', 2)),
            patch.object(om_compressor, '_embed_text', return_value=[0.3, 0.4]),
        ):
            os.environ.pop('OM_REWRITE_EMBEDDINGS', None)
            with self.assertRaises(om_compressor.NodeContentMismatchError):
                om_compressor._process_chunk_tx(
                    tx,
                    messages=[self.message],
                    chunk_id='chunk-1',
                    cfg=self.cfg,
                    observed_node_ids=[],
                    group_id='s1_observational_memory',
                )

        rewrite_queries = [q for q in tx.queries if 'SET n.content = $content' in q]
        self.assertEqual(rewrite_queries, [])

    def test_node_content_rewrite_requires_flag(self) -> None:
        tx = _FakeTx(existing_content='old content')
        with (
            patch.object(om_compressor, '_extract_items', return_value=self._extracted('new content')),
            patch.object(om_compressor, '_embedding_config', return_value=('embeddinggemma', 2)),
            patch.object(om_compressor, '_embed_text', return_value=[0.3, 0.4]),
        ):
            os.environ['OM_REWRITE_EMBEDDINGS'] = '1'
            result = om_compressor._process_chunk_tx(
                tx,
                messages=[self.message],
                chunk_id='chunk-1',
                cfg=self.cfg,
                observed_node_ids=[],
                group_id='s1_observational_memory',
            )

        self.assertEqual(result['nodes'], 1)
        rewrite_queries = [q for q in tx.queries if 'SET n.content = $content' in q]
        self.assertEqual(len(rewrite_queries), 1)


if __name__ == '__main__':
    unittest.main()
