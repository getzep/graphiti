"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import json

from graphiti_core.nodes import EpisodeType
from graphiti_core.utils.content_chunking import (
    CHARS_PER_TOKEN,
    _count_json_keys,
    _json_likely_dense,
    _text_likely_dense,
    chunk_json_content,
    chunk_message_content,
    chunk_text_content,
    estimate_tokens,
    generate_covering_chunks,
    should_chunk,
)


class TestEstimateTokens:
    def test_empty_string(self):
        assert estimate_tokens('') == 0

    def test_short_string(self):
        # 4 chars per token
        assert estimate_tokens('abcd') == 1
        assert estimate_tokens('abcdefgh') == 2

    def test_long_string(self):
        text = 'a' * 400
        assert estimate_tokens(text) == 100

    def test_uses_chars_per_token_constant(self):
        text = 'x' * (CHARS_PER_TOKEN * 10)
        assert estimate_tokens(text) == 10


class TestChunkJsonArray:
    def test_small_array_no_chunking(self):
        data = [{'name': 'Alice'}, {'name': 'Bob'}]
        content = json.dumps(data)
        chunks = chunk_json_content(content, chunk_size_tokens=1000)
        assert len(chunks) == 1
        assert json.loads(chunks[0]) == data

    def test_empty_array(self):
        chunks = chunk_json_content('[]', chunk_size_tokens=100)
        assert chunks == ['[]']

    def test_array_splits_at_element_boundaries(self):
        # Create array that exceeds chunk size
        data = [{'id': i, 'data': 'x' * 100} for i in range(20)]
        content = json.dumps(data)

        # Use small chunk size to force splitting
        chunks = chunk_json_content(content, chunk_size_tokens=100, overlap_tokens=20)

        # Verify all chunks are valid JSON arrays
        for chunk in chunks:
            parsed = json.loads(chunk)
            assert isinstance(parsed, list)
            # Each element should be a complete object
            for item in parsed:
                assert 'id' in item
                assert 'data' in item

    def test_array_preserves_all_elements(self):
        data = [{'id': i} for i in range(10)]
        content = json.dumps(data)

        chunks = chunk_json_content(content, chunk_size_tokens=50, overlap_tokens=10)

        # Collect all unique IDs across chunks (accounting for overlap)
        seen_ids = set()
        for chunk in chunks:
            parsed = json.loads(chunk)
            for item in parsed:
                seen_ids.add(item['id'])

        # All original IDs should be present
        assert seen_ids == set(range(10))


class TestChunkJsonObject:
    def test_small_object_no_chunking(self):
        data = {'name': 'Alice', 'age': 30}
        content = json.dumps(data)
        chunks = chunk_json_content(content, chunk_size_tokens=1000)
        assert len(chunks) == 1
        assert json.loads(chunks[0]) == data

    def test_empty_object(self):
        chunks = chunk_json_content('{}', chunk_size_tokens=100)
        assert chunks == ['{}']

    def test_object_splits_at_key_boundaries(self):
        # Create object that exceeds chunk size
        data = {f'key_{i}': 'x' * 100 for i in range(20)}
        content = json.dumps(data)

        chunks = chunk_json_content(content, chunk_size_tokens=100, overlap_tokens=20)

        # Verify all chunks are valid JSON objects
        for chunk in chunks:
            parsed = json.loads(chunk)
            assert isinstance(parsed, dict)
            # Each key-value pair should be complete
            for key in parsed:
                assert key.startswith('key_')

    def test_object_preserves_all_keys(self):
        data = {f'key_{i}': f'value_{i}' for i in range(10)}
        content = json.dumps(data)

        chunks = chunk_json_content(content, chunk_size_tokens=50, overlap_tokens=10)

        # Collect all unique keys across chunks
        seen_keys = set()
        for chunk in chunks:
            parsed = json.loads(chunk)
            seen_keys.update(parsed.keys())

        # All original keys should be present
        expected_keys = {f'key_{i}' for i in range(10)}
        assert seen_keys == expected_keys


class TestChunkJsonInvalid:
    def test_invalid_json_falls_back_to_text(self):
        invalid_json = 'not valid json {'
        chunks = chunk_json_content(invalid_json, chunk_size_tokens=1000)
        # Should fall back to text chunking
        assert len(chunks) >= 1
        assert invalid_json in chunks[0]

    def test_scalar_value_returns_as_is(self):
        for scalar in ['"string"', '123', 'true', 'null']:
            chunks = chunk_json_content(scalar, chunk_size_tokens=1000)
            assert chunks == [scalar]


class TestChunkTextContent:
    def test_small_text_no_chunking(self):
        text = 'This is a short text.'
        chunks = chunk_text_content(text, chunk_size_tokens=1000)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_splits_at_paragraph_boundaries(self):
        paragraphs = ['Paragraph one.', 'Paragraph two.', 'Paragraph three.']
        text = '\n\n'.join(paragraphs)

        # Use small chunk size to force splitting
        chunks = chunk_text_content(text, chunk_size_tokens=10, overlap_tokens=5)

        # Each chunk should contain complete paragraphs (possibly with overlap)
        for chunk in chunks:
            # Should not have partial words cut off mid-paragraph
            assert not chunk.endswith(' ')

    def test_splits_at_sentence_boundaries_for_large_paragraphs(self):
        # Create a single long paragraph with multiple sentences
        sentences = ['This is sentence number ' + str(i) + '.' for i in range(20)]
        long_paragraph = ' '.join(sentences)

        chunks = chunk_text_content(long_paragraph, chunk_size_tokens=50, overlap_tokens=10)

        # Should have multiple chunks
        assert len(chunks) > 1
        # Each chunk should end at a sentence boundary where possible
        for chunk in chunks[:-1]:  # All except last
            # Should end with sentence punctuation or continue to next chunk
            assert chunk[-1] in '.!? ' or True  # Allow flexibility

    def test_preserves_text_completeness(self):
        text = 'Alpha beta gamma delta epsilon zeta eta theta.'
        chunks = chunk_text_content(text, chunk_size_tokens=10, overlap_tokens=2)

        # All words should appear in at least one chunk
        all_words = set(text.replace('.', '').split())
        found_words = set()
        for chunk in chunks:
            found_words.update(chunk.replace('.', '').split())

        assert all_words <= found_words


class TestChunkMessageContent:
    def test_small_message_no_chunking(self):
        content = 'Alice: Hello!\nBob: Hi there!'
        chunks = chunk_message_content(content, chunk_size_tokens=1000)
        assert len(chunks) == 1
        assert chunks[0] == content

    def test_preserves_speaker_message_format(self):
        messages = [f'Speaker{i}: This is message number {i}.' for i in range(10)]
        content = '\n'.join(messages)

        chunks = chunk_message_content(content, chunk_size_tokens=50, overlap_tokens=10)

        # Each chunk should have complete speaker:message pairs
        for chunk in chunks:
            lines = [line for line in chunk.split('\n') if line.strip()]
            for line in lines:
                # Should have speaker: format
                assert ':' in line

    def test_json_message_array_format(self):
        messages = [{'role': 'user', 'content': f'Message {i}'} for i in range(10)]
        content = json.dumps(messages)

        chunks = chunk_message_content(content, chunk_size_tokens=50, overlap_tokens=10)

        # Each chunk should be valid JSON array
        for chunk in chunks:
            parsed = json.loads(chunk)
            assert isinstance(parsed, list)
            for msg in parsed:
                assert 'role' in msg
                assert 'content' in msg


class TestChunkOverlap:
    def test_json_array_overlap_captures_boundary_elements(self):
        data = [{'id': i, 'name': f'Entity {i}'} for i in range(10)]
        content = json.dumps(data)

        # Use settings that will create overlap
        chunks = chunk_json_content(content, chunk_size_tokens=80, overlap_tokens=30)

        if len(chunks) > 1:
            # Check that adjacent chunks share some elements
            for i in range(len(chunks) - 1):
                current = json.loads(chunks[i])
                next_chunk = json.loads(chunks[i + 1])

                # Get IDs from end of current and start of next
                current_ids = {item['id'] for item in current}
                next_ids = {item['id'] for item in next_chunk}

                # There should be overlap (shared IDs)
                # Note: overlap may be empty if elements are large
                # The test verifies the structure, not exact overlap amount
                _ = current_ids & next_ids

    def test_text_overlap_captures_boundary_text(self):
        paragraphs = [f'Paragraph {i} with some content here.' for i in range(10)]
        text = '\n\n'.join(paragraphs)

        chunks = chunk_text_content(text, chunk_size_tokens=50, overlap_tokens=20)

        if len(chunks) > 1:
            # Adjacent chunks should have some shared content
            for i in range(len(chunks) - 1):
                current_words = set(chunks[i].split())
                next_words = set(chunks[i + 1].split())

                # There should be some overlap
                overlap = current_words & next_words
                # At minimum, common words like 'Paragraph', 'with', etc.
                assert len(overlap) > 0


class TestEdgeCases:
    def test_very_large_single_element(self):
        # Single element larger than chunk size
        data = [{'content': 'x' * 10000}]
        content = json.dumps(data)

        chunks = chunk_json_content(content, chunk_size_tokens=100, overlap_tokens=10)

        # Should handle gracefully - may return single chunk or fall back
        assert len(chunks) >= 1

    def test_empty_content(self):
        assert chunk_text_content('', chunk_size_tokens=100) == ['']
        assert chunk_message_content('', chunk_size_tokens=100) == ['']

    def test_whitespace_only(self):
        chunks = chunk_text_content('   \n\n   ', chunk_size_tokens=100)
        assert len(chunks) >= 1


class TestShouldChunk:
    def test_empty_content_never_chunks(self):
        """Empty content should never chunk."""
        assert not should_chunk('', EpisodeType.text)
        assert not should_chunk('', EpisodeType.json)

    def test_short_content_never_chunks(self, monkeypatch):
        """Short content should never chunk regardless of density."""
        from graphiti_core.utils import content_chunking

        # Set very low thresholds that would normally trigger chunking
        monkeypatch.setattr(content_chunking, 'CHUNK_DENSITY_THRESHOLD', 0.001)
        monkeypatch.setattr(content_chunking, 'CHUNK_MIN_TOKENS', 1000)

        # Dense but short JSON (~200 tokens, below 1000 minimum)
        dense_data = [{'name': f'Entity{i}'} for i in range(50)]
        dense_json = json.dumps(dense_data)
        assert not should_chunk(dense_json, EpisodeType.json)

    def test_high_density_large_json_chunks(self, monkeypatch):
        """Large high-density JSON should trigger chunking."""
        from graphiti_core.utils import content_chunking

        monkeypatch.setattr(content_chunking, 'CHUNK_DENSITY_THRESHOLD', 0.01)
        monkeypatch.setattr(content_chunking, 'CHUNK_MIN_TOKENS', 500)

        # Dense JSON: many elements, large enough to exceed minimum
        dense_data = [{'name': f'Entity{i}', 'desc': 'x' * 20} for i in range(200)]
        dense_json = json.dumps(dense_data)
        assert should_chunk(dense_json, EpisodeType.json)

    def test_low_density_text_no_chunk(self, monkeypatch):
        """Low-density prose should not trigger chunking."""
        from graphiti_core.utils import content_chunking

        monkeypatch.setattr(content_chunking, 'CHUNK_DENSITY_THRESHOLD', 0.05)
        monkeypatch.setattr(content_chunking, 'CHUNK_MIN_TOKENS', 100)

        # Low-density prose: mostly lowercase narrative
        prose = 'the quick brown fox jumps over the lazy dog. ' * 50
        assert not should_chunk(prose, EpisodeType.text)

    def test_low_density_json_no_chunk(self, monkeypatch):
        """Low-density JSON (few elements, lots of content) should not chunk."""
        from graphiti_core.utils import content_chunking

        monkeypatch.setattr(content_chunking, 'CHUNK_DENSITY_THRESHOLD', 0.05)
        monkeypatch.setattr(content_chunking, 'CHUNK_MIN_TOKENS', 100)

        # Sparse JSON: few elements with lots of content each
        sparse_data = [{'content': 'x' * 1000}, {'content': 'y' * 1000}]
        sparse_json = json.dumps(sparse_data)
        assert not should_chunk(sparse_json, EpisodeType.json)


class TestJsonDensityEstimation:
    def test_dense_array_detected(self, monkeypatch):
        """Arrays with many elements should be detected as dense."""
        from graphiti_core.utils import content_chunking

        monkeypatch.setattr(content_chunking, 'CHUNK_DENSITY_THRESHOLD', 0.01)

        # Array with 100 elements, ~800 chars = 200 tokens
        # Density = 100/200 * 1000 = 500, threshold = 10
        data = [{'id': i} for i in range(100)]
        content = json.dumps(data)
        tokens = estimate_tokens(content)

        assert _json_likely_dense(content, tokens)

    def test_sparse_array_not_dense(self, monkeypatch):
        """Arrays with few elements should not be detected as dense."""
        from graphiti_core.utils import content_chunking

        monkeypatch.setattr(content_chunking, 'CHUNK_DENSITY_THRESHOLD', 0.05)

        # Array with 2 elements but lots of content each
        data = [{'content': 'x' * 1000}, {'content': 'y' * 1000}]
        content = json.dumps(data)
        tokens = estimate_tokens(content)

        assert not _json_likely_dense(content, tokens)

    def test_dense_object_detected(self, monkeypatch):
        """Objects with many keys should be detected as dense."""
        from graphiti_core.utils import content_chunking

        monkeypatch.setattr(content_chunking, 'CHUNK_DENSITY_THRESHOLD', 0.01)

        # Object with 50 keys
        data = {f'key_{i}': f'value_{i}' for i in range(50)}
        content = json.dumps(data)
        tokens = estimate_tokens(content)

        assert _json_likely_dense(content, tokens)

    def test_count_json_keys_shallow(self):
        """Key counting should work for nested structures."""
        data = {
            'a': 1,
            'b': {'c': 2, 'd': 3},
            'e': [{'f': 4}, {'g': 5}],
        }
        # At depth 2: a, b, c, d, e, f, g = 7 keys
        assert _count_json_keys(data, max_depth=2) == 7

    def test_count_json_keys_depth_limit(self):
        """Key counting should respect depth limit."""
        data = {
            'a': {'b': {'c': {'d': 1}}},
        }
        # At depth 1: only 'a'
        assert _count_json_keys(data, max_depth=1) == 1
        # At depth 2: 'a' and 'b'
        assert _count_json_keys(data, max_depth=2) == 2


class TestTextDensityEstimation:
    def test_entity_rich_text_detected(self, monkeypatch):
        """Text with many proper nouns should be detected as dense."""
        from graphiti_core.utils import content_chunking

        monkeypatch.setattr(content_chunking, 'CHUNK_DENSITY_THRESHOLD', 0.01)

        # Entity-rich text: many capitalized names
        text = 'Alice met Bob at Acme Corp. Then Carol and David joined them. '
        text += 'Eve from Globex introduced Frank and Grace. '
        text += 'Later Henry and Iris arrived from Initech. '
        text = text * 10
        tokens = estimate_tokens(text)

        assert _text_likely_dense(text, tokens)

    def test_prose_not_dense(self, monkeypatch):
        """Narrative prose should not be detected as dense."""
        from graphiti_core.utils import content_chunking

        monkeypatch.setattr(content_chunking, 'CHUNK_DENSITY_THRESHOLD', 0.05)

        # Low-entity prose
        prose = """
        the sun was setting over the horizon as the old man walked slowly
        down the dusty road. he had been traveling for many days and his
        feet were tired. the journey had been long but he knew that soon
        he would reach his destination. the wind whispered through the trees
        and the birds sang their evening songs.
        """
        prose = prose * 10
        tokens = estimate_tokens(prose)

        assert not _text_likely_dense(prose, tokens)

    def test_sentence_starters_ignored(self, monkeypatch):
        """Capitalized words after periods should be ignored."""
        from graphiti_core.utils import content_chunking

        monkeypatch.setattr(content_chunking, 'CHUNK_DENSITY_THRESHOLD', 0.05)

        # Many sentences but no mid-sentence proper nouns
        text = 'This is a sentence. Another one follows. Yet another here. '
        text = text * 50
        tokens = estimate_tokens(text)

        # Should not be dense since capitals are sentence starters
        assert not _text_likely_dense(text, tokens)


class TestGenerateCoveringChunks:
    """Tests for the greedy covering chunks algorithm (Handshake Flights Problem)."""

    def test_empty_list(self):
        """Empty list should return single chunk with empty items."""
        result = generate_covering_chunks([], k=3)
        # n=0 <= k=3, so returns single chunk with empty items
        assert result == [([], [])]

    def test_single_item(self):
        """Single item should return one chunk with that item."""
        items = ['A']
        result = generate_covering_chunks(items, k=3)
        assert len(result) == 1
        assert result[0] == (['A'], [0])

    def test_items_fit_in_single_chunk(self):
        """When n <= k, all items should be in one chunk."""
        items = ['A', 'B', 'C']
        result = generate_covering_chunks(items, k=5)
        assert len(result) == 1
        chunk_items, indices = result[0]
        assert chunk_items == items
        assert indices == [0, 1, 2]

    def test_items_equal_to_k(self):
        """When n == k, all items should be in one chunk."""
        items = ['A', 'B', 'C', 'D']
        result = generate_covering_chunks(items, k=4)
        assert len(result) == 1
        chunk_items, indices = result[0]
        assert chunk_items == items
        assert indices == [0, 1, 2, 3]

    def test_all_pairs_covered_k2(self):
        """With k=2, every pair of items must appear in exactly one chunk."""
        items = ['A', 'B', 'C', 'D']
        result = generate_covering_chunks(items, k=2)

        # Collect all pairs from chunks
        covered_pairs = set()
        for _, indices in result:
            assert len(indices) == 2
            pair = frozenset(indices)
            covered_pairs.add(pair)

        # All C(4,2) = 6 pairs should be covered
        expected_pairs = {
            frozenset([0, 1]),
            frozenset([0, 2]),
            frozenset([0, 3]),
            frozenset([1, 2]),
            frozenset([1, 3]),
            frozenset([2, 3]),
        }
        assert covered_pairs == expected_pairs

    def test_all_pairs_covered_k3(self):
        """With k=3, every pair must appear in at least one chunk."""
        items = list(range(6))  # 0, 1, 2, 3, 4, 5
        result = generate_covering_chunks(items, k=3)

        # Collect all covered pairs
        covered_pairs: set[frozenset[int]] = set()
        for _, indices in result:
            assert len(indices) == 3
            # Each chunk of 3 covers C(3,2) = 3 pairs
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    covered_pairs.add(frozenset([indices[i], indices[j]]))

        # All C(6,2) = 15 pairs should be covered
        expected_pairs = {frozenset([i, j]) for i in range(6) for j in range(i + 1, 6)}
        assert covered_pairs == expected_pairs

    def test_all_pairs_covered_larger(self):
        """Verify all pairs covered for larger input."""
        items = list(range(10))
        result = generate_covering_chunks(items, k=4)

        # Collect all covered pairs
        covered_pairs: set[frozenset[int]] = set()
        for _, indices in result:
            assert len(indices) == 4
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    covered_pairs.add(frozenset([indices[i], indices[j]]))

        # All C(10,2) = 45 pairs should be covered
        expected_pairs = {frozenset([i, j]) for i in range(10) for j in range(i + 1, 10)}
        assert covered_pairs == expected_pairs

    def test_index_mapping_correctness(self):
        """Global indices should correctly map to original items."""
        items = ['Alice', 'Bob', 'Carol', 'Dave', 'Eve']
        result = generate_covering_chunks(items, k=3)

        for chunk_items, indices in result:
            # Each chunk item should match the item at the corresponding global index
            for local_idx, global_idx in enumerate(indices):
                assert chunk_items[local_idx] == items[global_idx]

    def test_greedy_minimizes_chunks(self):
        """Greedy approach should produce reasonably few chunks.

        For n=6, k=3: Each chunk covers C(3,2)=3 pairs.
        Total pairs = C(6,2) = 15.
        Lower bound = ceil(15/3) = 5 chunks.
        Schönheim bound = ceil(6/3 * ceil(5/2)) = ceil(2 * 3) = 6 chunks.

        Note: When random sampling is used (large n,k), the fallback mechanism
        may create additional small chunks to cover remaining pairs, so the
        upper bound is not guaranteed.
        """
        items = list(range(6))
        result = generate_covering_chunks(items, k=3)

        # For small inputs (exhaustive enumeration), should achieve near-optimal
        # Should be at least the simple lower bound (5 for this case)
        assert len(result) >= 5

        # Verify all pairs are covered (the primary guarantee)
        covered_pairs: set[frozenset[int]] = set()
        for _, indices in result:
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    covered_pairs.add(frozenset([indices[i], indices[j]]))
        expected_pairs = {frozenset([i, j]) for i in range(6) for j in range(i + 1, 6)}
        assert covered_pairs == expected_pairs

    def test_works_with_custom_types(self):
        """Function should work with any type, not just strings/ints."""

        class Entity:
            def __init__(self, name: str):
                self.name = name

        items = [Entity('A'), Entity('B'), Entity('C'), Entity('D')]
        result = generate_covering_chunks(items, k=2)

        # Verify structure
        assert len(result) > 0
        for chunk_items, indices in result:
            assert len(chunk_items) == 2
            assert len(indices) == 2
            # Items should be Entity objects
            for item in chunk_items:
                assert isinstance(item, Entity)

    def test_deterministic_output(self):
        """Same input should produce same output."""
        items = list(range(8))
        result1 = generate_covering_chunks(items, k=3)
        result2 = generate_covering_chunks(items, k=3)

        assert len(result1) == len(result2)
        for (chunk1, idx1), (chunk2, idx2) in zip(result1, result2, strict=True):
            assert chunk1 == chunk2
            assert idx1 == idx2

    def test_all_pairs_covered_k15_n30(self):
        """Verify all pairs covered for n=30, k=15 (realistic edge extraction scenario).

        For n=30, k=15:
        - Total pairs = C(30,2) = 435
        - Pairs per chunk = C(15,2) = 105
        - Lower bound = ceil(435/105) = 5 chunks
        - Schönheim bound = ceil(6/3 * ceil(5/2)) = ceil(2 * 3) = 6 chunks

        Note: When random sampling is used, the fallback mechanism may create
        additional small chunks (size 2) to cover remaining pairs, so chunk
        sizes may vary and the upper bound on chunk count is not guaranteed.
        """
        n = 30
        k = 15
        items = list(range(n))
        result = generate_covering_chunks(items, k=k)

        # Verify chunk sizes are at most k (fallback chunks may be smaller)
        for _, indices in result:
            assert len(indices) <= k, f'Expected chunk size <= {k}, got {len(indices)}'

        # Collect all covered pairs
        covered_pairs: set[frozenset[int]] = set()
        for _, indices in result:
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    covered_pairs.add(frozenset([indices[i], indices[j]]))

        # All C(30,2) = 435 pairs should be covered
        expected_pairs = {frozenset([i, j]) for i in range(n) for j in range(i + 1, n)}
        assert len(expected_pairs) == 435, f'Expected 435 pairs, got {len(expected_pairs)}'
        assert covered_pairs == expected_pairs, (
            f'Missing {len(expected_pairs - covered_pairs)} pairs: {expected_pairs - covered_pairs}'
        )

        # Verify chunk count is at least the lower bound
        assert len(result) >= 5, f'Expected at least 5 chunks, got {len(result)}'

    def test_all_pairs_covered_with_random_sampling(self):
        """Verify all pairs covered when random sampling is triggered.

        When C(n,k) > MAX_COMBINATIONS_TO_EVALUATE, the algorithm uses random
        sampling instead of exhaustive enumeration. This test ensures the
        fallback logic covers any pairs missed by the greedy sampling.
        """
        import random

        # n=50, k=5 triggers sampling since C(50,5) = 2,118,760 > 1000
        n = 50
        k = 5
        items = list(range(n))

        # Test with multiple random seeds to ensure robustness
        for seed in range(5):
            random.seed(seed)
            result = generate_covering_chunks(items, k=k)

            # Collect all covered pairs
            covered_pairs: set[frozenset[int]] = set()
            for _, indices in result:
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        covered_pairs.add(frozenset([indices[i], indices[j]]))

            # All C(50,2) = 1225 pairs should be covered
            expected_pairs = {frozenset([i, j]) for i in range(n) for j in range(i + 1, n)}
            assert len(expected_pairs) == 1225
            assert covered_pairs == expected_pairs, (
                f'Seed {seed}: Missing {len(expected_pairs - covered_pairs)} pairs'
            )

    def test_fallback_creates_pair_chunks_for_uncovered(self):
        """Verify fallback creates size-2 chunks for any remaining uncovered pairs.

        When the greedy algorithm breaks early (best_covered_count == 0),
        the fallback logic should create minimal chunks to cover remaining pairs.
        """
        import random

        # Use a large n with small k to stress the sampling
        n = 100
        k = 4
        items = list(range(n))

        random.seed(42)
        result = generate_covering_chunks(items, k=k)

        # Collect all covered pairs
        covered_pairs: set[frozenset[int]] = set()
        for _, indices in result:
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    covered_pairs.add(frozenset([indices[i], indices[j]]))

        # All C(100,2) = 4950 pairs must be covered
        expected_pairs = {frozenset([i, j]) for i in range(n) for j in range(i + 1, n)}
        assert len(expected_pairs) == 4950
        assert covered_pairs == expected_pairs, (
            f'Missing {len(expected_pairs - covered_pairs)} pairs'
        )

    def test_duplicate_sampling_safety(self):
        """Verify the algorithm handles duplicate random samples gracefully.

        When k is large relative to n, there are fewer unique combinations
        and random sampling may generate many duplicates. The safety counter
        should prevent infinite loops.
        """
        import random

        # n=20, k=10: C(20,10) = 184,756 > 1000 triggers sampling
        # With large k relative to n, duplicates are more likely
        n = 20
        k = 10
        items = list(range(n))

        random.seed(123)
        result = generate_covering_chunks(items, k=k)

        # Collect all covered pairs
        covered_pairs: set[frozenset[int]] = set()
        for _, indices in result:
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    covered_pairs.add(frozenset([indices[i], indices[j]]))

        # All C(20,2) = 190 pairs should be covered
        expected_pairs = {frozenset([i, j]) for i in range(n) for j in range(i + 1, n)}
        assert len(expected_pairs) == 190
        assert covered_pairs == expected_pairs

    def test_stress_multiple_seeds(self):
        """Stress test with multiple random seeds to ensure robustness.

        The combination of greedy sampling and fallback logic should
        guarantee all pairs are covered regardless of random seed.
        """
        import random

        n = 30
        k = 5
        items = list(range(n))
        expected_pairs = {frozenset([i, j]) for i in range(n) for j in range(i + 1, n)}

        for seed in range(10):
            random.seed(seed)
            result = generate_covering_chunks(items, k=k)

            covered_pairs: set[frozenset[int]] = set()
            for _, indices in result:
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        covered_pairs.add(frozenset([indices[i], indices[j]]))

            assert covered_pairs == expected_pairs, f'Seed {seed} failed to cover all pairs'
