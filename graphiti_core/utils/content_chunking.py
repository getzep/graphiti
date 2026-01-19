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
import logging
import random
import re
from itertools import combinations
from math import comb
from typing import TypeVar

from graphiti_core.helpers import (
    CHUNK_DENSITY_THRESHOLD,
    CHUNK_MIN_TOKENS,
    CHUNK_OVERLAP_TOKENS,
    CHUNK_TOKEN_SIZE,
)
from graphiti_core.nodes import EpisodeType

logger = logging.getLogger(__name__)

# Approximate characters per token (conservative estimate)
CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Estimate token count using character-based heuristic.

    Uses ~4 characters per token as a conservative estimate.
    This is faster than actual tokenization and works across all LLM providers.

    Args:
        text: The text to estimate tokens for

    Returns:
        Estimated token count
    """
    return len(text) // CHARS_PER_TOKEN


def _tokens_to_chars(tokens: int) -> int:
    """Convert token count to approximate character count."""
    return tokens * CHARS_PER_TOKEN


def should_chunk(content: str, episode_type: EpisodeType) -> bool:
    """Determine whether content should be chunked based on size and entity density.

    Only chunks content that is both:
    1. Large enough to potentially cause LLM issues (>= CHUNK_MIN_TOKENS)
    2. High entity density (many entities per token)

    Short content processes fine regardless of density. This targets the specific
    failure case of large entity-dense inputs while preserving context for
    prose/narrative content and avoiding unnecessary chunking of small inputs.

    Args:
        content: The content to evaluate
        episode_type: Type of episode (json, message, text)

    Returns:
        True if content is large and has high entity density
    """
    tokens = estimate_tokens(content)

    # Short content always processes fine - no need to chunk
    if tokens < CHUNK_MIN_TOKENS:
        return False

    return _estimate_high_density(content, episode_type, tokens)


def _estimate_high_density(content: str, episode_type: EpisodeType, tokens: int) -> bool:
    """Estimate whether content has high entity density.

    High-density content (many entities per token) benefits from chunking.
    Low-density content (prose, narratives) loses context when chunked.

    Args:
        content: The content to analyze
        episode_type: Type of episode
        tokens: Pre-computed token count

    Returns:
        True if content appears to have high entity density
    """
    if episode_type == EpisodeType.json:
        return _json_likely_dense(content, tokens)
    else:
        return _text_likely_dense(content, tokens)


def _json_likely_dense(content: str, tokens: int) -> bool:
    """Estimate entity density for JSON content.

    JSON is considered dense if it has many array elements or object keys,
    as each typically represents a distinct entity or data point.

    Heuristics:
    - Array: Count elements, estimate entities per 1000 tokens
    - Object: Count top-level keys

    Args:
        content: JSON string content
        tokens: Token count

    Returns:
        True if JSON appears to have high entity density
    """
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # Invalid JSON, fall back to text heuristics
        return _text_likely_dense(content, tokens)

    if isinstance(data, list):
        # For arrays, each element likely contains entities
        element_count = len(data)
        # Estimate density: elements per 1000 tokens
        density = (element_count / tokens) * 1000 if tokens > 0 else 0
        return density > CHUNK_DENSITY_THRESHOLD * 1000  # Scale threshold
    elif isinstance(data, dict):
        # For objects, count keys recursively (shallow)
        key_count = _count_json_keys(data, max_depth=2)
        density = (key_count / tokens) * 1000 if tokens > 0 else 0
        return density > CHUNK_DENSITY_THRESHOLD * 1000
    else:
        # Scalar value, no need to chunk
        return False


def _count_json_keys(data: dict, max_depth: int = 2, current_depth: int = 0) -> int:
    """Count keys in a JSON object up to a certain depth.

    Args:
        data: Dictionary to count keys in
        max_depth: Maximum depth to traverse
        current_depth: Current recursion depth

    Returns:
        Count of keys
    """
    if current_depth >= max_depth:
        return 0

    count = len(data)
    for value in data.values():
        if isinstance(value, dict):
            count += _count_json_keys(value, max_depth, current_depth + 1)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    count += _count_json_keys(item, max_depth, current_depth + 1)
    return count


def _text_likely_dense(content: str, tokens: int) -> bool:
    """Estimate entity density for text content.

    Uses capitalized words as a proxy for named entities (people, places,
    organizations, products). High ratio of capitalized words suggests
    high entity density.

    Args:
        content: Text content
        tokens: Token count

    Returns:
        True if text appears to have high entity density
    """
    if tokens == 0:
        return False

    # Split into words
    words = content.split()
    if not words:
        return False

    # Count capitalized words (excluding sentence starters)
    # A word is "capitalized" if it starts with uppercase and isn't all caps
    capitalized_count = 0
    for i, word in enumerate(words):
        # Skip if it's likely a sentence starter (after . ! ? or first word)
        if i == 0:
            continue
        if i > 0 and words[i - 1].rstrip()[-1:] in '.!?':
            continue

        # Check if capitalized (first char upper, not all caps)
        cleaned = word.strip('.,!?;:\'"()[]{}')
        if cleaned and cleaned[0].isupper() and not cleaned.isupper():
            capitalized_count += 1

    # Calculate density: capitalized words per 1000 tokens
    density = (capitalized_count / tokens) * 1000 if tokens > 0 else 0

    # Text density threshold is typically lower than JSON
    # A well-written article might have 5-10% named entities
    return density > CHUNK_DENSITY_THRESHOLD * 500  # Half the JSON threshold


def chunk_json_content(
    content: str,
    chunk_size_tokens: int | None = None,
    overlap_tokens: int | None = None,
) -> list[str]:
    """Split JSON content into chunks while preserving structure.

    For arrays: splits at element boundaries, keeping complete objects.
    For objects: splits at top-level key boundaries.

    Args:
        content: JSON string to chunk
        chunk_size_tokens: Target size per chunk in tokens (default from env)
        overlap_tokens: Overlap between chunks in tokens (default from env)

    Returns:
        List of JSON string chunks
    """
    chunk_size_tokens = chunk_size_tokens or CHUNK_TOKEN_SIZE
    overlap_tokens = overlap_tokens or CHUNK_OVERLAP_TOKENS

    chunk_size_chars = _tokens_to_chars(chunk_size_tokens)
    overlap_chars = _tokens_to_chars(overlap_tokens)

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        logger.warning('Failed to parse JSON, falling back to text chunking')
        return chunk_text_content(content, chunk_size_tokens, overlap_tokens)

    if isinstance(data, list):
        return _chunk_json_array(data, chunk_size_chars, overlap_chars)
    elif isinstance(data, dict):
        return _chunk_json_object(data, chunk_size_chars, overlap_chars)
    else:
        # Scalar value, return as-is
        return [content]


def _chunk_json_array(
    data: list,
    chunk_size_chars: int,
    overlap_chars: int,
) -> list[str]:
    """Chunk a JSON array by splitting at element boundaries."""
    if not data:
        return ['[]']

    chunks: list[str] = []
    current_elements: list = []
    current_size = 2  # Account for '[]'

    for element in data:
        element_json = json.dumps(element)
        element_size = len(element_json) + 2  # Account for comma and space

        # Check if adding this element would exceed chunk size
        if current_elements and current_size + element_size > chunk_size_chars:
            # Save current chunk
            chunks.append(json.dumps(current_elements))

            # Start new chunk with overlap (include last few elements)
            overlap_elements = _get_overlap_elements(current_elements, overlap_chars)
            current_elements = overlap_elements
            current_size = len(json.dumps(current_elements)) if current_elements else 2

        current_elements.append(element)
        current_size += element_size

    # Don't forget the last chunk
    if current_elements:
        chunks.append(json.dumps(current_elements))

    return chunks if chunks else ['[]']


def _get_overlap_elements(elements: list, overlap_chars: int) -> list:
    """Get elements from the end of a list that fit within overlap_chars."""
    if not elements:
        return []

    overlap_elements: list = []
    current_size = 2  # Account for '[]'

    for element in reversed(elements):
        element_json = json.dumps(element)
        element_size = len(element_json) + 2

        if current_size + element_size > overlap_chars:
            break

        overlap_elements.insert(0, element)
        current_size += element_size

    return overlap_elements


def _chunk_json_object(
    data: dict,
    chunk_size_chars: int,
    overlap_chars: int,
) -> list[str]:
    """Chunk a JSON object by splitting at top-level key boundaries."""
    if not data:
        return ['{}']

    chunks: list[str] = []
    current_keys: list[str] = []
    current_dict: dict = {}
    current_size = 2  # Account for '{}'

    for key, value in data.items():
        entry_json = json.dumps({key: value})
        entry_size = len(entry_json)

        # Check if adding this entry would exceed chunk size
        if current_dict and current_size + entry_size > chunk_size_chars:
            # Save current chunk
            chunks.append(json.dumps(current_dict))

            # Start new chunk with overlap (include last few keys)
            overlap_dict = _get_overlap_dict(current_dict, current_keys, overlap_chars)
            current_dict = overlap_dict
            current_keys = list(overlap_dict.keys())
            current_size = len(json.dumps(current_dict)) if current_dict else 2

        current_dict[key] = value
        current_keys.append(key)
        current_size += entry_size

    # Don't forget the last chunk
    if current_dict:
        chunks.append(json.dumps(current_dict))

    return chunks if chunks else ['{}']


def _get_overlap_dict(data: dict, keys: list[str], overlap_chars: int) -> dict:
    """Get key-value pairs from the end of a dict that fit within overlap_chars."""
    if not data or not keys:
        return {}

    overlap_dict: dict = {}
    current_size = 2  # Account for '{}'

    for key in reversed(keys):
        if key not in data:
            continue
        entry_json = json.dumps({key: data[key]})
        entry_size = len(entry_json)

        if current_size + entry_size > overlap_chars:
            break

        overlap_dict[key] = data[key]
        current_size += entry_size

    # Reverse to maintain original order
    return dict(reversed(list(overlap_dict.items())))


def chunk_text_content(
    content: str,
    chunk_size_tokens: int | None = None,
    overlap_tokens: int | None = None,
) -> list[str]:
    """Split text content at natural boundaries (paragraphs, sentences).

    Includes overlap to capture entities at chunk boundaries.

    Args:
        content: Text to chunk
        chunk_size_tokens: Target size per chunk in tokens (default from env)
        overlap_tokens: Overlap between chunks in tokens (default from env)

    Returns:
        List of text chunks
    """
    chunk_size_tokens = chunk_size_tokens or CHUNK_TOKEN_SIZE
    overlap_tokens = overlap_tokens or CHUNK_OVERLAP_TOKENS

    chunk_size_chars = _tokens_to_chars(chunk_size_tokens)
    overlap_chars = _tokens_to_chars(overlap_tokens)

    if len(content) <= chunk_size_chars:
        return [content]

    # Split into paragraphs first
    paragraphs = re.split(r'\n\s*\n', content)

    chunks: list[str] = []
    current_chunk: list[str] = []
    current_size = 0

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        para_size = len(paragraph)

        # If a single paragraph is too large, split it by sentences
        if para_size > chunk_size_chars:
            # First, save current chunk if any
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = []
                current_size = 0

            # Split large paragraph by sentences
            sentence_chunks = _chunk_by_sentences(paragraph, chunk_size_chars, overlap_chars)
            chunks.extend(sentence_chunks)
            continue

        # Check if adding this paragraph would exceed chunk size
        if current_chunk and current_size + para_size + 2 > chunk_size_chars:
            # Save current chunk
            chunks.append('\n\n'.join(current_chunk))

            # Start new chunk with overlap
            overlap_text = _get_overlap_text('\n\n'.join(current_chunk), overlap_chars)
            if overlap_text:
                current_chunk = [overlap_text]
                current_size = len(overlap_text)
            else:
                current_chunk = []
                current_size = 0

        current_chunk.append(paragraph)
        current_size += para_size + 2  # Account for '\n\n'

    # Don't forget the last chunk
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))

    return chunks if chunks else [content]


def _chunk_by_sentences(
    text: str,
    chunk_size_chars: int,
    overlap_chars: int,
) -> list[str]:
    """Split text by sentence boundaries."""
    # Split on sentence-ending punctuation followed by whitespace
    sentence_pattern = r'(?<=[.!?])\s+'
    sentences = re.split(sentence_pattern, text)

    chunks: list[str] = []
    current_chunk: list[str] = []
    current_size = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        sent_size = len(sentence)

        # If a single sentence is too large, split it by fixed size
        if sent_size > chunk_size_chars:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_size = 0

            # Split by fixed size as last resort
            fixed_chunks = _chunk_by_size(sentence, chunk_size_chars, overlap_chars)
            chunks.extend(fixed_chunks)
            continue

        # Check if adding this sentence would exceed chunk size
        if current_chunk and current_size + sent_size + 1 > chunk_size_chars:
            chunks.append(' '.join(current_chunk))

            # Start new chunk with overlap
            overlap_text = _get_overlap_text(' '.join(current_chunk), overlap_chars)
            if overlap_text:
                current_chunk = [overlap_text]
                current_size = len(overlap_text)
            else:
                current_chunk = []
                current_size = 0

        current_chunk.append(sentence)
        current_size += sent_size + 1

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


def _chunk_by_size(
    text: str,
    chunk_size_chars: int,
    overlap_chars: int,
) -> list[str]:
    """Split text by fixed character size (last resort)."""
    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size_chars, len(text))

        # Try to break at word boundary
        if end < len(text):
            space_idx = text.rfind(' ', start, end)
            if space_idx > start:
                end = space_idx

        chunks.append(text[start:end].strip())

        # Move start forward, ensuring progress even if overlap >= chunk_size
        # Always advance by at least (chunk_size - overlap) or 1 char minimum
        min_progress = max(1, chunk_size_chars - overlap_chars)
        start = max(start + min_progress, end - overlap_chars)

    return chunks


def _get_overlap_text(text: str, overlap_chars: int) -> str:
    """Get the last overlap_chars characters of text, breaking at word boundary."""
    if len(text) <= overlap_chars:
        return text

    overlap_start = len(text) - overlap_chars
    # Find the next word boundary after overlap_start
    space_idx = text.find(' ', overlap_start)
    if space_idx != -1:
        return text[space_idx + 1 :]
    return text[overlap_start:]


def chunk_message_content(
    content: str,
    chunk_size_tokens: int | None = None,
    overlap_tokens: int | None = None,
) -> list[str]:
    """Split conversation content preserving message boundaries.

    Never splits mid-message. Messages are identified by patterns like:
    - "Speaker: message"
    - JSON message arrays
    - Newline-separated messages

    Args:
        content: Conversation content to chunk
        chunk_size_tokens: Target size per chunk in tokens (default from env)
        overlap_tokens: Overlap between chunks in tokens (default from env)

    Returns:
        List of conversation chunks
    """
    chunk_size_tokens = chunk_size_tokens or CHUNK_TOKEN_SIZE
    overlap_tokens = overlap_tokens or CHUNK_OVERLAP_TOKENS

    chunk_size_chars = _tokens_to_chars(chunk_size_tokens)
    overlap_chars = _tokens_to_chars(overlap_tokens)

    if len(content) <= chunk_size_chars:
        return [content]

    # Try to detect message format
    # Check if it's JSON (array of message objects)
    try:
        data = json.loads(content)
        if isinstance(data, list):
            return _chunk_message_array(data, chunk_size_chars, overlap_chars)
    except json.JSONDecodeError:
        pass

    # Try speaker pattern (e.g., "Alice: Hello")
    speaker_pattern = r'^([A-Za-z_][A-Za-z0-9_\s]*):(.+?)(?=^[A-Za-z_][A-Za-z0-9_\s]*:|$)'
    if re.search(speaker_pattern, content, re.MULTILINE | re.DOTALL):
        return _chunk_speaker_messages(content, chunk_size_chars, overlap_chars)

    # Fallback to line-based chunking
    return _chunk_by_lines(content, chunk_size_chars, overlap_chars)


def _chunk_message_array(
    messages: list,
    chunk_size_chars: int,
    overlap_chars: int,
) -> list[str]:
    """Chunk a JSON array of message objects."""
    # Delegate to JSON array chunking
    chunks = _chunk_json_array(messages, chunk_size_chars, overlap_chars)
    return chunks


def _chunk_speaker_messages(
    content: str,
    chunk_size_chars: int,
    overlap_chars: int,
) -> list[str]:
    """Chunk messages in 'Speaker: message' format."""
    # Split on speaker patterns
    pattern = r'(?=^[A-Za-z_][A-Za-z0-9_\s]*:)'
    messages = re.split(pattern, content, flags=re.MULTILINE)
    messages = [m.strip() for m in messages if m.strip()]

    if not messages:
        return [content]

    chunks: list[str] = []
    current_messages: list[str] = []
    current_size = 0

    for message in messages:
        msg_size = len(message)

        # If a single message is too large, include it as its own chunk
        if msg_size > chunk_size_chars:
            if current_messages:
                chunks.append('\n'.join(current_messages))
                current_messages = []
                current_size = 0
            chunks.append(message)
            continue

        if current_messages and current_size + msg_size + 1 > chunk_size_chars:
            chunks.append('\n'.join(current_messages))

            # Get overlap (last message(s) that fit)
            overlap_messages = _get_overlap_messages(current_messages, overlap_chars)
            current_messages = overlap_messages
            current_size = sum(len(m) for m in current_messages) + len(current_messages) - 1

        current_messages.append(message)
        current_size += msg_size + 1

    if current_messages:
        chunks.append('\n'.join(current_messages))

    return chunks if chunks else [content]


def _get_overlap_messages(messages: list[str], overlap_chars: int) -> list[str]:
    """Get messages from the end that fit within overlap_chars."""
    if not messages:
        return []

    overlap: list[str] = []
    current_size = 0

    for msg in reversed(messages):
        msg_size = len(msg) + 1
        if current_size + msg_size > overlap_chars:
            break
        overlap.insert(0, msg)
        current_size += msg_size

    return overlap


def _chunk_by_lines(
    content: str,
    chunk_size_chars: int,
    overlap_chars: int,
) -> list[str]:
    """Chunk content by line boundaries."""
    lines = content.split('\n')

    chunks: list[str] = []
    current_lines: list[str] = []
    current_size = 0

    for line in lines:
        line_size = len(line) + 1

        if current_lines and current_size + line_size > chunk_size_chars:
            chunks.append('\n'.join(current_lines))

            # Get overlap lines
            overlap_text = '\n'.join(current_lines)
            overlap = _get_overlap_text(overlap_text, overlap_chars)
            if overlap:
                current_lines = overlap.split('\n')
                current_size = len(overlap)
            else:
                current_lines = []
                current_size = 0

        current_lines.append(line)
        current_size += line_size

    if current_lines:
        chunks.append('\n'.join(current_lines))

    return chunks if chunks else [content]


T = TypeVar('T')

MAX_COMBINATIONS_TO_EVALUATE = 1000


def _random_combination(n: int, k: int) -> tuple[int, ...]:
    """Generate a random combination of k items from range(n)."""
    return tuple(sorted(random.sample(range(n), k)))


def generate_covering_chunks(items: list[T], k: int) -> list[tuple[list[T], list[int]]]:
    """Generate chunks of items that cover all pairs using a greedy approach.

    Based on the Handshake Flights Problem / Covering Design problem.
    Each chunk of K items covers C(K,2) = K(K-1)/2 pairs. We greedily select
    chunks to maximize coverage of uncovered pairs, minimizing the total number
    of chunks needed to ensure every pair of items appears in at least one chunk.

    For large inputs where C(n,k) > MAX_COMBINATIONS_TO_EVALUATE, random sampling
    is used instead of exhaustive search to maintain performance.

    Lower bound (Schönheim): F >= ceil(N/K * ceil((N-1)/(K-1)))

    Args:
        items: List of items to partition into covering chunks
        k: Maximum number of items per chunk

    Returns:
        List of tuples (chunk_items, global_indices) where global_indices maps
        each position in chunk_items to its index in the original items list.
    """
    n = len(items)
    if n <= k:
        return [(items, list(range(n)))]

    # Track uncovered pairs using frozensets of indices
    uncovered_pairs: set[frozenset[int]] = {
        frozenset([i, j]) for i in range(n) for j in range(i + 1, n)
    }

    chunks: list[tuple[list[T], list[int]]] = []

    # Determine if we need to sample or can enumerate all combinations
    total_combinations = comb(n, k)
    use_sampling = total_combinations > MAX_COMBINATIONS_TO_EVALUATE

    while uncovered_pairs:
        # Greedy selection: find the chunk that covers the most uncovered pairs
        best_chunk_indices: tuple[int, ...] | None = None
        best_covered_count = 0

        if use_sampling:
            # Sample random combinations when there are too many to enumerate
            seen_combinations: set[tuple[int, ...]] = set()
            # Limit total attempts (including duplicates) to prevent infinite loops
            max_total_attempts = MAX_COMBINATIONS_TO_EVALUATE * 3
            total_attempts = 0
            samples_evaluated = 0
            while samples_evaluated < MAX_COMBINATIONS_TO_EVALUATE:
                total_attempts += 1
                if total_attempts > max_total_attempts:
                    # Too many total attempts, break to avoid infinite loop
                    break
                chunk_indices = _random_combination(n, k)
                if chunk_indices in seen_combinations:
                    continue
                seen_combinations.add(chunk_indices)
                samples_evaluated += 1

                # Count how many uncovered pairs this chunk covers
                covered_count = sum(
                    1
                    for i, idx_i in enumerate(chunk_indices)
                    for idx_j in chunk_indices[i + 1 :]
                    if frozenset([idx_i, idx_j]) in uncovered_pairs
                )

                if covered_count > best_covered_count:
                    best_covered_count = covered_count
                    best_chunk_indices = chunk_indices
        else:
            # Enumerate all combinations when feasible
            for chunk_indices in combinations(range(n), k):
                # Count how many uncovered pairs this chunk covers
                covered_count = sum(
                    1
                    for i, idx_i in enumerate(chunk_indices)
                    for idx_j in chunk_indices[i + 1 :]
                    if frozenset([idx_i, idx_j]) in uncovered_pairs
                )

                if covered_count > best_covered_count:
                    best_covered_count = covered_count
                    best_chunk_indices = chunk_indices

        if best_chunk_indices is None or best_covered_count == 0:
            # Greedy search couldn't find a chunk covering uncovered pairs.
            # This can happen with random sampling. Fall back to creating
            # small chunks that directly cover remaining pairs.
            break

        # Mark pairs in this chunk as covered
        for i, idx_i in enumerate(best_chunk_indices):
            for idx_j in best_chunk_indices[i + 1 :]:
                uncovered_pairs.discard(frozenset([idx_i, idx_j]))

        chunk_items = [items[idx] for idx in best_chunk_indices]
        chunks.append((chunk_items, list(best_chunk_indices)))

    # Handle any remaining uncovered pairs that the greedy algorithm missed.
    # This can happen when random sampling fails to find covering chunks.
    # Create minimal chunks (size 2) to guarantee all pairs are covered.
    for pair in uncovered_pairs:
        pair_indices = sorted(pair)
        chunk_items = [items[idx] for idx in pair_indices]
        chunks.append((chunk_items, pair_indices))

    return chunks
