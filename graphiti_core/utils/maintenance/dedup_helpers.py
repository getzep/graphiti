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

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import lru_cache
from hashlib import blake2b
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from graphiti_core.nodes import EntityNode

_NAME_ENTROPY_THRESHOLD = 1.5
_MIN_NAME_LENGTH = 6
_MIN_TOKEN_COUNT = 2
_FUZZY_JACCARD_THRESHOLD = 0.9
_MINHASH_PERMUTATIONS = 32
_MINHASH_BAND_SIZE = 4


def _normalize_string_exact(name: str) -> str:
    """Lowercase text and collapse whitespace so equal names map to the same key."""
    normalized = re.sub(r'[\s]+', ' ', name.lower())
    return normalized.strip()


def _normalize_name_for_fuzzy(name: str) -> str:
    """Produce a fuzzier form that keeps alphanumerics and apostrophes for n-gram shingles."""
    normalized = re.sub(r"[^a-z0-9' ]", ' ', _normalize_string_exact(name))
    normalized = normalized.strip()
    return re.sub(r'[\s]+', ' ', normalized)


def _name_entropy(normalized_name: str) -> float:
    """Approximate text specificity using Shannon entropy over characters.

    We strip spaces, count how often each character appears, and sum
    probability * -log2(probability). Short or repetitive names yield low
    entropy, which signals we should defer resolution to the LLM instead of
    trusting fuzzy similarity.
    """
    if not normalized_name:
        return 0.0

    counts: dict[str, int] = {}
    for char in normalized_name.replace(' ', ''):
        counts[char] = counts.get(char, 0) + 1

    total = sum(counts.values())
    if total == 0:
        return 0.0

    entropy = 0.0
    for count in counts.values():
        probability = count / total
        entropy -= probability * math.log2(probability)

    return entropy


def _has_high_entropy(normalized_name: str) -> bool:
    """Filter out very short or low-entropy names that are unreliable for fuzzy matching."""
    token_count = len(normalized_name.split())
    if len(normalized_name) < _MIN_NAME_LENGTH and token_count < _MIN_TOKEN_COUNT:
        return False

    return _name_entropy(normalized_name) >= _NAME_ENTROPY_THRESHOLD


def _shingles(normalized_name: str) -> set[str]:
    """Create 3-gram shingles from the normalized name for MinHash calculations."""
    cleaned = normalized_name.replace(' ', '')
    if len(cleaned) < 2:
        return {cleaned} if cleaned else set()

    return {cleaned[i : i + 3] for i in range(len(cleaned) - 2)}


def _hash_shingle(shingle: str, seed: int) -> int:
    """Generate a deterministic 64-bit hash for a shingle given the permutation seed."""
    digest = blake2b(f'{seed}:{shingle}'.encode(), digest_size=8)
    return int.from_bytes(digest.digest(), 'big')


def _minhash_signature(shingles: Iterable[str]) -> tuple[int, ...]:
    """Compute the MinHash signature for the shingle set across predefined permutations."""
    if not shingles:
        return tuple()

    seeds = range(_MINHASH_PERMUTATIONS)
    signature: list[int] = []
    for seed in seeds:
        min_hash = min(_hash_shingle(shingle, seed) for shingle in shingles)
        signature.append(min_hash)

    return tuple(signature)


def _lsh_bands(signature: Iterable[int]) -> list[tuple[int, ...]]:
    """Split the MinHash signature into fixed-size bands for locality-sensitive hashing."""
    signature_list = list(signature)
    if not signature_list:
        return []

    bands: list[tuple[int, ...]] = []
    for start in range(0, len(signature_list), _MINHASH_BAND_SIZE):
        band = tuple(signature_list[start : start + _MINHASH_BAND_SIZE])
        if len(band) == _MINHASH_BAND_SIZE:
            bands.append(band)
    return bands


def _jaccard_similarity(a: set[str], b: set[str]) -> float:
    """Return the Jaccard similarity between two shingle sets, handling empty edge cases."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    intersection = len(a.intersection(b))
    union = len(a.union(b))
    return intersection / union if union else 0.0


@lru_cache(maxsize=512)
def _cached_shingles(name: str) -> set[str]:
    """Cache shingle sets per normalized name to avoid recomputation within a worker."""
    return _shingles(name)


@dataclass
class DedupCandidateIndexes:
    """Precomputed lookup structures that drive entity deduplication heuristics."""

    existing_nodes: list[EntityNode]
    nodes_by_uuid: dict[str, EntityNode]
    normalized_existing: defaultdict[str, list[EntityNode]]
    shingles_by_candidate: dict[str, set[str]]
    lsh_buckets: defaultdict[tuple[int, tuple[int, ...]], list[str]]


@dataclass
class DedupResolutionState:
    """Mutable resolution bookkeeping shared across deterministic and LLM passes."""

    resolved_nodes: list[EntityNode | None]
    uuid_map: dict[str, str]
    unresolved_indices: list[int]
    duplicate_pairs: list[tuple[EntityNode, EntityNode]] = field(default_factory=list)


def _build_candidate_indexes(existing_nodes: list[EntityNode]) -> DedupCandidateIndexes:
    """Precompute exact and fuzzy lookup structures once per dedupe run."""
    normalized_existing: defaultdict[str, list[EntityNode]] = defaultdict(list)
    nodes_by_uuid: dict[str, EntityNode] = {}
    shingles_by_candidate: dict[str, set[str]] = {}
    lsh_buckets: defaultdict[tuple[int, tuple[int, ...]], list[str]] = defaultdict(list)

    for candidate in existing_nodes:
        normalized = _normalize_string_exact(candidate.name)
        normalized_existing[normalized].append(candidate)
        nodes_by_uuid[candidate.uuid] = candidate

        shingles = _cached_shingles(_normalize_name_for_fuzzy(candidate.name))
        shingles_by_candidate[candidate.uuid] = shingles

        signature = _minhash_signature(shingles)
        for band_index, band in enumerate(_lsh_bands(signature)):
            lsh_buckets[(band_index, band)].append(candidate.uuid)

    return DedupCandidateIndexes(
        existing_nodes=existing_nodes,
        nodes_by_uuid=nodes_by_uuid,
        normalized_existing=normalized_existing,
        shingles_by_candidate=shingles_by_candidate,
        lsh_buckets=lsh_buckets,
    )


def _resolve_with_similarity(
    extracted_nodes: list[EntityNode],
    indexes: DedupCandidateIndexes,
    state: DedupResolutionState,
) -> None:
    """Attempt deterministic resolution using exact name hits and fuzzy MinHash comparisons."""
    for idx, node in enumerate(extracted_nodes):
        normalized_exact = _normalize_string_exact(node.name)
        normalized_fuzzy = _normalize_name_for_fuzzy(node.name)

        if not _has_high_entropy(normalized_fuzzy):
            state.unresolved_indices.append(idx)
            continue

        existing_matches = indexes.normalized_existing.get(normalized_exact, [])
        if len(existing_matches) == 1:
            match = existing_matches[0]
            state.resolved_nodes[idx] = match
            state.uuid_map[node.uuid] = match.uuid
            if match.uuid != node.uuid:
                state.duplicate_pairs.append((node, match))
            continue
        if len(existing_matches) > 1:
            state.unresolved_indices.append(idx)
            continue

        shingles = _cached_shingles(normalized_fuzzy)
        signature = _minhash_signature(shingles)
        candidate_ids: set[str] = set()
        for band_index, band in enumerate(_lsh_bands(signature)):
            candidate_ids.update(indexes.lsh_buckets.get((band_index, band), []))

        best_candidate: EntityNode | None = None
        best_score = 0.0
        for candidate_id in candidate_ids:
            candidate_shingles = indexes.shingles_by_candidate.get(candidate_id, set())
            score = _jaccard_similarity(shingles, candidate_shingles)
            if score > best_score:
                best_score = score
                best_candidate = indexes.nodes_by_uuid.get(candidate_id)

        if best_candidate is not None and best_score >= _FUZZY_JACCARD_THRESHOLD:
            state.resolved_nodes[idx] = best_candidate
            state.uuid_map[node.uuid] = best_candidate.uuid
            if best_candidate.uuid != node.uuid:
                state.duplicate_pairs.append((node, best_candidate))
            continue

        state.unresolved_indices.append(idx)


__all__ = [
    'DedupCandidateIndexes',
    'DedupResolutionState',
    '_normalize_string_exact',
    '_normalize_name_for_fuzzy',
    '_has_high_entropy',
    '_minhash_signature',
    '_lsh_bands',
    '_jaccard_similarity',
    '_cached_shingles',
    '_FUZZY_JACCARD_THRESHOLD',
    '_build_candidate_indexes',
    '_resolve_with_similarity',
]
