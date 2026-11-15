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

from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

from graphiti_core.edges import EntityEdge
from graphiti_core.ingestion.change_set import ChangeSet, PatchNode
from graphiti_core.ingestion.state_store import DeltaUnit
from graphiti_core.nodes import EntityNode
from graphiti_core.search.search_config import DEFAULT_SEARCH_LIMIT
from graphiti_core.search.search_utils import DEFAULT_MIN_SCORE
from graphiti_core.utils.maintenance.dedup_helpers import (
    _FUZZY_JACCARD_THRESHOLD,
    DedupCandidateIndexes,
    _build_candidate_indexes,
    _cached_shingles,
    _has_high_entropy,
    _identity_key_for_node,
    _jaccard_similarity,
    _lsh_bands,
    _minhash_signature,
    _normalize_name_for_fuzzy,
    _normalize_string_exact,
    _promote_resolved_node,
)
from graphiti_core.utils.maintenance.node_operations import (
    NODE_DEDUP_CANDIDATE_LIMIT,
    NODE_DEDUP_COSINE_MIN_SCORE,
)

# The number of delta units for which the process keeps a digest.
DIGEST_CACHE_SIZE = 1024


def _unit_matrix(vectors: list[list[float]]) -> np.ndarray:
    """Return the vectors as rows with a length of 1."""
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2:
        return np.zeros((0, 0), dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def patch_as_node(node: EntityNode, patch: PatchNode) -> EntityNode:
    """Return the labels and the attributes of a patch as a node, for `_promote_resolved_node`."""
    return EntityNode(
        uuid=node.uuid,
        name=node.name,
        group_id=node.group_id,
        labels=list(patch.labels),
        attributes=dict(patch.attributes),
    )


def _fuzzy_query(name: str) -> tuple[set[str], list[tuple[int, ...]]] | None:
    """Return the shingles and the LSH bands of a name, or None when the name has low entropy."""
    normalized = _normalize_name_for_fuzzy(name)
    if not _has_high_entropy(normalized):
        return None
    shingles = _cached_shingles(normalized)
    return shingles, list(_lsh_bands(_minhash_signature(shingles)))


def _fuzzy_match(
    query: tuple[set[str], list[tuple[int, ...]]] | None, indexes: DedupCandidateIndexes
) -> str | None:
    """Return the UUID that the fuzzy rule of `_resolve_with_similarity` selects.

    The function changes no node.
    """
    if query is None:
        return None
    shingles, bands = query
    candidate_ids: set[str] = set()
    for band_index, band in enumerate(bands):
        candidate_ids.update(indexes.lsh_buckets.get((band_index, band), []))
    best_uuid: str | None = None
    best_score = 0.0
    for candidate_id in candidate_ids:
        score = _jaccard_similarity(
            shingles, indexes.shingles_by_candidate.get(candidate_id, set())
        )
        if score > best_score:
            best_score = score
            best_uuid = candidate_id
    return best_uuid if best_score >= _FUZZY_JACCARD_THRESHOLD else None


@dataclass
class _UnitDigest:
    """The parts of one final change set that the repair compares with a new unit.

    A digest has no full record. The repair reads a record that matches from the graph.
    """

    seq: int
    status: str
    nodes: list[EntityNode]
    node_indexes: DedupCandidateIndexes
    node_vector_uuids: list[str]
    node_vectors: np.ndarray
    edges: list[tuple[str, str, str]]
    edge_vector_uuids: list[str]
    edge_vectors: np.ndarray
    patches: list[PatchNode]


def _digest(unit: DeltaUnit) -> _UnitDigest:
    change_set = ChangeSet.from_bytes(unit.payload)
    nodes = [
        node.model_copy(update={'name_embedding': None, 'summary': ''})
        for node in change_set.create_nodes
    ]
    nodes_with_vector = [node for node in change_set.create_nodes if node.name_embedding]
    edges_with_vector = [edge for edge in change_set.create_edges if edge.fact_embedding]
    return _UnitDigest(
        seq=unit.seq,
        status=unit.status,
        nodes=nodes,
        node_indexes=_build_candidate_indexes(nodes),
        node_vector_uuids=[node.uuid for node in nodes_with_vector],
        node_vectors=_unit_matrix([node.name_embedding or [] for node in nodes_with_vector]),
        edges=[
            (edge.uuid, edge.source_node_uuid, edge.target_node_uuid)
            for edge in change_set.create_edges
        ],
        edge_vector_uuids=[edge.uuid for edge in edges_with_vector],
        edge_vectors=_unit_matrix([edge.fact_embedding or [] for edge in edges_with_vector]),
        patches=[
            operation for operation in change_set.operations if isinstance(operation, PatchNode)
        ],
    )


_digest_cache: OrderedDict[tuple[str, int, str, int, int], _UnitDigest] = OrderedDict()


def load_delta_digests(group_id: str, units: list[DeltaUnit]) -> 'Delta':
    """Return the delta of a commit from the final change sets that the store returned.

    The final change set of a unit does not change. The process thus keeps the digest
    of each payload, and a later commit does not decode the payload again.
    """
    digests: list[_UnitDigest] = []
    for unit in sorted(units, key=lambda unit: unit.seq):
        key = (group_id, unit.seq, unit.status, len(unit.payload), hash(unit.payload))
        digest = _digest_cache.get(key)
        if digest is None:
            digest = _digest(unit)
            _digest_cache[key] = digest
            while len(_digest_cache) > DIGEST_CACHE_SIZE:
                _digest_cache.popitem(last=False)
        else:
            _digest_cache.move_to_end(key)
        digests.append(digest)
    return Delta(digests)


def _stack(matrices: list[np.ndarray]) -> np.ndarray:
    matrices = [matrix for matrix in matrices if matrix.size]
    if not matrices:
        return np.zeros((0, 0), dtype=np.float32)
    width = matrices[0].shape[1]
    return np.vstack([matrix for matrix in matrices if matrix.shape[1] == width])


def _stack_uuids(uuid_lists: list[list[str]], matrices: list[np.ndarray]) -> list[str]:
    matrices_with_rows = [
        (uuids, matrix) for uuids, matrix in zip(uuid_lists, matrices, strict=True) if matrix.size
    ]
    if not matrices_with_rows:
        return []
    width = matrices_with_rows[0][1].shape[1]
    return [
        uuid for uuids, matrix in matrices_with_rows if matrix.shape[1] == width for uuid in uuids
    ]


def _top_matches(
    vectors: list[list[float] | None],
    candidate_uuids: list[str],
    candidate_vectors: np.ndarray,
    min_score: float,
    limit: int,
) -> list[list[str]]:
    """Return, for each vector, the candidates with a cosine similarity of `min_score` or more."""
    results: list[list[str]] = [[] for _ in vectors]
    rows = [index for index, vector in enumerate(vectors) if vector]
    if not rows or not candidate_uuids:
        return results
    queries = _unit_matrix([vectors[index] or [] for index in rows])
    if queries.shape[1] != candidate_vectors.shape[1]:
        return results
    scores = queries @ candidate_vectors.T
    for row, index in enumerate(rows):
        order = np.argsort(-scores[row])
        results[index] = [
            candidate_uuids[column] for column in order[:limit] if scores[row, column] >= min_score
        ]
    return results


@dataclass
class Delta:
    """The records and the patches of the units that committed after a snapshot."""

    digests: list[_UnitDigest] = field(default_factory=list)
    patches: dict[str, list[PatchNode]] = field(init=False)
    created_node_uuids: set[str] = field(init=False)

    def __post_init__(self) -> None:
        self.patches = {}
        self.created_node_uuids = set()
        for digest in self.digests:
            for patch in digest.patches:
                self.patches.setdefault(patch.node_uuid, []).append(patch)
            self.created_node_uuids.update(node.uuid for node in digest.nodes)

    @property
    def unit_count(self) -> int:
        return len(self.digests)

    def _patched_copy(self, node: EntityNode) -> EntityNode:
        patches = self.patches.get(node.uuid)
        if not patches:
            return node
        copy = node.model_copy(deep=True)
        for patch in patches:
            _promote_resolved_node(patch_as_node(copy, patch), copy)
        return copy

    def match_nodes(
        self,
        nodes: list[EntityNode],
        identity_properties: dict[str, list[str]] | None = None,
    ) -> list[list[str]]:
        """Return, for each node, the UUIDs of the delta nodes that can be the same entity.

        A delta node matches by the same normalized name, the same identity key, a fuzzy
        name match, or a cosine similarity of the name embeddings of
        `NODE_DEDUP_COSINE_MIN_SCORE` or more. A match by one of the first three rules
        comes first. The result has at most `NODE_DEDUP_CANDIDATE_LIMIT` UUIDs for a node.
        """
        results: list[list[str]] = [[] for _ in nodes]
        if not self.digests:
            return results

        for index, node in enumerate(nodes):
            normalized_name = _normalize_string_exact(node.name)
            identity_key = _identity_key_for_node(node, identity_properties)
            fuzzy_query = _fuzzy_query(node.name)
            for digest in self.digests:
                for candidate in digest.node_indexes.normalized_existing.get(normalized_name, []):
                    results[index].append(candidate.uuid)
                if identity_key is not None:
                    for candidate in digest.nodes:
                        patched = self._patched_copy(candidate)
                        if _identity_key_for_node(patched, identity_properties) == identity_key:
                            results[index].append(candidate.uuid)
                fuzzy_match = _fuzzy_match(fuzzy_query, digest.node_indexes)
                if fuzzy_match is not None:
                    results[index].append(fuzzy_match)

        similar = _top_matches(
            [node.name_embedding for node in nodes],
            _stack_uuids(
                [digest.node_vector_uuids for digest in self.digests],
                [digest.node_vectors for digest in self.digests],
            ),
            _stack([digest.node_vectors for digest in self.digests]),
            NODE_DEDUP_COSINE_MIN_SCORE,
            NODE_DEDUP_CANDIDATE_LIMIT,
        )
        own_uuids = {node.uuid for node in nodes}
        return [
            [
                uuid
                for uuid in dict.fromkeys([*rule_matches, *similar_matches])
                if uuid not in own_uuids
            ][:NODE_DEDUP_CANDIDATE_LIMIT]
            for rule_matches, similar_matches in zip(results, similar, strict=True)
        ]

    def patched_identity_matches(
        self,
        nodes: list[EntityNode],
        identity_properties: dict[str, list[str]] | None,
    ) -> list[list[str]]:
        """Return, for each node, the stored nodes that a patch of the delta gave its identity key."""
        results: list[list[str]] = [[] for _ in nodes]
        if not identity_properties or not self.patches:
            return results
        keys = [_identity_key_for_node(node, identity_properties) for node in nodes]
        if not any(key is not None for key in keys):
            return results
        for node_uuid in self.patches:
            if node_uuid in self.created_node_uuids:
                continue
            patched = self._patched_copy(
                EntityNode(uuid=node_uuid, name='', group_id=nodes[0].group_id, labels=[])
            )
            patched_key = _identity_key_for_node(patched, identity_properties)
            if patched_key is None:
                continue
            for index, key in enumerate(keys):
                if key == patched_key:
                    results[index].append(node_uuid)
        return results

    def edges_between(self, edges: list[EntityEdge]) -> list[list[str]]:
        """Return, for each edge, the delta edges between its two nodes, in either direction."""
        by_pair: dict[frozenset[str], list[str]] = {}
        for digest in self.digests:
            for edge_uuid, source_uuid, target_uuid in digest.edges:
                by_pair.setdefault(frozenset((source_uuid, target_uuid)), []).append(edge_uuid)
        return [
            list(by_pair.get(frozenset((edge.source_node_uuid, edge.target_node_uuid)), []))
            for edge in edges
        ]

    def similar_edges(self, edges: list[EntityEdge]) -> list[list[str]]:
        """Return, for each edge, the delta edges with a similar fact embedding.

        The limits are those of the graph-wide search: a cosine similarity of
        `DEFAULT_MIN_SCORE` or more, and at most `DEFAULT_SEARCH_LIMIT` edges.
        """
        return _top_matches(
            [edge.fact_embedding for edge in edges],
            _stack_uuids(
                [digest.edge_vector_uuids for digest in self.digests],
                [digest.edge_vectors for digest in self.digests],
            ),
            _stack([digest.edge_vectors for digest in self.digests]),
            DEFAULT_MIN_SCORE,
            DEFAULT_SEARCH_LIMIT,
        )
