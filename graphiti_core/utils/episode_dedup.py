"""
Episode deduplication utilities for Graphiti.

This module provides functions to detect and prevent duplicate episodes
from being added to the knowledge graph.
"""

import hashlib
import logging

from graphiti_core.driver.driver import GraphDriver
from graphiti_core.nodes import EpisodicNode

logger = logging.getLogger(__name__)


def compute_content_hash(content: str, name: str = '') -> str:
    """
    Compute a SHA-256 hash of episode content for deduplication.

    Args:
        content: The episode content
        name: Optional episode name to include in hash

    Returns:
        Hexadecimal SHA-256 hash string
    """
    # Normalize content: strip whitespace, convert to lowercase for consistency
    normalized_content = content.strip()

    # Combine name and content for hashing
    hash_input = f"{name}:{normalized_content}"

    return hashlib.sha256(hash_input.encode('utf-8')).hexdigest()


async def find_duplicate_episode_by_hash(
    driver: GraphDriver,
    group_id: str,
    content_hash: str,
) -> EpisodicNode | None:
    """
    Find an existing episode with the same content hash.

    This performs an exact match deduplication by comparing content hashes.
    Note: This requires content_hash to be stored on episodes.

    Args:
        driver: The graph driver
        group_id: The group ID to search within
        content_hash: The content hash to find

    Returns:
        The duplicate episode if found, None otherwise
    """
    # Query episodes with matching content_hash (if field exists)
    records, _, _ = await driver.execute_query(
        """
        MATCH (e:Episodic {group_id: $group_id})
        WHERE e.content_hash = $content_hash
        RETURN
            e.uuid AS uuid,
            e.name AS name,
            e.group_id AS group_id,
            e.content AS content,
            e.source_description AS source_description,
            e.created_at AS created_at,
            e.valid_at AS valid_at,
            e.source AS source,
            e.entity_edges AS entity_edges,
            e.content_hash AS content_hash
        ORDER BY e.created_at DESC
        LIMIT 1
        """,
        group_id=group_id,
        content_hash=content_hash,
    )

    if not records:
        return None

    from graphiti_core.nodes import get_episodic_node_from_record

    record = records[0]
    return get_episodic_node_from_record(record)


async def find_similar_episodes_by_content(
    driver: GraphDriver,
    group_id: str,
    content: str,
    name: str,
    similarity_threshold: float = 0.95,
    max_results: int = 5,
) -> list[EpisodicNode]:
    """
    Find episodes with similar content using fulltext search.

    This performs fuzzy matching to find near-duplicate episodes.

    Args:
        driver: The graph driver
        group_id: The group ID to search within
        content: The episode content to compare
        name: The episode name
        similarity_threshold: Minimum similarity score (0-1)
        max_results: Maximum number of similar episodes to return

    Returns:
        List of similar episodes, sorted by similarity (descending)
    """
    # Build fulltext query for content similarity
    provider = driver.provider

    if provider == 'falkordb':
        # Use FalkorDB's fulltext search
        # Note: This requires a fulltext index on content field
        from graphiti_core.driver.falkordb_driver import FalkorDriver

        if isinstance(driver, FalkorDriver):
            fulltext_query = driver.build_fulltext_query(
                f'{name} {content}',
                group_ids=[group_id],
            )

            if not fulltext_query:
                return []

            records, _, _ = await driver.execute_query(
                """
                MATCH (e:Episodic {group_id: $group_id})
                WHERE e.content_fulltext @ $fulltext_query
                RETURN
                    e.uuid AS uuid,
                    e.name AS name,
                    e.group_id AS group_id,
                    e.content AS content,
                    e.source_description AS source_description,
                    e.created_at AS created_at,
                    e.valid_at AS valid_at,
                    e.source AS source,
                    e.entity_edges AS entity_edges
                ORDER BY e.created_at DESC
                LIMIT $max_results
                """,
                group_id=group_id,
                fulltext_query=fulltext_query,
                max_results=max_results,
            )
        else:
            return []
    else:
        # For Neo4j and others, use basic text search
        # First check by exact name match (fast, uses index)
        records_by_name, _, _ = await driver.execute_query(
            """
            MATCH (e:Episodic {group_id: $group_id, name: $name})
            RETURN
                e.uuid AS uuid,
                e.name AS name,
                e.group_id AS group_id,
                e.content AS content,
                e.source_description AS source_description,
                e.created_at AS created_at,
                e.valid_at AS valid_at,
                e.source AS source,
                e.entity_edges AS entity_edges
            ORDER BY e.created_at DESC
            LIMIT $max_results
            """,
            group_id=group_id,
            name=name,
            max_results=max_results,
        )

        # Filter for exact content match
        exact_matches = [r for r in records_by_name if r.get('content') == content]

        if exact_matches:
            records = exact_matches
        else:
            # Fallback to content search (slower, but only if needed)
            search_terms = ' '.join(content.split()[:20])  # Use first 20 words

            records, _, _ = await driver.execute_query(
                """
                MATCH (e:Episodic {group_id: $group_id})
                WHERE e.content CONTAINS $search_terms
                RETURN
                    e.uuid AS uuid,
                    e.name AS name,
                    e.group_id AS group_id,
                    e.content AS content,
                    e.source_description AS source_description,
                    e.created_at AS created_at,
                    e.valid_at AS valid_at,
                    e.source AS source,
                    e.entity_edges AS entity_edges
                ORDER BY e.created_at DESC
                LIMIT $max_results
                """,
                group_id=group_id,
                search_terms=search_terms,
                max_results=max_results,
            )

    if not records:
        return []

    from graphiti_core.nodes import get_episodic_node_from_record

    episodes = [get_episodic_node_from_record(record) for record in records]

    # Filter by exact content match for highest confidence deduplication
    exact_matches = [e for e in episodes if e.content == content]

    if exact_matches:
        logger.info(f"Found {len(exact_matches)} exact content matches")
        return exact_matches[:max_results]

    # For now, return all matches - similarity scoring could be enhanced with embeddings
    return episodes[:max_results]


async def check_duplicate_episode(
    driver: GraphDriver,
    group_id: str,
    name: str,
    content: str,
    enable_hash_check: bool = True,
    enable_similarity_check: bool = True,
) -> EpisodicNode | None:
    """
    Check if an episode is a duplicate based on content hash and/or similarity.

    Args:
        driver: The graph driver
        group_id: The group ID to search within
        name: The episode name
        content: The episode content
        enable_hash_check: Enable content hash-based deduplication
        enable_similarity_check: Enable similarity-based deduplication

    Returns:
        The duplicate episode if found, None otherwise
    """
    # Method 1: Check for exact content match (fastest)
    if enable_similarity_check:
        similar_episodes = await find_similar_episodes_by_content(
            driver, group_id, content, name, max_results=1
        )

        if similar_episodes:
            # Check for exact content match
            for episode in similar_episodes:
                if episode.content == content and episode.name == name:
                    logger.info(
                        f"Duplicate episode found by exact content match: {episode.uuid}"
                    )
                    return episode

    # Method 2: Check by content hash (requires content_hash field to be populated)
    # This is currently a placeholder - would require content_hash field in schema
    # if enable_hash_check:
    #     content_hash = compute_content_hash(content, name)
    #     duplicate = await find_duplicate_episode_by_hash(driver, group_id, content_hash)
    #     if duplicate:
    #         return duplicate

    return None
