"""Core document synchronization logic.

This module manages the synchronization of markdown documents from a corpus directory
into the Graphiti knowledge graph. It implements intelligent change detection and
diff-based updates to maintain an evolving knowledge base.

Key Concepts:
    Dual Episode Pattern:
        - Full sync episodes: Authoritative document snapshots (content mutated on updates)
        - Diff episodes: Semantic summaries of changes for knowledge graph ingestion

    Change Detection:
        - SHA256 content hashing to detect actual changes
        - Query graph for previous state (no local cache)
        - Skip sync if content unchanged

    Diff Strategy:
        - Compare against latest full sync episode (authoritative snapshot)
        - Generate unified diff with full context (100 lines)
        - LLM summarization produces semantic change description
        - Update full sync episode content via MERGE (mutation for optimization)

    File Operations:
        - Rename: Update all episode metadata URIs via Cypher
        - Delete: No-op (append-only graph preserves history)

Episode Metadata Schema:
    {
        "document_uri": str,      # Relative to corpus root
        "content_hash": str,      # SHA256 with prefix
        "sync_type": str,         # "full" | "diff"
        "sync_timestamp": str     # ISO 8601 UTC timestamp
    }
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType, EpisodicNode

from .diff_generator import compute_content_hash, generate_unified_diff
from .diff_summarizer import summarize_diff

logger = logging.getLogger(__name__)


class DocumentSyncManager:
    """Manages synchronization of documents from corpus directory to knowledge graph."""

    def __init__(
        self,
        corpus_path: Path,
        graphiti: Graphiti,
        group_id: str,
    ):
        """Initialize document sync manager.

        Args:
            corpus_path: Root directory containing documents to sync
            graphiti: Graphiti client instance
            group_id: Graph partition identifier
        """
        self.corpus_path = Path(corpus_path)
        self.graphiti = graphiti
        self.group_id = group_id

    def _get_relative_uri(self, file_path: Path) -> str:
        """Convert absolute file path to relative URI.

        Args:
            file_path: Absolute path to file

        Returns:
            Relative path from corpus root (e.g., 'tasks/active.md')
        """
        return str(file_path.relative_to(self.corpus_path))

    async def get_latest_episode_for_document(self, uri: str) -> EpisodicNode | None:
        """Query graph for most recent episode with matching document URI.

        Args:
            uri: Document URI to search for

        Returns:
            Latest EpisodicNode for this document, or None if not found
        """
        # Build Cypher query to find episodes with this URI in metadata
        # Use string matching on JSON since metadata is stored as JSON string
        # Pattern: "document_uri": "uri_value"
        search_pattern = f'"document_uri": "{uri}"'

        query = """
        MATCH (e:Episodic)
        WHERE e.group_id = $group_id
          AND e.metadata IS NOT NULL
          AND e.metadata CONTAINS $search_pattern
        RETURN e
        ORDER BY e.created_at DESC
        LIMIT 1
        """

        try:
            # execute_query returns (records, summary, keys)
            records, _, _ = await self.graphiti.driver.execute_query(
                query,
                group_id=self.group_id,
                search_pattern=search_pattern,
            )

            if records and len(records) > 0:
                # Parse the episode node from the result
                record = records[0]
                episode_data = record['e']

                # Convert Neo4j datetime objects to Python datetime
                # Neo4j returns neo4j.time.DateTime which needs conversion
                created_at = episode_data['created_at']
                if hasattr(created_at, 'to_native'):
                    created_at = created_at.to_native()

                valid_at = episode_data['valid_at']
                if hasattr(valid_at, 'to_native'):
                    valid_at = valid_at.to_native()

                # Convert Neo4j node to EpisodicNode
                return EpisodicNode(
                    uuid=episode_data['uuid'],
                    name=episode_data['name'],
                    group_id=episode_data['group_id'],
                    created_at=created_at,
                    source=EpisodeType(episode_data['source']),
                    source_description=episode_data.get('source_description', ''),
                    content=episode_data['content'],
                    valid_at=valid_at,
                    entity_edges=episode_data.get('entity_edges', []),
                    metadata=json.loads(episode_data['metadata'])
                    if episode_data.get('metadata')
                    else {},
                )
        except Exception as e:
            logger.error(f'Error querying for document {uri}: {e}')
            return None

        return None

    async def get_latest_full_sync_for_document(self, uri: str) -> EpisodicNode | None:
        """Get most recent full sync episode for document (authoritative snapshot).

        Args:
            uri: Document URI relative to corpus root

        Returns:
            Latest full sync EpisodicNode for this document, or None if not found
        """
        search_pattern = f'"document_uri": "{uri}"'
        full_sync_pattern = f'"sync_type": "full"'

        query = """
        MATCH (e:Episodic)
        WHERE e.group_id = $group_id
          AND e.metadata IS NOT NULL
          AND e.metadata CONTAINS $search_pattern
          AND e.metadata CONTAINS $full_sync_pattern
        RETURN e
        ORDER BY e.created_at DESC
        LIMIT 1
        """

        try:
            records, _, _ = await self.graphiti.driver.execute_query(
                query,
                group_id=self.group_id,
                search_pattern=search_pattern,
                full_sync_pattern=full_sync_pattern,
            )

            if records and len(records) > 0:
                record = records[0]
                episode_data = record['e']

                # Convert Neo4j datetime objects
                created_at = episode_data['created_at']
                if hasattr(created_at, 'to_native'):
                    created_at = created_at.to_native()

                valid_at = episode_data['valid_at']
                if hasattr(valid_at, 'to_native'):
                    valid_at = valid_at.to_native()

                return EpisodicNode(
                    uuid=episode_data['uuid'],
                    name=episode_data['name'],
                    group_id=episode_data['group_id'],
                    created_at=created_at,
                    source=EpisodeType(episode_data['source']),
                    source_description=episode_data.get('source_description', ''),
                    content=episode_data['content'],
                    valid_at=valid_at,
                    entity_edges=episode_data.get('entity_edges', []),
                    metadata=json.loads(episode_data['metadata'])
                    if episode_data.get('metadata')
                    else {},
                )
        except Exception as e:
            logger.error(f'Error querying for full sync of document {uri}: {e}')
            return None

        return None

    async def sync_document(self, file_path: Path) -> dict[str, Any]:
        """Sync a single document to the knowledge graph.

        Args:
            file_path: Absolute path to document file

        Returns:
            Sync result with status and metadata
        """
        uri = self._get_relative_uri(file_path)

        try:
            # Read file content
            content = file_path.read_text(encoding='utf-8')
            new_hash = compute_content_hash(content)

            # Get latest episode for this document
            latest_episode = await self.get_latest_episode_for_document(uri)

            # Check if content has changed
            if latest_episode:
                old_hash = latest_episode.metadata.get('content_hash')
                if old_hash == new_hash:
                    return {
                        'status': 'skipped',
                        'uri': uri,
                        'reason': 'unchanged',
                    }

                # Get full sync episode (authoritative document snapshot)
                full_sync_episode = await self.get_latest_full_sync_for_document(uri)

                if full_sync_episode:
                    # Generate REAL diff: old document vs new document
                    old_content = full_sync_episode.content
                    diff_content = generate_unified_diff(old_content, content, uri)

                    # Summarize the diff for ingestion (LLMs can't resist context lines)
                    summary = await summarize_diff(
                        self.graphiti.llm_client,
                        diff_content,
                        uri,
                    )

                    # Prefix with document URI for ingestion context
                    episode_body = f'Document: {uri}\n\n{summary}'
                    sync_type = 'diff'

                    # Update full sync episode with new content (mutation for snapshot)
                    full_sync_episode.content = content
                    await full_sync_episode.save(self.graphiti.driver)
                else:
                    # No full sync found - treat as new document
                    episode_body = f'Document: {uri}\n\n{content}'
                    sync_type = 'full'
            else:
                # First sync ever
                episode_body = f'Document: {uri}\n\n{content}'
                sync_type = 'full'

            # Create episode metadata
            metadata = {
                'document_uri': uri,
                'content_hash': new_hash,
                'sync_type': sync_type,
                'sync_timestamp': datetime.now(timezone.utc).isoformat(),
            }

            # Add episode to graph
            await self.graphiti.add_episode(
                name=f'Document sync: {uri}',
                episode_body=episode_body,
                source_description=f'Document sync from {uri}',
                reference_time=datetime.now(timezone.utc),
                source=EpisodeType.text,
                group_id=self.group_id,
                metadata=metadata,
            )

            return {
                'status': 'synced',
                'uri': uri,
                'sync_type': sync_type,
                'content_hash': new_hash,
            }

        except Exception as e:
            logger.error(f'Error syncing document {uri}: {e}')
            return {
                'status': 'error',
                'uri': uri,
                'error': str(e),
            }

    async def handle_rename(self, old_uri: str, new_uri: str) -> dict[str, Any]:
        """Update all episode metadata URIs when file is renamed.

        Args:
            old_uri: Previous document URI
            new_uri: New document URI

        Returns:
            Result with count of updated episodes
        """
        # Find all episodes with the old URI
        # We'll update them one by one using Python to avoid APOC dependency
        search_pattern = f'"document_uri": "{old_uri}"'

        query_find = """
        MATCH (e:Episodic)
        WHERE e.group_id = $group_id
          AND e.metadata IS NOT NULL
          AND e.metadata CONTAINS $search_pattern
        RETURN e.uuid as uuid, e.metadata as metadata
        """

        try:
            # Find episodes to update
            # execute_query returns (records, summary, keys)
            records, _, _ = await self.graphiti.driver.execute_query(
                query_find,
                group_id=self.group_id,
                search_pattern=search_pattern,
            )

            if not records:
                return {
                    'status': 'renamed',
                    'old_uri': old_uri,
                    'new_uri': new_uri,
                    'updated_count': 0,
                }

            # Update each episode's metadata
            updated_count = 0
            for record in records:
                episode_uuid = record['uuid']
                old_metadata_json = record['metadata']

                # Parse metadata, update URI, re-serialize
                metadata = json.loads(old_metadata_json)
                metadata['document_uri'] = new_uri
                new_metadata_json = json.dumps(metadata)

                # Update the episode
                query_update = """
                MATCH (e:Episodic)
                WHERE e.uuid = $uuid
                SET e.metadata = $new_metadata
                """

                await self.graphiti.driver.execute_query(
                    query_update,
                    uuid=episode_uuid,
                    new_metadata=new_metadata_json,
                )
                updated_count += 1

            return {
                'status': 'renamed',
                'old_uri': old_uri,
                'new_uri': new_uri,
                'updated_count': updated_count,
            }

        except Exception as e:
            logger.error(f'Error renaming document {old_uri} → {new_uri}: {e}')
            return {
                'status': 'error',
                'old_uri': old_uri,
                'new_uri': new_uri,
                'error': str(e),
            }

    async def sync_all(self) -> dict[str, Any]:
        """Sync all markdown documents in corpus directory.

        Returns:
            Statistics: synced, skipped, errors
        """
        synced = 0
        skipped = 0
        errors = []

        # Find all .md files in corpus
        md_files = list(self.corpus_path.rglob('*.md'))

        for file_path in md_files:
            result = await self.sync_document(file_path)

            if result['status'] == 'synced':
                synced += 1
            elif result['status'] == 'skipped':
                skipped += 1
            elif result['status'] == 'error':
                errors.append(result)

        return {
            'synced': synced,
            'skipped': skipped,
            'errors': errors,
            'total': len(md_files),
        }
