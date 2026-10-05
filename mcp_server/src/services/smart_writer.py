"""Smart memory writer for multi-group knowledge storage.

This module provides the SmartMemoryWriter which coordinates
memory classification and writes to appropriate group_ids.
"""

import asyncio
import logging
from typing import TYPE_CHECKING
from uuid import NAMESPACE_OID, uuid5

from graphiti_core.nodes import EpisodeType

from classifiers.base import ClassificationResult, MemoryCategory, MemoryClassifier
from utils.project_config import ProjectConfig

if TYPE_CHECKING:
    from services.queue_service import QueueService

logger = logging.getLogger(__name__)


class WriteResult:
    """Result of a smart memory write operation.

    Attributes:
        success: Whether the write operation succeeded
        written_to: List of group_ids that were written to
        category: The memory category that was determined
        classification: The full classification result
        error: Error message if success is False
        task_id: The background task ID for tracking
    """

    def __init__(
        self,
        success: bool,
        written_to: list[str] | None = None,
        category: str | None = None,
        classification: ClassificationResult | None = None,
        error: str | None = None,
        task_id: str | None = None,
    ):
        self.success = success
        self.written_to = written_to or []
        self.category = category
        self.classification = classification
        self.error = error
        self.task_id = task_id

    def __repr__(self) -> str:
        if self.success:
            return (
                f'WriteResult(success=True, task_id={self.task_id}, '
                f'written_to={self.written_to}, category={self.category})'
            )
        else:
            return f'WriteResult(success=False, error={self.error})'


class SmartMemoryWriter:
    """Coordinates memory classification and multi-group writes.

    This class uses a MemoryClassifier to determine where memories should
    be stored, then writes them to the appropriate group_ids via the
    QueueService for background processing.
    """

    def __init__(
        self,
        classifier: MemoryClassifier,
        graphiti_client,
        queue_service: 'QueueService',
        entity_types: dict | None = None,
        edge_types: dict | None = None,
        edge_type_map: dict | None = None,
    ):
        """Initialize the smart memory writer.

        Args:
            classifier: The memory classifier to use
            graphiti_client: Graphiti client for storage operations
            queue_service: Queue service for background episode processing
            entity_types: Custom entity types for episode processing
        """
        self.classifier = classifier
        self.graphiti_client = graphiti_client
        self.queue_service = queue_service
        self.entity_types = entity_types or {}
        self.edge_types = edge_types
        self.edge_type_map = edge_type_map
        logger.info(f'SmartMemoryWriter initialized with {type(classifier).__name__}')
        # Strong references to spawned background tasks (GC guard)
        self._background_tasks: set[asyncio.Task] = set()

    async def add_memory(
        self,
        name: str,
        episode_body: str,
        project_config: ProjectConfig,
        metadata: dict | None = None,
        uuid: str | None = None,
        reference_time=None,
        excluded_entity_types: list[str] | None = None,
        custom_extraction_instructions: str | None = None,
        previous_episode_uuids: list[str] | None = None,
        update_communities: bool = False,
        saga: str | None = None,
        saga_previous_episode_uuid: str | None = None,
        edge_types: dict | None = None,
        edge_type_map: dict | None = None,
    ) -> WriteResult:
        """Intelligently write memory to appropriate groups (async, returns immediately).

        This method returns immediately after spawning a background task that:
        1. Classifies the memory using LLM or rules
        2. Routes to appropriate groups (project, shared, or both)
        3. Submits to queue for background processing

        Args:
            name: Name of the memory episode
            episode_body: Content of the memory
            project_config: Project configuration with group settings
            metadata: Optional metadata (e.g., timestamp, source)
            uuid: Optional UUID for the episode

        Returns:
            WriteResult with task_id for tracking (processing happens in background)
        """
        # Generate a unique task ID for tracking
        task_id = f'{name}_{uuid or id(asyncio.current_task())}_{asyncio.get_event_loop().time()}'

        # Create background task for classification and queuing; keep a strong
        # reference so CPython cannot garbage-collect it mid-execution.
        task = asyncio.create_task(
            self._classify_and_queue(
                name=name,
                episode_body=episode_body,
                project_config=project_config,
                metadata=metadata,
                uuid=uuid,
                task_id=task_id,
                reference_time=reference_time,
                excluded_entity_types=excluded_entity_types,
                custom_extraction_instructions=custom_extraction_instructions,
                previous_episode_uuids=previous_episode_uuids,
                update_communities=update_communities,
                saga=saga,
                saga_previous_episode_uuid=saga_previous_episode_uuid,
                edge_types=edge_types,
                edge_type_map=edge_type_map,
            )
        )
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

        logger.info(
            f"Memory '{name}' queued for background processing (task_id: {task_id[:20]}...)"
        )

        return WriteResult(
            success=True,
            task_id=task_id,
        )

    async def _classify_and_queue(
        self,
        name: str,
        episode_body: str,
        project_config: ProjectConfig,
        metadata: dict | None,
        uuid: str | None,
        task_id: str,
        reference_time=None,
        excluded_entity_types: list[str] | None = None,
        custom_extraction_instructions: str | None = None,
        previous_episode_uuids: list[str] | None = None,
        update_communities: bool = False,
        saga: str | None = None,
        saga_previous_episode_uuid: str | None = None,
        edge_types: dict | None = None,
        edge_type_map: dict | None = None,
    ) -> None:
        """Background task that classifies and queues the memory for processing.

        Args:
            name: Name of the memory episode
            episode_body: Content of the memory
            project_config: Project configuration with group settings
            metadata: Optional metadata (e.g., timestamp, source)
            uuid: Optional UUID for the episode
            task_id: Task ID for tracking
        """

        # Fan-out uuid derivation: when one caller uuid routes to
        # multiple groups, derive a deterministic per-group uuid so the
        # second insert cannot collide with the first (single-graph
        # drivers share one uuid space).
        # Initialized before the try: the classification-failure fail-safe
        # calls route_uuid too, and the closure must not reference an
        # unassigned variable when classify itself raised.
        fanout = False

        def route_uuid(group: str) -> str | None:
            # Derive only on multi-group fan-out: a single-group write keeps
            # the caller's uuid verbatim (get_episode_entities /
            # delete_episode / previous_episode_uuids reference it).
            if uuid is None or not fanout:
                return None
            return str(uuid5(NAMESPACE_OID, f'{uuid}:{group}'))

        try:
            # Step 1: Classify the memory
            logger.debug(f'[{task_id[:20]}...] Classifying memory: {name}')
            classification = await self.classifier.classify(
                episode_body=episode_body, project_config=project_config
            )

            # Single-group writes keep the caller's uuid; multi-group fan-out
            # derives per-group ones (route_uuid reads this via late binding).
            fanout = classification.category != MemoryCategory.PROJECT_SPECIFIC

            logger.debug(
                f'[{task_id[:20]}...] Classification result: {classification.category.value}, '
                f'confidence={classification.confidence}, '
                f'reasoning={classification.reasoning}'
            )

            # Step 2: Route based on classification and queue for processing
            if classification.category == MemoryCategory.SHARED:
                # Queue for all shared groups
                for shared_gid in project_config.shared_group_ids:
                    await self._queue_for_group(
                        name=name,
                        episode_body=episode_body,
                        group_id=shared_gid,
                        group_uuid=route_uuid(shared_gid),
                        metadata=metadata,
                        uuid=uuid,
                        reference_time=reference_time,
                        excluded_entity_types=excluded_entity_types,
                        custom_extraction_instructions=custom_extraction_instructions,
                        previous_episode_uuids=previous_episode_uuids,
                        update_communities=update_communities,
                        saga=saga,
                        saga_previous_episode_uuid=saga_previous_episode_uuid,
                    )
                    logger.debug(f'[{task_id[:20]}...] Queued for shared group: {shared_gid}')

            elif classification.category == MemoryCategory.PROJECT_SPECIFIC:
                # Queue for project group only
                await self._queue_for_group(
                    name=name,
                    episode_body=episode_body,
                    group_id=project_config.group_id,
                    group_uuid=route_uuid(project_config.group_id),
                    metadata=metadata,
                    uuid=uuid,
                    reference_time=reference_time,
                    excluded_entity_types=excluded_entity_types,
                    custom_extraction_instructions=custom_extraction_instructions,
                    previous_episode_uuids=previous_episode_uuids,
                    update_communities=update_communities,
                    saga=saga,
                    saga_previous_episode_uuid=saga_previous_episode_uuid,
                    edge_types=edge_types if edge_types is not None else self.edge_types,
                    edge_type_map=edge_type_map
                    if edge_type_map is not None
                    else self.edge_type_map,
                )
                logger.debug(
                    f'[{task_id[:20]}...] Queued for project group: {project_config.group_id}'
                )

            elif classification.category == MemoryCategory.MIXED:
                # Queue for both project and shared groups with content splitting
                # Use split content if available, otherwise queue full content for both

                # Check if we have split content
                if classification.shared_part or classification.project_part:
                    # Content splitting is available
                    shared_content = classification.shared_part or episode_body
                    project_content = classification.project_part or episode_body

                    logger.debug(
                        f'[{task_id[:20]}...] Using split content: shared={len(shared_content)} chars, '
                        f'project={len(project_content)} chars'
                    )

                    # Queue shared part for shared groups
                    for shared_gid in project_config.shared_group_ids:
                        await self._queue_for_group(
                            name=name,
                            episode_body=shared_content,
                            group_id=shared_gid,
                            group_uuid=route_uuid(shared_gid),
                            metadata=metadata,
                            uuid=uuid,
                            reference_time=reference_time,
                            excluded_entity_types=excluded_entity_types,
                            custom_extraction_instructions=custom_extraction_instructions,
                            previous_episode_uuids=previous_episode_uuids,
                            update_communities=update_communities,
                            saga=saga,
                            saga_previous_episode_uuid=saga_previous_episode_uuid,
                        )
                        logger.debug(f'[{task_id[:20]}...] Queued shared part for: {shared_gid}')

                    # Queue project part for project group
                    await self._queue_for_group(
                        name=name,
                        episode_body=project_content,
                        group_id=project_config.group_id,
                        group_uuid=route_uuid(project_config.group_id),
                        metadata=metadata,
                        uuid=uuid,
                        reference_time=reference_time,
                        excluded_entity_types=excluded_entity_types,
                        custom_extraction_instructions=custom_extraction_instructions,
                        previous_episode_uuids=previous_episode_uuids,
                        update_communities=update_communities,
                        saga=saga,
                        saga_previous_episode_uuid=saga_previous_episode_uuid,
                    )
                    logger.debug(
                        f'[{task_id[:20]}...] Queued project part for: {project_config.group_id}'
                    )
                else:
                    # No split content available, queue full content for both
                    logger.debug(
                        f'[{task_id[:20]}...] No split content available, '
                        'queueing full content for both'
                    )

                    # Queue for project group
                    await self._queue_for_group(
                        name=name,
                        episode_body=episode_body,
                        group_id=project_config.group_id,
                        group_uuid=route_uuid(project_config.group_id),
                        metadata=metadata,
                        uuid=uuid,
                        reference_time=reference_time,
                        excluded_entity_types=excluded_entity_types,
                        custom_extraction_instructions=custom_extraction_instructions,
                        previous_episode_uuids=previous_episode_uuids,
                        update_communities=update_communities,
                        saga=saga,
                        saga_previous_episode_uuid=saga_previous_episode_uuid,
                    )
                    logger.debug(
                        f'[{task_id[:20]}...] Queued for project group: {project_config.group_id}'
                    )

                    # Queue for shared groups
                    for shared_gid in project_config.shared_group_ids:
                        await self._queue_for_group(
                            name=name,
                            episode_body=episode_body,
                            group_id=shared_gid,
                            group_uuid=route_uuid(shared_gid),
                            metadata=metadata,
                            uuid=uuid,
                            reference_time=reference_time,
                            excluded_entity_types=excluded_entity_types,
                            custom_extraction_instructions=custom_extraction_instructions,
                            previous_episode_uuids=previous_episode_uuids,
                            update_communities=update_communities,
                            saga=saga,
                            saga_previous_episode_uuid=saga_previous_episode_uuid,
                        )
                        logger.debug(f'[{task_id[:20]}...] Queued for shared group: {shared_gid}')

            logger.info(f"[{task_id[:20]}...] Memory '{name}' classification and queuing complete")

        except Exception as e:
            # Fail-safe: never drop the episode after add_memory already
            # returned success. Queue it directly to the project group so the
            # data survives; classification can be corrected later.
            error_msg = f'Error in background task for memory {name}: {e}'
            logger.error(error_msg, exc_info=True)
            try:
                await self._queue_for_group(
                    name=name,
                    episode_body=episode_body,
                    group_id=project_config.group_id,
                    group_uuid=route_uuid(project_config.group_id),
                    metadata=metadata,
                    uuid=uuid,
                    reference_time=reference_time,
                    excluded_entity_types=excluded_entity_types,
                    custom_extraction_instructions=custom_extraction_instructions,
                    previous_episode_uuids=previous_episode_uuids,
                    update_communities=update_communities,
                    saga=saga,
                    saga_previous_episode_uuid=saga_previous_episode_uuid,
                )
                logger.info(
                    f"Fail-safe: memory '{name}' queued to project group "
                    f"'{project_config.group_id}' after classification failure"
                )
            except Exception as fallback_error:
                # Dead letter: the last-resort log line carries the full
                # episode content so nothing disappears silently.
                logger.critical(
                    f'DEAD LETTER - memory lost, queueing failed twice. '
                    f'name={name!r} group={project_config.group_id!r} '
                    f'uuid={uuid!r} error={fallback_error!r} '
                    f'episode_body={episode_body!r}'
                )

    async def _queue_for_group(
        self,
        name: str,
        episode_body: str,
        group_id: str,
        metadata: dict | None,
        uuid: str | None,
        group_uuid: str | None = None,
        reference_time=None,
        excluded_entity_types: list[str] | None = None,
        custom_extraction_instructions: str | None = None,
        previous_episode_uuids: list[str] | None = None,
        update_communities: bool = False,
        saga: str | None = None,
        saga_previous_episode_uuid: str | None = None,
        edge_types: dict | None = None,
        edge_type_map: dict | None = None,
    ):
        """Queue an episode for processing to a specific group_id.

        Args:
            name: Name of the episode
            episode_body: Content to store
            group_id: Target group_id
            metadata: Optional metadata (should include 'source' and 'source_description')
            uuid: Optional UUID for the episode
        """
        # Prepare metadata
        episode_metadata = {}
        if metadata:
            episode_metadata.update(metadata)

        # Convert source string to EpisodeType enum
        source_str = episode_metadata.get('source', 'text')
        episode_type = EpisodeType.text  # Default
        if source_str:
            try:
                episode_type = EpisodeType[source_str.lower()]
            except (KeyError, AttributeError):
                # If the source doesn't match any enum value, use text as default
                logger.warning(f"Unknown source type '{source_str}', using 'text' as default")
                episode_type = EpisodeType.text

        # Submit to queue service for background processing
        await self.queue_service.add_episode(
            group_id=group_id,
            name=name,
            content=episode_body,
            source_description=episode_metadata.get('source_description', 'Smart Memory Writer'),
            episode_type=episode_type,
            entity_types=self.entity_types,
            uuid=group_uuid or uuid,
            reference_time=reference_time,
            excluded_entity_types=excluded_entity_types,
            custom_extraction_instructions=custom_extraction_instructions,
            previous_episode_uuids=previous_episode_uuids,
            update_communities=update_communities,
            saga=saga,
            saga_previous_episode_uuid=saga_previous_episode_uuid,
            edge_types=edge_types if edge_types is not None else self.edge_types,
            edge_type_map=edge_type_map if edge_type_map is not None else self.edge_type_map,
        )

    def should_use_smart_writer(self, project_config: ProjectConfig) -> bool:
        """Check if smart writer should be used for this project.

        Args:
            project_config: Project configuration

        Returns:
            True if project has shared configuration
        """
        return project_config.has_shared_config
