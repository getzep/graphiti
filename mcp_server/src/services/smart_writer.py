"""Smart memory writer for multi-group knowledge storage.

This module provides the SmartMemoryWriter which coordinates
memory classification and writes to appropriate group_ids.
"""

import logging
from typing import Optional

from classifiers.base import MemoryClassifier, MemoryCategory, ClassificationResult
from utils.project_config import ProjectConfig

logger = logging.getLogger(__name__)


class WriteResult:
    """Result of a smart memory write operation.

    Attributes:
        success: Whether the write operation succeeded
        written_to: List of group_ids that were written to
        category: The memory category that was determined
        classification: The full classification result
        error: Error message if success is False
    """

    def __init__(
        self,
        success: bool,
        written_to: list[str],
        category: str,
        classification: Optional[ClassificationResult] = None,
        error: Optional[str] = None
    ):
        self.success = success
        self.written_to = written_to
        self.category = category
        self.classification = classification
        self.error = error

    def __repr__(self) -> str:
        if self.success:
            return f"WriteResult(success=True, written_to={self.written_to}, category={self.category})"
        else:
            return f"WriteResult(success=False, error={self.error})"


class SmartMemoryWriter:
    """Coordinates memory classification and multi-group writes.

    This class uses a MemoryClassifier to determine where memories should
    be stored, then writes them to the appropriate group_ids via the
    Graphiti client.
    """

    def __init__(
        self,
        classifier: MemoryClassifier,
        graphiti_client
    ):
        """Initialize the smart memory writer.

        Args:
            classifier: The memory classifier to use
            graphiti_client: Graphiti client for storage operations
        """
        self.classifier = classifier
        self.graphiti_client = graphiti_client
        logger.info(f"SmartMemoryWriter initialized with {type(classifier).__name__}")

    async def add_memory(
        self,
        name: str,
        episode_body: str,
        project_config: ProjectConfig,
        metadata: Optional[dict] = None
    ) -> WriteResult:
        """Intelligently write memory to appropriate groups.

        Args:
            name: Name of the memory episode
            episode_body: Content of the memory
            project_config: Project configuration with group settings
            metadata: Optional metadata (e.g., timestamp, source)

        Returns:
            WriteResult with details of where the memory was written
        """
        try:
            # Step 1: Classify the memory
            logger.debug(f"Classifying memory: {name}")
            classification = await self.classifier.classify(
                episode_body=episode_body,
                project_config=project_config
            )

            logger.debug(
                f"Classification result: {classification.category.value}, "
                f"confidence={classification.confidence}, "
                f"reasoning={classification.reasoning}"
            )

            written_groups = []

            # Step 2: Route based on classification
            if classification.category == MemoryCategory.SHARED:
                # Write to all shared groups
                for shared_gid in project_config.shared_group_ids:
                    await self._write_to_group(
                        name=name,
                        episode_body=episode_body,
                        group_id=shared_gid,
                        metadata=metadata
                    )
                    written_groups.append(shared_gid)
                    logger.debug(f"Written to shared group: {shared_gid}")

            elif classification.category == MemoryCategory.PROJECT_SPECIFIC:
                # Write to project group only
                await self._write_to_group(
                    name=name,
                    episode_body=episode_body,
                    group_id=project_config.group_id,
                    metadata=metadata
                )
                written_groups.append(project_config.group_id)
                logger.debug(f"Written to project group: {project_config.group_id}")

            elif classification.category == MemoryCategory.MIXED:
                # Write to both project and shared groups
                # For MVP: write full content to both
                # Future: could split content based on classification.shared_part/project_part

                # Write to project group
                await self._write_to_group(
                    name=name,
                    episode_body=episode_body,
                    group_id=project_config.group_id,
                    metadata=metadata
                )
                written_groups.append(project_config.group_id)
                logger.debug(f"Written to project group: {project_config.group_id}")

                # Write to shared groups
                for shared_gid in project_config.shared_group_ids:
                    await self._write_to_group(
                        name=name,
                        episode_body=episode_body,
                        group_id=shared_gid,
                        metadata=metadata
                    )
                    written_groups.append(shared_gid)
                    logger.debug(f"Written to shared group: {shared_gid}")

            logger.info(
                f"Memory '{name}' written to {len(written_groups)} group(s): {written_groups}"
            )

            return WriteResult(
                success=True,
                written_to=written_groups,
                category=classification.category.value,
                classification=classification
            )

        except Exception as e:
            error_msg = f"Error writing memory: {e}"
            logger.error(error_msg, exc_info=True)
            return WriteResult(
                success=False,
                written_to=[],
                category="unknown",
                error=error_msg
            )

    async def _write_to_group(
        self,
        name: str,
        episode_body: str,
        group_id: str,
        metadata: Optional[dict] = None
    ):
        """Write to a specific group_id using Graphiti Core API.

        Args:
            name: Name of the episode
            episode_body: Content to store
            group_id: Target group_id
            metadata: Optional metadata
        """
        # Prepare metadata
        episode_metadata = {}
        if metadata:
            episode_metadata.update(metadata)

        # Call Graphiti Core API
        # The graphiti_client should have add_episode method
        await self.graphiti_client.add_episode(
            name=name,
            episode_body=episode_body,
            group_id=group_id,
            reference_time=episode_metadata.get('timestamp')
        )

    def should_use_smart_writer(self, project_config: ProjectConfig) -> bool:
        """Check if smart writer should be used for this project.

        Args:
            project_config: Project configuration

        Returns:
            True if project has shared configuration
        """
        return project_config.has_shared_config
