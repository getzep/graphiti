"""Rule-based memory classifier implementation.

This classifier uses simple rule-based matching to determine whether
a memory should be shared across projects or kept project-specific.
"""

import logging

from utils.project_config import ProjectConfig

from .base import ClassificationResult, MemoryCategory, MemoryClassifier

logger = logging.getLogger(__name__)


class RuleBasedClassifier(MemoryClassifier):
    """Rule-based memory classifier.

    Classification rules:
    1. Check if episode contains shared entity types (Preference, Procedure, etc.)
    2. Check if episode matches shared patterns (regex-like keyword matching)
    3. Return SHARED if any rule matches, PROJECT_SPECIFIC otherwise
    """

    # Default entity types that are considered "shared" across projects
    DEFAULT_SHARED_TYPES: set[str] = {
        'Preference',
        'Procedure',
        'Requirement',
    }

    def __init__(self, confidence_shared: float = 0.7, confidence_project: float = 0.6):
        """Initialize the rule-based classifier.

        Args:
            confidence_shared: Confidence when classifying as SHARED
            confidence_project: Confidence when classifying as PROJECT_SPECIFIC
        """
        self.confidence_shared = confidence_shared
        self.confidence_project = confidence_project

    async def classify(
        self, episode_body: str, project_config: ProjectConfig
    ) -> ClassificationResult:
        """Classify memory using rule-based matching.

        Args:
            episode_body: The content to classify
            project_config: Project configuration with shared settings

        Returns:
            ClassificationResult with category and reasoning
        """
        episode_lower = episode_body.lower()

        # Rule 1: Check shared entity types
        shared_types = set(project_config.shared_entity_types) or self.DEFAULT_SHARED_TYPES

        for entity_type in shared_types:
            # Check if entity type is mentioned in the episode
            if entity_type.lower() in episode_lower:
                logger.debug(f"Classified as SHARED: contains entity type '{entity_type}'")
                return ClassificationResult(
                    category=MemoryCategory.SHARED,
                    confidence=self.confidence_shared,
                    reasoning=f'Contains shared entity type: {entity_type}',
                )

        # Rule 2: Check shared patterns
        if project_config.shared_patterns:
            for pattern in project_config.shared_patterns:
                # Simple keyword matching (not full regex, for MVP)
                if pattern.lower() in episode_lower:
                    logger.debug(f"Classified as SHARED: matches pattern '{pattern}'")
                    return ClassificationResult(
                        category=MemoryCategory.SHARED,
                        confidence=self.confidence_shared,
                        reasoning=f'Matches shared pattern: {pattern}',
                    )

        # No rules matched - classify as project-specific
        logger.debug('Classified as PROJECT_SPECIFIC: no shared rules matched')
        return ClassificationResult(
            category=MemoryCategory.PROJECT_SPECIFIC,
            confidence=self.confidence_project,
            reasoning='No shared entity types or patterns detected',
        )

    def supports(self, strategy: str) -> bool:
        """Check if this classifier supports the given strategy."""
        return strategy in ['simple', 'rule_based', 'default']

    def get_confidence_threshold(self) -> float:
        """Get confidence threshold for this classifier."""
        # Lower threshold since rules are deterministic
        return 0.5
