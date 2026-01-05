"""Base classes and interfaces for memory classifiers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class MemoryCategory(Enum):
    """Memory classification categories.

    Determines where a memory should be stored:
    - PROJECT_SPECIFIC: Only in the project's group_id
    - SHARED: In shared group_ids (accessible by all projects)
    - MIXED: Both project and shared (with content splitting)
    """

    PROJECT_SPECIFIC = "project_specific"
    SHARED = "shared"
    MIXED = "mixed"


@dataclass
class ClassificationResult:
    """Result of memory classification.

    Attributes:
        category: The determined category
        confidence: Confidence score (0.0 to 1.0)
        reasoning: Human-readable explanation
        shared_part: Content suitable for shared spaces (for MIXED category)
        project_part: Content specific to project (for MIXED category)
    """

    category: MemoryCategory
    confidence: float
    reasoning: str = ""
    shared_part: str = ""
    project_part: str = ""

    def __post_init__(self):
        """Validate confidence score."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")

    @property
    def is_shared(self) -> bool:
        """Check if this memory should go to shared spaces."""
        return self.category in (MemoryCategory.SHARED, MemoryCategory.MIXED)

    @property
    def is_project_specific(self) -> bool:
        """Check if this memory should go to project space."""
        return self.category in (MemoryCategory.PROJECT_SPECIFIC, MemoryCategory.MIXED)


class MemoryClassifier(ABC):
    """Abstract base class for memory classifiers.

    Implementations determine whether a memory should be stored in
    project-specific or shared knowledge spaces based on configuration
    and content analysis.
    """

    @abstractmethod
    async def classify(
        self,
        episode_body: str,
        project_config: 'ProjectConfig'  # type: ignore
    ) -> ClassificationResult:
        """Classify a memory episode.

        Args:
            episode_body: The content to classify
            project_config: Project configuration including shared settings

        Returns:
            ClassificationResult with category and reasoning
        """
        pass

    @abstractmethod
    def supports(self, strategy: str) -> bool:
        """Check if this classifier supports the given strategy.

        Args:
            strategy: Strategy name (e.g., 'simple', 'rule_based', 'llm_based')

        Returns:
            True if this classifier supports the strategy
        """
        pass

    def get_confidence_threshold(self) -> float:
        """Get minimum confidence threshold for classification.

        Can be overridden by implementations.

        Returns:
            Minimum confidence (0.0 to 1.0)
        """
        return 0.5

    def is_confident(self, result: ClassificationResult) -> bool:
        """Check if classification result meets confidence threshold.

        Args:
            result: Classification result to check

        Returns:
            True if confidence meets threshold
        """
        return result.confidence >= self.get_confidence_threshold()
