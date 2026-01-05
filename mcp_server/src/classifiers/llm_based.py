"""LLM-based memory classifier implementation.

This classifier uses a language model to determine whether
a memory should be shared across projects or kept project-specific.
"""

import json
import logging
from typing import Optional

from .base import MemoryClassifier, MemoryCategory, ClassificationResult
from utils.project_config import ProjectConfig

logger = logging.getLogger(__name__)


# Classification prompt template
CLASSIFICATION_PROMPT = """You are a memory classification assistant. Your task is to determine whether a given memory should be stored in shared knowledge spaces (accessible across multiple projects) or kept project-specific.

**Shared Knowledge** includes:
- User preferences (coding style, tool preferences, workflows)
- Team procedures and conventions (coding standards, Git workflows)
- General requirements and standards
- Domain knowledge that applies across projects

**Project-Specific Knowledge** includes:
- API endpoints, routes, and URLs specific to this project
- Implementation details (frameworks, libraries used)
- Project-specific files and configurations
- Architecture details unique to this project
- Database schemas specific to this project

**Instructions:**
1. Analyze the memory content
2. Classify it as one of: SHARED, PROJECT_SPECIFIC, or MIXED
3. Provide a confidence score (0.0 to 1.0)
4. Explain your reasoning briefly

**Memory to Classify:**
{episode_body}

**Project Context:**
- Shared Entity Types: {shared_entity_types}
- Shared Patterns: {shared_patterns}

**Response Format (JSON):**
{{
  "category": "SHARED|PROJECT_SPECIFIC|MIXED",
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation"
}}
"""


# Content splitting prompt for MIXED category
SPLIT_PROMPT = """You are a content splitting assistant. Your task is to separate a memory into two parts:
1. **Shared Part**: Knowledge that would be useful across multiple projects (preferences, conventions, procedures)
2. **Project Part**: Knowledge specific to this project (APIs, implementation details, project files)

**Original Memory:**
{episode_body}

**Instructions:**
- Extract the shared knowledge into a concise summary
- Extract the project-specific knowledge into a concise summary
- If one part is empty, use an empty string for that part

**Response Format (JSON):**
{{
  "shared_part": "Summary of shared knowledge",
  "project_part": "Summary of project-specific knowledge"
}}
"""


class LLMClassifier(MemoryClassifier):
    """LLM-based memory classifier.

    Uses a language model to classify memories with high accuracy.
    Can also split mixed content into shared and project-specific parts.
    """

    def __init__(self, llm_client, confidence_threshold: float = 0.6):
        """Initialize the LLM-based classifier.

        Args:
            llm_client: LLM client with chat completion support
            confidence_threshold: Minimum confidence for classification (default: 0.6)
        """
        self.llm_client = llm_client
        self.confidence_threshold = confidence_threshold
        logger.info(f"LLMClassifier initialized with confidence_threshold={confidence_threshold}")

    async def classify(
        self,
        episode_body: str,
        project_config: ProjectConfig
    ) -> ClassificationResult:
        """Classify memory using LLM analysis.

        Args:
            episode_body: The content to classify
            project_config: Project configuration with shared settings

        Returns:
            ClassificationResult with category, confidence, and reasoning
        """
        # Build prompt with context
        shared_types_str = ", ".join(project_config.shared_entity_types) if project_config.shared_entity_types else "None"
        shared_patterns_str = ", ".join(project_config.shared_patterns) if project_config.shared_patterns else "None"

        prompt = CLASSIFICATION_PROMPT.format(
            episode_body=episode_body,
            shared_entity_types=shared_types_str,
            shared_patterns=shared_patterns_str
        )

        try:
            # Call LLM with structured output
            response = await self._call_llm(prompt)

            # Parse response
            result = self._parse_classification_response(response)

            # If MIXED and confidence is high enough, split the content
            if result.category == MemoryCategory.MIXED and result.confidence >= self.confidence_threshold:
                shared_part, project_part = await self._split_content(episode_body)
                result.shared_part = shared_part
                result.project_part = project_part
                logger.debug(f"Split MIXED content: shared={len(shared_part)} chars, project={len(project_part)} chars")

            logger.debug(
                f"LLM classification: {result.category.value}, "
                f"confidence={result.confidence}, "
                f"reasoning={result.reasoning}"
            )

            return result

        except Exception as e:
            logger.error(f"LLM classification failed: {e}, falling back to PROJECT_SPECIFIC")
            # Fallback to project-specific on error
            return ClassificationResult(
                category=MemoryCategory.PROJECT_SPECIFIC,
                confidence=0.5,
                reasoning=f"LLM classification failed: {str(e)}"
            )

    async def _split_content(self, episode_body: str) -> tuple[str, str]:
        """Split content into shared and project-specific parts.

        Args:
            episode_body: The content to split

        Returns:
            Tuple of (shared_part, project_part)
        """
        prompt = SPLIT_PROMPT.format(episode_body=episode_body)

        try:
            response = await self._call_llm(prompt)
            return self._parse_split_response(response)
        except Exception as e:
            logger.error(f"Content splitting failed: {e}")
            # Fallback: put everything in project part
            return "", episode_body

    async def _call_llm(self, prompt: str) -> str:
        """Call LLM with prompt and get response.

        Args:
            prompt: The prompt to send

        Returns:
            LLM response text
        """
        # Try different LLM client interfaces
        if hasattr(self.llm_client, 'generate_structured_output'):
            # Use structured output if available
            result = await self.llm_client.generate_structured_output(
                prompt=prompt,
                response_format={
                    "type": "json_object"
                }
            )
            return result
        elif hasattr(self.llm_client, 'chat'):
            # Use chat completion
            messages = [
                {"role": "system", "content": "You are a helpful assistant that responds only in valid JSON format."},
                {"role": "user", "content": prompt}
            ]

            response = await self.llm_client.chat(
                messages=messages
            )
            return response
        elif hasattr(self.llm_client, 'generate_messages'):
            # Use generate_messages (Anthropic-style)
            messages = [
                {"role": "user", "content": prompt}
            ]

            response = await self.llm_client.generate_messages(
                messages=messages
            )
            return response
        else:
            raise ValueError(f"Unsupported LLM client type: {type(self.llm_client)}")

    def _parse_classification_response(self, response: str) -> ClassificationResult:
        """Parse LLM classification response.

        Args:
            response: JSON response from LLM

        Returns:
            ClassificationResult
        """
        try:
            data = json.loads(response)

            category_str = data.get("category", "PROJECT_SPECIFIC").upper()
            # Map string to enum
            if category_str == "SHARED":
                category = MemoryCategory.SHARED
            elif category_str == "PROJECT_SPECIFIC":
                category = MemoryCategory.PROJECT_SPECIFIC
            elif category_str == "MIXED":
                category = MemoryCategory.MIXED
            else:
                logger.warning(f"Unknown category '{category_str}', defaulting to PROJECT_SPECIFIC")
                category = MemoryCategory.PROJECT_SPECIFIC

            confidence = float(data.get("confidence", 0.5))
            # Clamp confidence to valid range
            confidence = max(0.0, min(1.0, confidence))

            reasoning = data.get("reasoning", "")

            return ClassificationResult(
                category=category,
                confidence=confidence,
                reasoning=reasoning
            )

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to parse classification result: {e}")
            raise

    def _parse_split_response(self, response: str) -> tuple[str, str]:
        """Parse content split response.

        Args:
            response: JSON response from LLM

        Returns:
            Tuple of (shared_part, project_part)
        """
        try:
            data = json.loads(response)
            shared_part = data.get("shared_part", "")
            project_part = data.get("project_part", "")
            return shared_part, project_part
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse split response as JSON: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to parse split result: {e}")
            raise

    def supports(self, strategy: str) -> bool:
        """Check if this classifier supports the given strategy.

        Args:
            strategy: Strategy name

        Returns:
            True if strategy is 'llm_based' or 'llm'
        """
        return strategy in ['llm_based', 'llm', 'smart']

    def get_confidence_threshold(self) -> float:
        """Get confidence threshold for this classifier.

        Returns:
            Confidence threshold (0.0 to 1.0)
        """
        return self.confidence_threshold
