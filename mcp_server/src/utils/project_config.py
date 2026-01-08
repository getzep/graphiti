"""Project configuration detection for Graphiti MCP server.

This module provides functionality to detect and load project-specific
Graphiti configuration from .graphiti.json files in the project directory.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ProjectConfig:
    """Project configuration loaded from .graphiti.json."""

    group_id: str
    config_path: Path
    description: str | None = None

    # Shared knowledge configuration
    shared_group_ids: list[str] = field(default_factory=list)
    shared_entity_types: list[str] = field(default_factory=list)
    shared_patterns: list[str] = field(default_factory=list)

    # Write strategy: 'simple' (default) | 'smart_split' | 'smart_duplicate'
    write_strategy: str = 'simple'

    @property
    def project_root(self) -> Path:
        """Get the project root directory."""
        return self.config_path.parent

    @property
    def has_shared_config(self) -> bool:
        """Check if this project has shared knowledge configuration."""
        return len(self.shared_group_ids) > 0


def find_project_config(start_dir: Path | None = None) -> ProjectConfig | None:
    """
    Find .graphiti.json by searching upward from start_dir (git-like approach).

    Args:
        start_dir: Directory to start search from (defaults to current working directory)

    Returns:
        ProjectConfig if found, None otherwise
    """
    if start_dir is None:
        start_dir = Path.cwd()

    current_dir = Path(start_dir).resolve()

    # Search upward until root or config found
    while current_dir != current_dir.parent:
        config_path = current_dir / '.graphiti.json'

        if config_path.exists():
            try:
                return load_project_config(config_path)
            except Exception as e:
                logger.warning(f'Failed to load {config_path}: {e}')
                return None

        current_dir = current_dir.parent

    return None


def load_project_config(config_path: Path) -> ProjectConfig | None:
    """
    Load project configuration from .graphiti.json file.

    Args:
        config_path: Path to .graphiti.json

    Returns:
        ProjectConfig object or None if loading fails
    """
    try:
        with open(config_path, encoding='utf-8') as f:
            data = json.load(f)

        group_id = data.get('group_id')
        if not group_id:
            logger.error(f"{config_path} missing required 'group_id' field")
            return None

        if not isinstance(group_id, str):
            logger.error(
                f"{config_path} 'group_id' must be a string, got {type(group_id).__name__}"
            )
            return None

        description = data.get('description')
        if description is not None and not isinstance(description, str):
            logger.warning(
                f"{config_path} 'description' must be a string, got {type(description).__name__}"
            )
            description = None

        # Load shared knowledge configuration
        shared_group_ids = data.get('shared_group_ids', [])
        if not isinstance(shared_group_ids, list):
            logger.warning(f"{config_path} 'shared_group_ids' must be a list")
            shared_group_ids = []

        shared_entity_types = data.get('shared_entity_types', [])
        if not isinstance(shared_entity_types, list):
            logger.warning(f"{config_path} 'shared_entity_types' must be a list")
            shared_entity_types = []

        shared_patterns = data.get('shared_patterns', [])
        if not isinstance(shared_patterns, list):
            logger.warning(f"{config_path} 'shared_patterns' must be a list")
            shared_patterns = []

        write_strategy = data.get('write_strategy', 'simple')
        if not isinstance(write_strategy, str):
            logger.warning(f"{config_path} 'write_strategy' must be a string")
            write_strategy = 'simple'

        logger.info(
            f'Loaded project config from {config_path}: '
            f'group_id={group_id}, '
            f'shared_groups={len(shared_group_ids)}, '
            f'strategy={write_strategy}'
        )

        return ProjectConfig(
            group_id=group_id,
            config_path=config_path,
            description=description,
            shared_group_ids=shared_group_ids,
            shared_entity_types=shared_entity_types,
            shared_patterns=shared_patterns,
            write_strategy=write_strategy,
        )

    except json.JSONDecodeError as e:
        logger.error(f'Invalid JSON in {config_path}: {e}')
        return None
    except Exception as e:
        logger.error(f'Error loading {config_path}: {e}')
        return None
