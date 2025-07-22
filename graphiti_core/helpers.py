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

import asyncio
import os
import re
from collections.abc import Coroutine
from datetime import datetime
from typing import Any

import numpy as np
from dotenv import load_dotenv
from neo4j import time as neo4j_time
from numpy._typing import NDArray
from pydantic import BaseModel
from typing_extensions import LiteralString

from graphiti_core.errors import GroupIdValidationError

load_dotenv()

USE_PARALLEL_RUNTIME = bool(os.getenv('USE_PARALLEL_RUNTIME', False))
SEMAPHORE_LIMIT = int(os.getenv('SEMAPHORE_LIMIT', 20))
MAX_REFLEXION_ITERATIONS = int(os.getenv('MAX_REFLEXION_ITERATIONS', 0))
DEFAULT_PAGE_LIMIT = 20

RUNTIME_QUERY: LiteralString = (
    'CYPHER runtime = parallel parallelRuntimeSupport=all\n' if USE_PARALLEL_RUNTIME else ''
)


def parse_db_date(neo_date: neo4j_time.DateTime | str | None) -> datetime | None:
    return (
        neo_date.to_native()
        if isinstance(neo_date, neo4j_time.DateTime)
        else datetime.fromisoformat(neo_date)
        if neo_date
        else None
    )


def get_default_group_id(db_type: str) -> str:
    """
    This function differentiates the default group id based on the database type.
    For most databases, the default group id is an empty string, while there are database types that require a specific default group id.
    """
    if db_type == 'falkordb':
        return '_'
    else:
        return ''


def lucene_sanitize(query: str) -> str:
    # Escape special characters from a query before passing into Lucene
    # + - && || ! ( ) { } [ ] ^ " ~ * ? : \ /
    escape_map = str.maketrans(
        {
            '+': r'\+',
            '-': r'\-',
            '&': r'\&',
            '|': r'\|',
            '!': r'\!',
            '(': r'\(',
            ')': r'\)',
            '{': r'\{',
            '}': r'\}',
            '[': r'\[',
            ']': r'\]',
            '^': r'\^',
            '"': r'\"',
            '~': r'\~',
            '*': r'\*',
            '?': r'\?',
            ':': r'\:',
            '\\': r'\\',
            '/': r'\/',
            'O': r'\O',
            'R': r'\R',
            'N': r'\N',
            'T': r'\T',
            'A': r'\A',
            'D': r'\D',
        }
    )

    sanitized = query.translate(escape_map)
    return sanitized


def normalize_l2(embedding: list[float]) -> NDArray:
    embedding_array = np.array(embedding)
    norm = np.linalg.norm(embedding_array, 2, axis=0, keepdims=True)
    return np.where(norm == 0, embedding_array, embedding_array / norm)


# Use this instead of asyncio.gather() to bound coroutines
async def semaphore_gather(
    *coroutines: Coroutine,
    max_coroutines: int | None = None,
) -> list[Any]:
    semaphore = asyncio.Semaphore(max_coroutines or SEMAPHORE_LIMIT)

    async def _wrap_coroutine(coroutine):
        async with semaphore:
            return await coroutine

    return await asyncio.gather(*(_wrap_coroutine(coroutine) for coroutine in coroutines))


def validate_group_id(group_id: str) -> bool:
    """
    Validate that a group_id contains only ASCII alphanumeric characters, dashes, and underscores.

    Args:
        group_id: The group_id to validate

    Returns:
        True if valid, False otherwise

    Raises:
        GroupIdValidationError: If group_id contains invalid characters
    """

    # Allow empty string (default case)
    if not group_id:
        return True

    # Check if string contains only ASCII alphanumeric characters, dashes, or underscores
    # Pattern matches: letters (a-z, A-Z), digits (0-9), hyphens (-), and underscores (_)
    if not re.match(r'^[a-zA-Z0-9_-]+$', group_id):
        raise GroupIdValidationError(group_id)

    return True


def validate_excluded_entity_types(
    excluded_entity_types: list[str] | None, entity_types: dict[str, BaseModel] | None = None
) -> bool:
    """
    Validate that excluded entity types are valid type names.

    Args:
        excluded_entity_types: List of entity type names to exclude
        entity_types: Dictionary of available custom entity types

    Returns:
        True if valid

    Raises:
        ValueError: If any excluded type names are invalid
    """
    if not excluded_entity_types:
        return True

    # Build set of available type names
    available_types = {'Entity'}  # Default type is always available
    if entity_types:
        available_types.update(entity_types.keys())

    # Check for invalid type names
    invalid_types = set(excluded_entity_types) - available_types
    if invalid_types:
        raise ValueError(
            f'Invalid excluded entity types: {sorted(invalid_types)}. Available types: {sorted(available_types)}'
        )

    return True
