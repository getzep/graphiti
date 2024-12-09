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

import os
from datetime import datetime

import numpy as np
from dotenv import load_dotenv
from neo4j import time as neo4j_time

load_dotenv()

DEFAULT_DATABASE = os.getenv('DEFAULT_DATABASE', None)
USE_PARALLEL_RUNTIME = bool(os.getenv('USE_PARALLEL_RUNTIME', False))
MAX_REFLEXION_ITERATIONS = 2
DEFAULT_PAGE_LIMIT = 20


def parse_db_date(neo_date: neo4j_time.DateTime | None) -> datetime | None:
    return neo_date.to_native() if neo_date else None


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


def normalize_l2(embedding: list[float]) -> list[float]:
    embedding_array = np.array(embedding)
    if embedding_array.ndim == 1:
        norm = np.linalg.norm(embedding_array)
        if norm == 0:
            return embedding_array.tolist()
        return (embedding_array / norm).tolist()
    else:
        norm = np.linalg.norm(embedding_array, 2, axis=1, keepdims=True)
        return (np.where(norm == 0, embedding_array, embedding_array / norm)).tolist()
