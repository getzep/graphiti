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

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from typing import Any

UNKNOWN_PROMPT_VERSION = 'unknown'
_TEMPLATE_PLACEHOLDER = re.compile(r'\{([a-z_][a-z0-9_]*)\}')


def render_template(template: str, **values: str) -> str:
    return _TEMPLATE_PLACEHOLDER.sub(
        lambda match: str(values[match.group(1)]) if match.group(1) in values else match.group(0),
        template,
    )


@dataclass(frozen=True)
class PromptTemplate:
    """Static text sections of one prompt template, in render order."""

    parts: tuple[str, ...]

    def version(self, shared_parts: tuple[str, ...] = ()) -> str:
        return _version(self.parts + shared_parts)


@cache
def _version(parts: tuple[str, ...]) -> str:
    digest = hashlib.sha256('\n'.join(parts).encode('utf-8')).hexdigest()[:12]
    return f'sha256:{digest}'


def prompt_template(*parts: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    template = PromptTemplate(parts)

    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        fn.__prompt_template__ = template
        return fn

    return decorate


def template_of(builder: Any, *, append_parts: tuple[str, ...] = ()) -> PromptTemplate | None:
    seen: set[int] = set()
    current = builder
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        template = getattr(current, '__prompt_template__', None)
        if isinstance(template, PromptTemplate):
            return PromptTemplate(template.parts + append_parts) if append_parts else template
        current = getattr(current, '__wrapped__', None)
    return None
