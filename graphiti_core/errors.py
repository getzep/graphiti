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


class GraphitiError(Exception):
    """Base exception class for Graphiti Core."""


class EdgeNotFoundError(GraphitiError):
    """Raised when an edge is not found."""

    def __init__(self, uuid: str):
        self.message = f'edge {uuid} not found'
        super().__init__(self.message)


class NodeNotFoundError(GraphitiError):
    """Raised when a node is not found."""

    def __init__(self, uuid: str):
        self.message = f'node {uuid} not found'
        super().__init__(self.message)


class SearchRerankerError(GraphitiError):
    """Raised when a node is not found."""

    def __init__(self, text: str):
        self.message = text
        super().__init__(self.message)
