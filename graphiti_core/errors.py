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


class EdgesNotFoundError(GraphitiError):
    """Raised when a list of edges is not found."""

    def __init__(self, uuids: list[str]):
        self.message = f'None of the edges for {uuids} were found.'
        super().__init__(self.message)


class GroupsEdgesNotFoundError(GraphitiError):
    """Raised when no edges are found for a list of group ids."""

    def __init__(self, group_ids: list[str]):
        self.message = f'no edges found for group ids {group_ids}'
        super().__init__(self.message)


class GroupsNodesNotFoundError(GraphitiError):
    """Raised when no nodes are found for a list of group ids."""

    def __init__(self, group_ids: list[str]):
        self.message = f'no nodes found for group ids {group_ids}'
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


class EntityTypeValidationError(GraphitiError):
    """Raised when an entity type uses protected attribute names."""

    def __init__(self, entity_type: str, entity_type_attribute: str):
        self.message = f'{entity_type_attribute} cannot be used as an attribute for {entity_type} as it is a protected attribute name.'
        super().__init__(self.message)


class IdentityPropertyValidationError(GraphitiError):
    """Raised when identity properties do not match the configured entity types."""

    def __init__(self, entity_type: str, reason: str):
        self.message = f'Invalid identity properties for {entity_type}: {reason}'
        super().__init__(self.message)


class GroupIdValidationError(GraphitiError):
    """Raised when a group_id contains invalid characters."""

    def __init__(self, group_id: str):
        self.message = f'group_id "{group_id}" must contain only alphanumeric characters, dashes, or underscores'
        super().__init__(self.message)


class NodeLabelValidationError(GraphitiError, ValueError):
    """Raised when a node label contains invalid characters."""

    def __init__(self, node_labels: list[str]):
        label_list = ', '.join(f'"{label}"' for label in node_labels)
        self.message = (
            'node_labels must start with a letter or underscore and contain only '
            f'alphanumeric characters or underscores: {label_list}'
        )
        super().__init__(self.message)


class UnitExpiredError(GraphitiError):
    """Raised when an ingestion unit calls the state store after its deadline or after it failed."""

    def __init__(self, group_id: str, seq: int):
        self.message = f'ingestion unit {seq} of group {group_id} is expired or failed'
        super().__init__(self.message)


class UnitInProgressError(GraphitiError):
    """Raised when an admission names the `unit_id` of an ingestion unit that is in progress.

    The admission writes nothing. The host can try again after the lease time of the unit.
    """

    def __init__(self, group_id: str, unit_id: str, seq: int):
        self.group_id = group_id
        self.unit_id = unit_id
        self.seq = seq
        self.message = (
            f'ingestion unit {seq} of group {group_id} with the unit_id {unit_id} is in progress'
        )
        super().__init__(self.message)


class StaleCommitError(GraphitiError):
    """Raised when a commit attempt uses a fence that is not the current fence of the unit."""

    def __init__(self, group_id: str, seq: int):
        self.message = f'commit attempt for ingestion unit {seq} of group {group_id} is stale'
        super().__init__(self.message)


class ChangeSetRejected(GraphitiError):
    """Raised by a host hook to reject the change set of an ingestion unit before the write."""

    def __init__(self, reason: str = 'change set rejected'):
        self.message = reason
        super().__init__(self.message)
