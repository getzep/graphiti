from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel

SCHEMA_ID = 'agent_memory_v1'


class UserEntity(BaseModel):
    """A stable identifier for the human user interacting with an agent.

    Prefer non-PII identifiers (hashed keys or provider ids) such as `github:12345` or `user:hash`.
    """


class WorkspaceEntity(BaseModel):
    """A stable identifier for the current project/workspace (repo + working directory context).

    Examples: `repo:owner/name`, `workspace:/abs/path`.
    """


class SessionEntity(BaseModel):
    """A stable identifier for a single interactive session/run.

    Examples: `session:<uuid>`, `chat:<uuid>`.
    """


ENTITY_TYPES: Mapping[str, type[BaseModel]] = {
    'User': UserEntity,
    'Workspace': WorkspaceEntity,
    'Session': SessionEntity,
}


class OwnsRelation(BaseModel):
    """Use for explicit ownership relations.

    Examples:
    - `User owns Workspace`
    - `User owns Session`
    - `Workspace owns Asset` (repo, folder, file, branch)
    """


class PrefersRelation(BaseModel):
    """Use for stable preferences and defaults that should influence future agent behavior.

    Examples:
    - Output preferences (concise, bullet-first, include diffs, etc.)
    - Tooling preferences (`rg` over `grep`, `just` over `make`, etc.)
    - Workflow preferences (spec-first, small commits, run tests before push)
    """


class MeansRelation(BaseModel):
    """Use for workspace-specific terminology mapping.

    Examples:
    - `'playbook' means 'runbook docs'`
    - `'MCP' means 'Model Context Protocol'` (if project-specific)
    """


class WorkingOnRelation(BaseModel):
    """Use for active goals, tasks, or planned work items.

    Examples:
    - `Workspace is working on 'graphiti memory integration'`
    - `User is working on 'fix flaky tests'`
    """


class DecidedRelation(BaseModel):
    """Use for decisions/policies agreed for a repo or project.

    Examples:
    - `Workspace decided to keep changes behind feature flags`
    - `Workspace decided to use Neo4j for memory backend`
    """


class LearnedRelation(BaseModel):
    """Use for generalized lessons learned or durable technical insights.

    Examples:
    - `User learned that /healthcheck is the Graphiti health endpoint`
    - `Workspace learned that background jobs must not capture closed resources`
    """


class BlockedByRelation(BaseModel):
    """Use for explicit dependency/blocker relationships between tasks or work items."""


EDGE_TYPES: Mapping[str, type[BaseModel]] = {
    'OWNS': OwnsRelation,
    'PREFERS': PrefersRelation,
    'MEANS': MeansRelation,
    'WORKING_ON': WorkingOnRelation,
    'DECIDED': DecidedRelation,
    'LEARNED': LearnedRelation,
    'BLOCKED_BY': BlockedByRelation,
}


EDGE_TYPE_MAP: Mapping[tuple[str, str], list[str]] = {
    ('Entity', 'Entity'): list(EDGE_TYPES.keys()),
}
