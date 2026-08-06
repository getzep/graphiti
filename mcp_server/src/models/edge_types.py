"""Edge (fact) type definitions for Graphiti MCP Server.

Edge types describe the kind of relationship a fact represents between two
entities. They are registered with graphiti-core via ``add_episode``'s
``edge_types`` argument, and constrained to specific source/target entity-type
pairs via ``edge_type_map``.

Attributes declared on an edge model are extracted by the LLM and stored on the
edge. Only use information present in the episode to populate them.
"""

from pydantic import BaseModel, Field


class RelatesTo(BaseModel):
    """A generic, untyped relationship between two entities.

    Use this only when no more specific edge type applies. Captures that two
    entities are associated without asserting the nature of the association.
    """

    ...


class MentionedIn(BaseModel):
    """An entity is referenced, described, or discussed within a source.

    Connects an entity to the document, message, or context in which it appears.
    """

    ...


class WorksFor(BaseModel):
    """An employment or membership relationship between a person and an organization."""

    role: str | None = Field(
        default=None,
        description='The role, title, or position held. Only use information present in the context.',
    )


class LocatedAt(BaseModel):
    """A spatial relationship indicating an entity is situated at or within a location."""

    ...


class ParticipatesIn(BaseModel):
    """A person or organization takes part in an event or activity."""

    ...


class Owns(BaseModel):
    """An ownership or possession relationship between an entity and an object."""

    ...


class Requires(BaseModel):
    """A dependency relationship: the source needs or depends on the target.

    Commonly connects a project or requirement to the thing it depends upon.
    """

    ...


EDGE_TYPES: dict[str, type[BaseModel]] = {
    'RelatesTo': RelatesTo,
    'MentionedIn': MentionedIn,
    'WorksFor': WorksFor,
    'LocatedAt': LocatedAt,
    'ParticipatesIn': ParticipatesIn,
    'Owns': Owns,
    'Requires': Requires,
}
