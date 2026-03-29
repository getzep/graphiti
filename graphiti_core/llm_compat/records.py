from dataclasses import dataclass


@dataclass(slots=True)
class ParsedEntityRecord:
    name: str
    entity_type: str | None


@dataclass(slots=True)
class ParsedSummaryRecord:
    name: str
    summary: str


@dataclass(slots=True)
class ParsedNodeDedupeRecord:
    idx: int
    name: str
    match: str | None


@dataclass(slots=True)
class ParsedEdgeRecord:
    source: str
    target: str
    relation_type: str | None
    fact: str
    valid_at: str | None
    invalid_at: str | None


@dataclass(slots=True)
class ParsedEdgeDedupeRecord:
    duplicate_facts: list[int]
    contradicted_facts: list[int]
