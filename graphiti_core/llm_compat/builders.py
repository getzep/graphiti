from .block_parser import extract_item_blocks, parse_fields
from .records import (
    ParsedEdgeDedupeRecord,
    ParsedEdgeRecord,
    ParsedEntityRecord,
    ParsedNodeDedupeRecord,
    ParsedSummaryRecord,
)


def _normalize_optional(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _parse_int_list(value: str | None) -> list[int]:
    if not value:
        return []

    parsed: list[int] = []
    for part in value.split(','):
        stripped = part.strip()
        if not stripped:
            continue
        try:
            parsed.append(int(stripped))
        except ValueError:
            continue
    return parsed


def build_entity_records(text: str) -> list[ParsedEntityRecord]:
    records: list[ParsedEntityRecord] = []
    for block in extract_item_blocks(text):
        fields = parse_fields(block)
        name = _normalize_optional(fields.get('NAME'))
        if name is None:
            continue
        records.append(
            ParsedEntityRecord(
                name=name,
                entity_type=_normalize_optional(fields.get('TYPE')),
            )
        )
    return records


def build_summary_records(text: str) -> list[ParsedSummaryRecord]:
    records: list[ParsedSummaryRecord] = []
    for block in extract_item_blocks(text):
        fields = parse_fields(block)
        name = _normalize_optional(fields.get('NAME'))
        summary = _normalize_optional(fields.get('SUMMARY'))
        if name is None or summary is None:
            continue
        records.append(ParsedSummaryRecord(name=name, summary=summary))
    return records


def build_node_dedupe_records(text: str) -> list[ParsedNodeDedupeRecord]:
    records: list[ParsedNodeDedupeRecord] = []
    for block in extract_item_blocks(text):
        fields = parse_fields(block)
        idx_value = _normalize_optional(fields.get('IDX'))
        name = _normalize_optional(fields.get('NAME'))
        if idx_value is None or name is None:
            continue
        try:
            idx = int(idx_value)
        except ValueError:
            continue
        records.append(
            ParsedNodeDedupeRecord(
                idx=idx,
                name=name,
                match=_normalize_optional(fields.get('MATCH')),
            )
        )
    return records


def build_edge_records(text: str) -> list[ParsedEdgeRecord]:
    records: list[ParsedEdgeRecord] = []
    for block in extract_item_blocks(text):
        fields = parse_fields(block)
        source = _normalize_optional(fields.get('SOURCE'))
        target = _normalize_optional(fields.get('TARGET'))
        fact = _normalize_optional(fields.get('FACT'))
        if source is None or target is None or fact is None:
            continue
        records.append(
            ParsedEdgeRecord(
                source=source,
                target=target,
                relation_type=_normalize_optional(fields.get('RELATION_TYPE')),
                fact=fact,
                valid_at=_normalize_optional(fields.get('VALID_AT')),
                invalid_at=_normalize_optional(fields.get('INVALID_AT')),
            )
        )
    return records


def build_edge_dedupe_record(text: str) -> ParsedEdgeDedupeRecord:
    blocks = extract_item_blocks(text)
    if not blocks:
        return ParsedEdgeDedupeRecord(duplicate_facts=[], contradicted_facts=[])

    fields = parse_fields(blocks[0])
    return ParsedEdgeDedupeRecord(
        duplicate_facts=_parse_int_list(fields.get('DUPLICATE_FACTS')),
        contradicted_facts=_parse_int_list(fields.get('CONTRADICTED_FACTS')),
    )
