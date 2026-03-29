from graphiti_core.llm_compat.block_parser import extract_item_blocks, parse_fields
from graphiti_core.llm_compat.builders import (
    build_edge_dedupe_record,
    build_edge_records,
    build_entity_records,
    build_node_dedupe_records,
    build_summary_records,
)


def test_extract_item_blocks_tolerates_wrapper_noise_and_missing_end_item():
    text = """
    Here is the result:
    ```text
    BEGIN ITEMS
    BEGIN ITEM
    NAME: Alice
    TYPE: Person
    END ITEM
    BEGIN ITEM
    NAME: Bob
    TYPE:
    ```
    trailing noise
    """

    blocks = extract_item_blocks(text)

    assert len(blocks) == 2
    assert 'NAME: Alice' in blocks[0]
    assert 'NAME: Bob' in blocks[1]


def test_parse_fields_ignores_malformed_lines_and_uses_last_duplicate_value():
    fields = parse_fields(
        """
        NAME: Alice
        malformed line
        TYPE: Person
        TYPE: Employee
        EMPTY:
        """
    )

    assert fields == {'NAME': 'Alice', 'TYPE': 'Employee', 'EMPTY': ''}


def test_build_entity_records_ignores_unknown_fields_and_missing_names():
    text = """
    BEGIN ITEMS
    BEGIN ITEM
    NAME: Alice
    TYPE: Person
    EXTRA: ignored
    END ITEM
    BEGIN ITEM
    TYPE: Organization
    END ITEM
    END ITEMS
    """

    records = build_entity_records(text)

    assert len(records) == 1
    assert records[0].name == 'Alice'
    assert records[0].entity_type == 'Person'


def test_build_summary_records_require_name_and_summary():
    text = """
    BEGIN ITEMS
    BEGIN ITEM
    NAME: Alice
    SUMMARY: Leads Product.
    END ITEM
    BEGIN ITEM
    NAME: Bob
    END ITEM
    END ITEMS
    """

    records = build_summary_records(text)

    assert [(record.name, record.summary) for record in records] == [('Alice', 'Leads Product.')]


def test_build_node_dedupe_records_parse_idx_and_empty_match():
    text = """
    BEGIN ITEMS
    BEGIN ITEM
    IDX: 0
    NAME: Alice
    MATCH: Alice Smith
    END ITEM
    BEGIN ITEM
    IDX: nope
    NAME: Bob
    MATCH:
    END ITEM
    END ITEMS
    """

    records = build_node_dedupe_records(text)

    assert len(records) == 1
    assert records[0].idx == 0
    assert records[0].match == 'Alice Smith'


def test_build_edge_records_keep_bad_datetimes_as_raw_strings_for_local_handling():
    text = """
    BEGIN ITEMS
    BEGIN ITEM
    SOURCE: Alice
    TARGET: Acme
    RELATION_TYPE: WORKS_AT
    FACT: Alice works at Acme.
    VALID_AT: invalid-date
    INVALID_AT:
    END ITEM
    BEGIN ITEM
    SOURCE: Alice
    TARGET:
    FACT: bad edge
    END ITEM
    END ITEMS
    """

    records = build_edge_records(text)

    assert len(records) == 1
    assert records[0].source == 'Alice'
    assert records[0].target == 'Acme'
    assert records[0].valid_at == 'invalid-date'
    assert records[0].invalid_at is None


def test_build_edge_dedupe_record_parses_integer_lists_defensively():
    text = """
    BEGIN ITEMS
    BEGIN ITEM
    DUPLICATE_FACTS: 0, 2, nope, 4
    CONTRADICTED_FACTS: 7, bad, 9
    END ITEM
    END ITEMS
    """

    record = build_edge_dedupe_record(text)

    assert record.duplicate_facts == [0, 2, 4]
    assert record.contradicted_facts == [7, 9]
