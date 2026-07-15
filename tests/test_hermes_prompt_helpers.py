from datetime import datetime, timezone
from decimal import Decimal

from graphiti_core.prompts.prompt_helpers import to_prompt_json


class Neo4jLikeDateTime:
    def iso_format(self) -> str:
        return '2026-06-06T01:02:03.000000000Z'


def test_to_prompt_json_serializes_common_non_json_values() -> None:
    payload = {
        'created_at': datetime(2026, 6, 6, 1, 2, 3, tzinfo=timezone.utc),
        'neo4j_time': Neo4jLikeDateTime(),
        'tags': {'b', 'a'},
        'amount': Decimal('1.25'),
        'blob': b'hi',
    }

    rendered = to_prompt_json(payload)

    assert '2026-06-06T01:02:03+00:00' in rendered
    assert '2026-06-06T01:02:03.000000000Z' in rendered
    assert '"tags": ["a", "b"]' in rendered
    assert '"amount": "1.25"' in rendered
    assert '"blob": "aGk="' in rendered
