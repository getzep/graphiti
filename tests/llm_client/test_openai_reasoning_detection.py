import pytest

from graphiti_core.llm_client.openai_client import _is_reasoning_model


@pytest.mark.parametrize(
    'model,expected',
    [
        # reasoning models -> param should be sent
        ('gpt-5', True),
        ('gpt-5-mini', True),
        ('gpt-5.4', True),
        ('o1', True),
        ('o3', True),
        ('o3-mini', True),
        # chat / search variants -> param must NOT be sent (issue #902)
        ('gpt-5-chat-latest', False),
        ('gpt-5.3-chat-latest', False),
        ('gpt-5-search-api', False),
        # non-reasoning models
        ('gpt-4o', False),
        ('gpt-4.1', False),
        ('gpt-4o-mini', False),
    ],
)
def test_is_reasoning_model(model: str, expected: bool) -> None:
    assert _is_reasoning_model(model) is expected
