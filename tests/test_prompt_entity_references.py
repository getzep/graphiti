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

import pytest

from graphiti_core.prompts.extract_nodes import extract_message as extract_nodes_message
from graphiti_core.prompts.extract_nodes_and_edges import (
    extract_message as extract_nodes_and_edges_message,
)


@pytest.mark.parametrize(
    ('build_prompt', 'unreferenced_exclusion'),
    [
        (extract_nodes_message, 'current message does not refer to them'),
        (extract_nodes_and_edges_message, 'current messages do not refer to'),
    ],
)
def test_message_prompt_extracts_pronoun_referenced_previous_entity(
    build_prompt, unreferenced_exclusion
):
    context = {
        'entity_types': [{'entity_type_id': 0, 'entity_type_name': 'Entity'}],
        'previous_episodes': [
            {'content': "User: This is my friend John. He's a software engineer."}
        ],
        'episode_content': 'User: He also has a friend named Mike.',
        'custom_extraction_instructions': '',
    }

    user_prompt = ' '.join(build_prompt(context)[1].content.lower().split())

    assert 'counts as implicitly mentioned' in user_prompt
    assert 'must be extracted' in user_prompt
    assert unreferenced_exclusion in user_prompt
