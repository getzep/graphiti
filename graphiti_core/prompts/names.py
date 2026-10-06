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

from typing import Literal

# Keep in sync with PROMPT_GROUPS / BUILTIN_PROMPT_SPECS in lib.py.
PromptGroup = Literal[
    'extract_nodes',
    'dedupe_nodes',
    'extract_edges',
    'extract_nodes_and_edges',
    'dedupe_edges',
    'summarize_nodes',
    'summarize_sagas',
    'eval',
]

PromptName = Literal[
    'extract_nodes.extract_message',
    'extract_nodes.extract_json',
    'extract_nodes.extract_text',
    'extract_nodes.classify_nodes',
    'extract_nodes.extract_attributes',
    'extract_nodes.extract_summary',
    'extract_nodes.extract_summaries_batch',
    'extract_nodes.extract_entity_summaries_from_episodes',
    'dedupe_nodes.node',
    'dedupe_nodes.node_list',
    'dedupe_nodes.nodes',
    'extract_edges.edge',
    'extract_edges.extract_attributes',
    'extract_edges.extract_timestamps',
    'extract_edges.extract_timestamps_batch',
    'extract_nodes_and_edges.extract_message',
    'dedupe_edges.resolve_edge',
    'summarize_nodes.summarize_pair',
    'summarize_nodes.summarize_context',
    'summarize_nodes.summary_description',
    'summarize_sagas.summarize_saga',
    'eval.query_expansion',
    'eval.qa_prompt',
    'eval.eval_prompt',
    'eval.eval_add_episode_results',
]
