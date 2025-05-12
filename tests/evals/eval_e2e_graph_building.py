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

import json
from datetime import datetime, timezone

import pandas as pd

from graphiti_core import Graphiti
from graphiti_core.graphiti import AddEpisodeResults
from graphiti_core.helpers import semaphore_gather
from graphiti_core.llm_client import LLMConfig, OpenAIClient
from graphiti_core.nodes import EpisodeType
from graphiti_core.prompts import prompt_library
from graphiti_core.prompts.eval import EvalAddEpisodeResults
from tests.test_graphiti_int import NEO4J_URI, NEO4j_PASSWORD, NEO4j_USER


async def build_subgraph(
    graphiti: Graphiti,
    user_id: str,
    multi_session,
    multi_session_dates,
    session_length: int,
    group_id_suffix: str,
) -> tuple[str, list[AddEpisodeResults], list[str]]:
    add_episode_results: list[AddEpisodeResults] = []
    add_episode_context: list[str] = []

    message_count = 0
    for session_idx, session in enumerate(multi_session):
        for _, msg in enumerate(session):
            if message_count >= session_length:
                continue
            message_count += 1
            date = multi_session_dates[session_idx] + ' UTC'
            date_format = '%Y/%m/%d (%a) %H:%M UTC'
            date_string = datetime.strptime(date, date_format).replace(tzinfo=timezone.utc)

            episode_body = f'{msg["role"]}: {msg["content"]}'
            results = await graphiti.add_episode(
                name='',
                episode_body=episode_body,
                reference_time=date_string,
                source=EpisodeType.message,
                source_description='',
                group_id=user_id + '_' + group_id_suffix,
            )
            for node in results.nodes:
                node.name_embedding = None
            for edge in results.edges:
                edge.fact_embedding = None

            add_episode_results.append(results)
            add_episode_context.append(msg['content'])

    return user_id, add_episode_results, add_episode_context


async def build_graph(
    group_id_suffix: str, multi_session_count: int, session_length: int, graphiti: Graphiti
) -> tuple[dict[str, list[AddEpisodeResults]], dict[str, list[str]]]:
    # Get longmemeval dataset
    lme_dataset_option = (
        'data/longmemeval_data/longmemeval_oracle.json'  # Can be _oracle, _s, or _m
    )
    lme_dataset_df = pd.read_json(lme_dataset_option)

    add_episode_results: dict[str, list[AddEpisodeResults]] = {}
    add_episode_context: dict[str, list[str]] = {}
    subgraph_results: list[tuple[str, list[AddEpisodeResults], list[str]]] = await semaphore_gather(
        *[
            build_subgraph(
                graphiti,
                user_id='lme_oracle_experiment_user_' + str(multi_session_idx),
                multi_session=lme_dataset_df['haystack_sessions'].iloc[multi_session_idx],
                multi_session_dates=lme_dataset_df['haystack_dates'].iloc[multi_session_idx],
                session_length=session_length,
                group_id_suffix=group_id_suffix,
            )
            for multi_session_idx in range(multi_session_count)
        ]
    )

    for user_id, episode_results, episode_context in subgraph_results:
        add_episode_results[user_id] = episode_results
        add_episode_context[user_id] = episode_context

    return add_episode_results, add_episode_context


async def build_baseline_graph(multi_session_count: int, session_length: int):
    # Use gpt-4.1-mini for graph building baseline
    llm_client = OpenAIClient(config=LLMConfig(model='gpt-4.1-mini'))
    graphiti = Graphiti(NEO4J_URI, NEO4j_USER, NEO4j_PASSWORD, llm_client=llm_client)

    add_episode_results, _ = await build_graph(
        'baseline', multi_session_count, session_length, graphiti
    )

    filename = 'baseline_graph_results.json'

    serializable_baseline_graph_results = {
        key: [item.model_dump(mode='json') for item in value]
        for key, value in add_episode_results.items()
    }

    with open(filename, 'w') as file:
        json.dump(serializable_baseline_graph_results, file, indent=4, default=str)


async def eval_graph(multi_session_count: int, session_length: int, llm_client=None) -> float:
    if llm_client is None:
        llm_client = OpenAIClient(config=LLMConfig(model='gpt-4.1-mini'))
    graphiti = Graphiti(NEO4J_URI, NEO4j_USER, NEO4j_PASSWORD, llm_client=llm_client)
    with open('baseline_graph_results.json') as file:
        baseline_results_raw = json.load(file)

        baseline_results: dict[str, list[AddEpisodeResults]] = {
            key: [AddEpisodeResults(**item) for item in value]
            for key, value in baseline_results_raw.items()
        }
    add_episode_results, add_episode_context = await build_graph(
        'candidate', multi_session_count, session_length, graphiti
    )

    filename = 'candidate_graph_results.json'

    candidate_baseline_graph_results = {
        key: [item.model_dump(mode='json') for item in value]
        for key, value in add_episode_results.items()
    }

    with open(filename, 'w') as file:
        json.dump(candidate_baseline_graph_results, file, indent=4, default=str)

    raw_score = 0
    user_count = 0
    for user_id in add_episode_results:
        user_count += 1
        user_raw_score = 0
        for baseline_result, add_episode_result, episodes in zip(
            baseline_results[user_id],
            add_episode_results[user_id],
            add_episode_context[user_id],
            strict=False,
        ):
            context = {
                'baseline': baseline_result,
                'candidate': add_episode_result,
                'message': episodes[0],
                'previous_messages': episodes[1:],
            }

            llm_response = await llm_client.generate_response(
                prompt_library.eval.eval_add_episode_results(context),
                response_model=EvalAddEpisodeResults,
            )

            candidate_is_worse = llm_response.get('candidate_is_worse', False)
            user_raw_score += 0 if candidate_is_worse else 1
            print('llm_response:', llm_response)
        user_score = user_raw_score / len(add_episode_results[user_id])
        raw_score += user_score
    score = raw_score / user_count

    return score
