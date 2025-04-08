from datetime import datetime, timezone
from typing import Tuple

import pandas as pd

from graphiti_core import Graphiti
from graphiti_core.graphiti import AddEpisodeResults
from graphiti_core.llm_client import LLMConfig, OpenAIClient
from graphiti_core.nodes import EpisodeType
from graphiti_core.utils.maintenance import clear_data
from tests.test_graphiti_int import NEO4J_URI, NEO4j_PASSWORD, NEO4j_USER


async def build_graph(
    multi_session: list[int], session_length: int, graphiti: Graphiti
) -> Tuple[dict[str, list[AddEpisodeResults]], dict[str, list[str]]]:
    # Get longmemeval dataset
    lme_dataset_option = 'data/longmemeval_oracle.json'  # Can be _oracle, _s, or _m
    lme_dataset_df = pd.read_json(lme_dataset_option)

    add_episode_results: dict[str, list[AddEpisodeResults]] = {}
    add_episode_context: dict[str, list[str]] = {}
    for multi_session_idx in multi_session:
        multi_session = lme_dataset_df['haystack_sessions'].iloc[multi_session_idx]
        multi_session_dates = lme_dataset_df['haystack_dates'].iloc[multi_session_idx]

        user_id = 'lme_oracle_experiment_user_' + str(multi_session_idx)
        await clear_data(graphiti.driver, [user_id])

        add_episode_results[user_id] = []
        add_episode_context[user_id] = []

        for session_idx, session in enumerate(multi_session):
            if session_idx >= session_length:
                continue
            for msx_idx, msg in enumerate(session):
                date = multi_session_dates[session_idx] + ' UTC'
                date_format = '%Y/%m/%d (%a) %H:%M UTC'
                date_string = datetime.strptime(date, date_format).replace(tzinfo=timezone.utc)

                episode_body = f"{msg["role"]}: {msg["content"]}"
                results = await graphiti.add_episode(
                    name=msg['name'],
                    episode_body=episode_body,
                    reference_time=date_string,
                    source=EpisodeType.message,
                    source_description='',
                    group_id=user_id,
                )

                add_episode_results[user_id].append(results)
    return add_episode_results, add_episode_context


async def build_baseline_graph(multi_session: list[int], session_length: int):
    # Use gpt-4o for graph building baseline
    llm_client = OpenAIClient(config=LLMConfig(model='gpt-4o'))
    graphiti = Graphiti(NEO4J_URI, NEO4j_USER, NEO4j_PASSWORD, llm_client=llm_client)

    add_episode_results, _ = await build_graph(multi_session, session_length, graphiti)


async def eval_graph(multi_session: list[int], session_length: int, llm_client=OpenAIClient()):
    graphiti = Graphiti(NEO4J_URI, NEO4j_USER, NEO4j_PASSWORD, llm_client=llm_client)
    baseline_results: dict[str, list[AddEpisodeResults]] = {}
    add_episode_results, add_episode_context = await build_graph(
        multi_session, session_length, graphiti
    )

    for user_id in add_episode_results:
        for baseline_result, add_episode_result, episodes in zip(
            baseline_results[user_id], add_episode_results[user_id], add_episode_context[user_id]
        ):
            context = {
                'baseline': baseline_result,
                'candidate': add_episode_result,
                'message': episodes[0],
                'previous_messages': episodes[1:],
            }
