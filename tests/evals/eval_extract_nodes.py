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

import os
import json
from tests.evals.utils import setup_logging, ingest_snippet
from datetime import datetime, timedelta

import pytest
from dotenv import load_dotenv

from graphiti_core.edges import EntityEdge, EpisodicEdge
from graphiti_core.graphiti import Graphiti
from graphiti_core.nodes import EntityNode, EpisodicNode

from graphiti_core.utils.maintenance.node_operations import extract_nodes
from graphiti_core.llm_client import OpenAIClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.nodes import EpisodeType

import csv  # Add this import at the top of the file




############# EVERYTHING BELOW IS OUTDATED

# Setup
load_dotenv()
pytestmark = pytest.mark.integration
pytest_plugins = ('pytest_asyncio',)
logger = setup_logging()


async def general_extract_nodes_test(llm_client, data_sample):
    episode = data_sample['episode']
    previous_episodes = data_sample['previous_episodes']
    gold_answer_names = data_sample['gold_answer_names']

    hypothesis_nodes = await extract_nodes(llm_client, episode, previous_episodes)
    hypothesis_node_names = [node.name for node in hypothesis_nodes]

    # Sort both lists by node name
    hypothesis_node_names.sort()
    gold_answer_names.sort()

    # assert hypothesis_node_names == gold_answer_names, \
    #     f"""Test Failed. Expected nodes: {gold_answer_names}. Got: {hypothesis_node_names}"""

    return hypothesis_node_names





def prepare_data_from_csv(data_file_name, question_id, session_idx, message_idx):

    samples_csv_path = "tests/evals/data/" + data_file_name + ".csv"

    # From CSV path, load everything
    with open(samples_csv_path, 'r') as file:
        csv_reader = csv.DictReader(file)
        lme_samples = list(csv_reader)


    data_samples = []

    # Loop through each row
    for row in lme_samples:

        ### Prepare episode
        current_time = datetime.now()
        message = json.loads(row["message"])
        role = message["role"]
        content = message["content"]
        message_content = role + ": " + content
        episode = EpisodicNode(
            name="",
            group_id="",
            source=EpisodeType.message,
            type=EpisodeType.message,
            source_description="",
            content=message_content,
            valid_at=current_time, 
        )

        ### Prepare previous episodes
        previous_messages = json.loads(row["previous_messages"])
        num_previous_messages = len(previous_messages)
        previous_times = [current_time - timedelta(minutes=num_previous_messages-i) for i in range(num_previous_messages)]
        previous_episodes = [EpisodicNode(
            name="",
            group_id="",
            source=EpisodeType.message,
            source_description="",
            content=message["role"] + ": " + message["content"],
            valid_at=previous_time,
        ) for message, previous_time in zip(previous_messages, previous_times)]

        ### TODO: Prepare gold answer names

        ### Add to data samples list
        data_samples.append({
            "episode": episode,
            "previous_episodes": previous_episodes,
            "gold_answer_names": [],
        })

    return data_samples





@pytest.mark.asyncio
async def test_extract_nodes():
    model_name = 'gpt-4o-mini'
    llm_config = LLMConfig(
        api_key=os.getenv('OPENAI_API_KEY'),
        model=model_name,
    )
    llm_client = OpenAIClient(config=llm_config)

    data_file_name = 'output_short'
    question_id = "gpt4_2655b836"
    session_idx = 0
    message_idx = 0
    data_samples = prepare_data_from_csv(data_file_name, question_id, session_idx, message_idx)

    for data_sample in data_samples:
        print(f"\n\nEpisode: {data_sample['episode']}")
        print("*"*50)
        print(f"Previous Episodes: {data_sample['previous_episodes']}")
        print("*"*50)
        # print(f"Gold Answer Names: {gold_answer_names}")

        await general_extract_nodes_test(llm_client, data_sample)

