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

import asyncio
import csv
import json
import logging
import os
import sys
from time import time

from dotenv import load_dotenv

from examples.multi_session_conversation_memory.parse_msc_messages import conversation_q_and_a
from graphiti_core import Graphiti
from graphiti_core.prompts import prompt_library
from graphiti_core.search.search_config_recipes import COMBINED_HYBRID_SEARCH_RRF

load_dotenv()

neo4j_uri = os.environ.get('NEO4J_URI') or 'bolt://localhost:7687'
neo4j_user = os.environ.get('NEO4J_USER') or 'neo4j'
neo4j_password = os.environ.get('NEO4J_PASSWORD') or 'password'


def setup_logging():
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Set the logging level to INFO

    # Create console handler and set level to INFO
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Add formatter to console handler
    console_handler.setFormatter(formatter)

    # Add console handler to logger
    logger.addHandler(console_handler)

    return logger


async def main():
    setup_logging()
    graphiti = Graphiti(neo4j_uri, neo4j_user, neo4j_password)

    qa = conversation_q_and_a()[0:5]

    fields = ['Group id', 'Question', 'Answer', 'Response', 'Score', 'Search Duration (ms)']
    msc_eval_map: list[dict] = []

    for group_id, (query, answer) in enumerate(qa):
        search_start = time()
        results = await graphiti._search(query, COMBINED_HYBRID_SEARCH_RRF, group_ids=[str(group_id)])
        search_end = time()
        search_duration = (search_end - search_start) * 1000

        facts = [edge.fact for edge in results.edges]
        entity_summaries = [node.name + ': ' + node.summary for node in results.nodes]
        context = {'facts': facts, 'entity_summaries': entity_summaries, 'query': query}

        llm_response = await graphiti.llm_client.generate_response(prompt_library.eval.qa_prompt(context))
        response = llm_response.get('ANSWER', '')

        eval_context = {'query': query, 'answer': answer, 'response': response}

        eval_llm_response = await graphiti.llm_client.generate_response(prompt_library.eval.eval_prompt(eval_context))
        eval_response = 1 if eval_llm_response.get('is_correct', False) else 0
        msc_eval_map.append({'Group id': int(group_id), 'Question': query, 'Answer': answer, 'Response': response,
                             'Score': eval_response, 'Search Duration (ms)': search_duration})

    with open('../data/msc_eval.csv', 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)

        writer.writerows(msc_eval_map)

    await graphiti.close()


asyncio.run(main())
