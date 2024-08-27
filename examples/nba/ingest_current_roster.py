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
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from nba_api.stats.endpoints import commonteamroster, teamdetails
from nba_api.stats.static import players, teams

from graphiti_core import Graphiti
from graphiti_core.llm_client.anthropic_client import AnthropicClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.nodes import EpisodeType
from graphiti_core.utils.bulk_utils import RawEpisode
from graphiti_core.utils.maintenance.graph_data_operations import clear_data

load_dotenv()

neo4j_uri = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
neo4j_user = os.environ.get('NEO4J_USER', 'neo4j')
neo4j_password = os.environ.get('NEO4J_PASSWORD', 'password')


class PlayerInfo(TypedDict):
    team_name: str
    player_id: int
    player_name: str
    player_number: str
    player_position: str
    player_school: str


def setup_logging():
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Set the logging level to INFO

    # Create console handler and set level to INFO
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')

    # Add formatter to console handler
    console_handler.setFormatter(formatter)

    # Add console handler to logger
    logger.addHandler(console_handler)

    return logger


def fetch_current_roster():
    all_teams = teams.get_teams()
    players_json = []
    for t in all_teams:
        name = t['full_name']
        print(name)
        if name == 'Golden State Warriors' or name == 'Boston Celtics' or name == 'Toronto Raptors':
            roster = commonteamroster.CommonTeamRoster(team_id=t['id']).get_dict()
            players_data = roster['resultSets'][0]
            headers = players_data['headers']
            row_set = players_data['rowSet']

            for row in row_set:
                player_dict = dict(zip(headers, row))
                player_dict['team_name'] = name
                print(player_dict)
                meaningful_data = {
                    'team_name': name,
                    'player_id': player_dict['PLAYER_ID'],
                    'player_name': player_dict['PLAYER'],
                    # 'player_number': player_dict['NUM'],
                    # 'player_position': player_dict['POSITION'],
                    # 'player_school': player_dict['SCHOOL'],
                }
                players_json.append(meaningful_data)
    script_dir = Path(__file__).parent
    filename = script_dir / 'current_nba_roster.json'
    print(players_json)
    with open(filename, 'w') as f:
        # write the players_json to the file and clear the file before doing so
        f.truncate(0)
        json.dump(players_json, f, indent=2)


async def main():
    # fetch_current_roster()
    current_roster_from_file: list[PlayerInfo] = []
    script_dir = Path(__file__).parent
    filename = script_dir / 'current_nba_roster.json'
    with open(filename) as f:
        current_roster_from_file = json.load(f)
    print(current_roster_from_file)
    client = Graphiti(neo4j_uri, neo4j_user, neo4j_password)
    await clear_data(client.driver)
    await client.build_indices_and_constraints()

    episodes: list[RawEpisode] = [
        RawEpisode(
            name=f'Player {player["player_id"]}',
            content=str(player),
            source_description='NBA current roster',
            source=EpisodeType.json,
            reference_time=datetime.now(),
        )
        for i, player in enumerate(current_roster_from_file)
    ]

    await client.add_episode_bulk(episodes)
    # client.llm_client = AnthropicClient(LLMConfig(api_key=os.environ.get('ANTHROPIC_API_KEY')))
    # await client.add_episode(
    #     name='Player Transfer',
    #     episode_body='DJ Carton got transeffered to Boston Celtics August 2nd',
    #     source_description='NBA transfer',
    #     reference_time=datetime.now(),
    #     source=EpisodeType.message,
    # )


if __name__ == '__main__':
    asyncio.run(main())
