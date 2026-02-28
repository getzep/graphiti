"""
Copyright 2025, Zep Software, Inc.

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
from datetime import datetime, timezone
from logging import INFO

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from graphiti_core import Graphiti
from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.gemini_client import GeminiClient
from graphiti_core.llm_client.gliner2_client import GLiNER2Client
from graphiti_core.nodes import EpisodeType

#################################################
# CUSTOM ENTITY TYPES
#################################################
# Define Pydantic models for entity classification.
# GLiNER2 uses the class docstrings as label
# descriptions for improved extraction accuracy.
# The LLM client uses these for edge extraction
# and summarization.
#################################################


class Person(BaseModel):
    """A human person, real or fictional."""

    occupation: str | None = Field(None, description='Professional role or job title')
    political_party: str | None = Field(None, description='Political party affiliation')


class Organization(BaseModel):
    """An organization such as a company, government agency, university, or political party."""

    org_type: str | None = Field(
        None, description='Type of organization (e.g., bank, university, government agency)'
    )


class Location(BaseModel):
    """A geographic location such as a city, state, or country."""

    location_type: str | None = Field(
        None, description='Type of location (e.g., city, state, county)'
    )


class Initiative(BaseModel):
    """A program, policy, initiative, or legal action."""

    description: str | None = Field(None, description='Brief description of the initiative')


entity_types: dict[str, type[BaseModel]] = {
    'Person': Person,
    'Organization': Organization,
    'Location': Location,
    'Initiative': Initiative,
}

#################################################
# CONFIGURATION
#################################################
# GLiNER2 is a lightweight extraction model
# (205M-340M params) that runs locally on CPU.
# It handles entity extraction (NER), while an
# OpenAI client handles edge/fact extraction,
# deduplication, summarization, and reasoning.
#################################################

# Configure logging
logging.basicConfig(
    level=INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

load_dotenv()

# Neo4j connection parameters
neo4j_uri = os.environ.get('NEO4J_URI')
neo4j_user = os.environ.get('NEO4J_USER')
neo4j_password = os.environ.get('NEO4J_PASSWORD')

if not neo4j_uri or not neo4j_user or not neo4j_password:
    raise ValueError('NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD must be set')

# GLiNER2 model configuration
gliner2_model = os.environ.get('GLINER2_MODEL', 'fastino/gliner2-large-v1')


async def main():
    #################################################
    # INITIALIZATION
    #################################################
    # Set up a hybrid LLM client: GLiNER2 handles
    # entity extraction locally using custom entity
    # types as labels, while OpenAI handles edge/fact
    # extraction, deduplication, and summarization.
    #################################################

    # Create the Gemini client for reasoning tasks
    gemini_client = GeminiClient(
        config=LLMConfig(
            api_key=os.environ.get('GOOGLE_API_KEY'),
            model='gemini-2.5-flash-lite',
            small_model='gemini-2.5-flash-lite',
        ),
    )

    # Create the GLiNER2 hybrid client
    gliner2_client = GLiNER2Client(
        config=LLMConfig(model=gliner2_model),
        llm_client=gemini_client,
        threshold=0.7,
    )

    # Create the Gemini embedder
    gemini_embedder = GeminiEmbedder(
        config=GeminiEmbedderConfig(
            api_key=os.environ.get('GOOGLE_API_KEY'),
            embedding_model='gemini-embedding-001',
        ),
    )

    # Initialize Graphiti with the GLiNER2 hybrid client and Gemini embedder
    graphiti = Graphiti(
        neo4j_uri,
        neo4j_user,
        neo4j_password,
        llm_client=gliner2_client,
        embedder=gemini_embedder,
    )

    try:
        #################################################
        # ADDING EPISODES
        #################################################
        # Entity extraction from these episodes will be
        # handled by GLiNER2 locally using the custom
        # entity types as labels. Edge/fact extraction,
        # deduplication, and summarization are delegated
        # to OpenAI.
        #################################################

        episodes = [
            # English: detailed political biography
            {
                'content': (
                    'Kamala Harris is the Attorney General of California. She was previously '
                    'the district attorney for San Francisco. Harris graduated from Howard '
                    'University in 1986 and earned her law degree from the University of '
                    'California, Hastings College of the Law in 1989. Before entering politics, '
                    'she worked as a deputy district attorney in Alameda County under District '
                    'Attorney John Orlovsky. In 2003, she defeated incumbent Terence Hallinan '
                    'to become San Francisco District Attorney, making her the first woman and '
                    'first African American to hold the position.'
                ),
                'type': EpisodeType.text,
                'description': 'podcast transcript',
            },
            {
                'content': (
                    'As AG, Harris was in office from January 3, 2011 to January 3, 2017. '
                    'During her tenure she launched the OpenJustice initiative, a data platform '
                    'for criminal justice statistics across California. She also led a $25 billion '
                    'national mortgage settlement against Bank of America, JPMorgan Chase, Wells '
                    'Fargo, Citigroup, and Ally Financial on behalf of homeowners affected by '
                    'the foreclosure crisis.'
                ),
                'type': EpisodeType.text,
                'description': 'podcast transcript',
            },
            # Spanish: same entities (Kamala Harris, California, San Francisco)
            {
                'content': (
                    'Kamala Harris fue la Fiscal General de California entre 2011 y 2017. '
                    'Anteriormente se desempeñó como fiscal de distrito de San Francisco. '
                    'Harris es graduada de la Universidad Howard y obtuvo su título de abogada '
                    'en la Facultad de Derecho Hastings de la Universidad de California. Durante '
                    'su mandato como Fiscal General, impulsó reformas en el sistema de justicia '
                    'penal del estado.'
                ),
                'type': EpisodeType.text,
                'description': 'artículo de noticias',
            },
            # French: same entities (Kamala Harris, California, San Francisco)
            {
                'content': (
                    'Kamala Harris a été procureure générale de Californie de 2011 à 2017. '
                    'Avant cela, elle a occupé le poste de procureure du district de '
                    'San Francisco. Elle est diplômée de l\'Université Howard et a obtenu '
                    'son diplôme de droit au Hastings College of the Law de l\'Université de '
                    'Californie. En tant que procureure générale, elle a négocié un accord '
                    'national de 25 milliards de dollars avec les grandes banques américaines.'
                ),
                'type': EpisodeType.text,
                'description': 'article de presse',
            },
            # JSON: structured political metadata
            {
                'content': {
                    'name': 'Gavin Newsom',
                    'position': 'Governor',
                    'state': 'California',
                    'previous_role': 'Lieutenant Governor',
                    'previous_location': 'San Francisco',
                    'party': 'Democratic Party',
                    'took_office': '2019-01-07',
                    'predecessor': 'Jerry Brown',
                },
                'type': EpisodeType.json,
                'description': 'political leadership metadata',
            },
            # Portuguese: overlapping entities (California, San Francisco, Gavin Newsom)
            {
                'content': (
                    'Gavin Newsom é o governador da Califórnia desde janeiro de 2019. '
                    'Antes disso, ele foi prefeito de San Francisco de 2004 a 2011 e '
                    'vice-governador da Califórnia de 2011 a 2019. Newsom é membro do '
                    'Partido Democrata e tem promovido políticas progressistas em áreas '
                    'como mudanças climáticas, imigração e reforma da justiça criminal.'
                ),
                'type': EpisodeType.text,
                'description': 'perfil político',
            },
        ]

        for i, episode in enumerate(episodes):
            result = await graphiti.add_episode(
                name=f'California Politics {i}',
                episode_body=(
                    episode['content']
                    if isinstance(episode['content'], str)
                    else json.dumps(episode['content'])
                ),
                source=episode['type'],
                source_description=episode['description'],
                reference_time=datetime.now(timezone.utc),
                entity_types=entity_types,
            )

            print(f'\n--- Episode: California Politics {i} ({episode["type"].value}) ---')

            if result.nodes:
                print(f'  Entities ({len(result.nodes)}):')
                for node in result.nodes:
                    labels_str = ', '.join(node.labels) if node.labels else 'Entity'
                    print(f'    - {node.name} [{labels_str}]')
                    if node.summary:
                        print(f'      Summary: {node.summary}')
                    if node.attributes:
                        print(f'      Attributes: {node.attributes}')

            if result.edges:
                print(f'  Edges ({len(result.edges)}):')
                for edge in result.edges:
                    temporal = ''
                    if edge.valid_at:
                        temporal += f' (valid: {edge.valid_at.isoformat()})'
                    if edge.invalid_at:
                        temporal += f' (invalid: {edge.invalid_at.isoformat()})'
                    print(f'    - [{edge.name}] {edge.fact}{temporal}')

        #################################################
        # SEARCH
        #################################################

        queries = [
            'Who was the California Attorney General?',
            'What banks were involved in the mortgage settlement?',
            'What is the relationship between Kamala Harris and San Francisco?',
        ]

        for query in queries:
            print(f"\nSearching for: '{query}'")
            results = await graphiti.search(query)

            print('Results:')
            for result in results:
                print(f'  Fact: {result.fact}')
                if hasattr(result, 'valid_at') and result.valid_at:
                    print(f'  Valid from: {result.valid_at}')
                if hasattr(result, 'invalid_at') and result.invalid_at:
                    print(f'  Valid until: {result.invalid_at}')
                print('  ---')

        #################################################
        # ENTITY EXTRACTION LATENCY
        #################################################

        latencies = gliner2_client.extraction_latencies
        if latencies:
            print(f'\nGLiNER2 entity extraction latency ({len(latencies)} calls):')
            print(f'  Mean:  {sum(latencies) / len(latencies):.1f} ms')
            print(f'  Min:   {min(latencies):.1f} ms')
            print(f'  Max:   {max(latencies):.1f} ms')
            print(f'  Total: {sum(latencies):.1f} ms')

    finally:
        await graphiti.close()
        print('\nConnection closed')


if __name__ == '__main__':
    asyncio.run(main())
