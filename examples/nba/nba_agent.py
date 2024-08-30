import asyncio
import json
import logging
import operator
import os
import re
from datetime import datetime
from typing import Annotated, Any, Dict, List, TypedDict

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType

logging.getLogger('langchain.callbacks.tracers.langchain').setLevel(logging.WARNING)
logging.getLogger('urllib3.connectionpool').setLevel(logging.ERROR)
DEFAULT_MODEL = 'gpt-4o-mini'
VALID_TEAMS = [
    'Toronto Raptors',
    'Boston Celtics',
    # 'Golden State Warriors',
    # 'Miami Heat',
    # 'Los Angeles Lakers',
]
load_dotenv()
logging.basicConfig(
    level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('nba_agent')
for name in logging.root.manager.loggerDict:
    if name != 'nba_agent':
        logging.getLogger(name).setLevel(logging.WARNING)

# Initialize Graphiti client
neo4j_uri = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
neo4j_user = os.environ.get('NEO4J_USER', 'neo4j')
neo4j_password = os.environ.get('NEO4J_PASSWORD', 'password')

graphiti_client = Graphiti(neo4j_uri, neo4j_user, neo4j_password)

openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    logger.error('OPENAI_API_KEY is not set in the environment variables.')
    raise ValueError('OPENAI_API_KEY is not set')


# Define the SimulationState
class SimulationState(TypedDict):
    messages: Annotated[List[str], operator.add]
    teams: Dict[str, Dict[str, int]]  # Changed to only store budget
    event: str
    transfer_offers: Annotated[List[Dict], operator.add]
    current_iteration: int
    all_events: List[str]
    max_iterations: int
    teams_context: Dict[str, List[dict]]


@tool
async def fetch_all_teams_context(teams: List[str]):
    """Get the current roster and player summaries for specified teams."""
    teams_with_players_dict = {}
    llm = ChatOpenAI(temperature=0.2, model=DEFAULT_MODEL).bind(
        response_format={'type': 'json_object'}
    )

    for team in teams:
        team_nodes = await graphiti_client.get_nodes_by_query(team, 1)
        if not team_nodes:
            logger.warning(f'No nodes found for team: {team}')
            continue

        team_node = team_nodes[0]
        search_result = await graphiti_client.search(
            f'plays for {team_node.name}',
            center_node_uuid=team_node.uuid,
            num_results=30,
        )

        # Include all facts and timestamps (expired at if exists)
        roster_facts = [get_fact_string(edge) for edge in search_result]

        prompt = PromptTemplate.from_template("""
        Given the following list of facts about players and their teams, extract only the names and provide brief summaries for players who currently play for {team_name}. Follow these guidelines:

        1. Only include players who are currently on the team.
        2. Discard any information about players who are no longer on the team or were never on the team.
        3. Use the 'expired_at' field to determine if a fact is still current. If 'expired_at' is not null, the fact is no longer current.
        4. If there are conflicting facts, use the most recent one based on the 'valid_at' or 'created_at' timestamps.

        Return the information as a JSON object containing a "players" field, which is an array of objects, each containing 'name' and 'summary' fields
        example output:
        {{
            "players": [
                {{
                    "name": "Player Name",
                    "summary": "Brief summary of the player. Make sure you include all the information you can get from the roster facts about the player. Do not include any information that is not in the roster facts. If dates are mentioned, make sure to integrate them well in the summary. If applicable it can tell the course of events"
                }},
                ...
            ]
        }}
        Facts:
        {facts}

        Current players for {team_name}:
        """)

        llm_response = await llm.ainvoke(
            prompt.format(
                team_name=team_node.name,
                facts='\n'.join(roster_facts),
            )
        )

        try:
            result = json.loads(llm_response.content)
            players = result.get('players', [])
            if not isinstance(players, list):
                raise ValueError('Expected a JSON array')
        except json.JSONDecodeError:
            logger.error(f'Failed to parse JSON from LLM response for {team_node.name}')
            players = []
        except ValueError as e:
            logger.error(f'Invalid data structure in LLM response for {team_node.name}: {e}')
            players = []

        teams_with_players_dict[team_node.name] = players

    return teams_with_players_dict


# Existing tools
@tool
async def get_team_roster(team_name: str):
    """Get the current roster for a specific team."""
    search_result = await graphiti_client.search(f'plays for {team_name}', num_results=30)
    roster_facts = [get_fact_string(edge) for edge in search_result]

    # Use LLM to extract player names
    llm = ChatOpenAI(temperature=0, model=DEFAULT_MODEL)
    prompt = PromptTemplate.from_template("""
    Given the following list of facts about players and their teams, extract only the names of players who play for {team_name}. Return the names as a comma-separated list.

    Facts:
    {facts}

    Players who play for {team_name}:
    """)

    llm_response = await llm.ainvoke(
        prompt.format(team_name=team_name, facts='\n'.join(roster_facts))
    )

    player_names = [name.strip() for name in llm_response.content.split(',')]

    return f"{team_name}'s roster: {', '.join(player_names)}"


@tool
async def search_player_info(player_name: str):
    """Search for information about a specific player."""
    search_result = await graphiti_client.search(f'{player_name}', num_results=30)
    all_facts = [get_fact_string(edge) for edge in search_result]

    # Use LLM to extract relevant player information
    llm = ChatOpenAI(temperature=0, model=DEFAULT_MODEL)
    prompt = PromptTemplate.from_template("""
    Given the following list of facts, extract only the information that is relevant to {player_name}. 
    Return the relevant facts as a list, with each fact on a new line.

    Facts:
    {facts}

    Relevant facts about {player_name}:
    """)

    llm_response = await llm.ainvoke(
        prompt.format(player_name=player_name, facts='\n'.join(all_facts))
    )

    relevant_facts = llm_response.content.strip().split('\n')

    return {'name': player_name, 'facts': relevant_facts}


@tool
async def execute_transfer(
    player_name: str, from_team: str, to_team: str, price: int, reason: str
) -> Dict[str, Any]:
    """Execute a transfer between two teams."""
    await graphiti_client.add_episode(
        name=f'Transfer {player_name}',
        episode_body=f'{player_name} transferred from {from_team} to {to_team} for ${price:,}',
        source_description='Player Transfer',
        reference_time=datetime.now(),
        source=EpisodeType.message,
    )
    return {
        'messages': [
            HumanMessage(
                content=f'Transfer executed: {player_name} moved from {from_team} to {to_team} for ${price:,} Reason: {reason}'
            )
        ],
    }


async def add_episode(event_description: str):
    """Add a new episode to the Graphiti client."""
    logger.info(f'Adding episode: {event_description}')
    await graphiti_client.add_episode(
        name='New Event',
        episode_body=event_description,
        source_description='User Input',
        reference_time=datetime.now(),
        source=EpisodeType.message,
    )
    return f"Episode '{event_description}' added successfully."


def get_fact_string(edge):
    fact_string = f'{edge.fact} Valid At: {edge.valid_at or edge.created_at}'
    if edge.expired_at:
        fact_string += f' Expired At: {edge.expired_at}'
    return fact_string


tools = [
    get_team_roster,
    search_player_info,
    execute_transfer,
]


# Define the team agent function
def create_team_agent(team_name: str):
    llm = ChatOpenAI(temperature=0.3, model=DEFAULT_MODEL).bind(
        response_format={'type': 'json_object'}
    )
    prompt = ChatPromptTemplate.from_template("""You are the manager of the {team_name} NBA team. Make decisions to improve your team.

Current event: {event}

Your task is to decide on an action based on the event. 
Use the available tools to gather information, but focus on making a decision quickly. 
If you think a player transfer would benefit your team, propose one following the guidelines below. Make sure to get familiar with the entire transfer history of a given player
Ensure that you use the current budget info and the current state of your team (use an appropriate tool to get the current state of your team) to make the best decision.
Current budget: ${budget}

Valid teams for transfers: {valid_teams}
Do not propose transfers you cannot afford.
IMPORTANT: After gathering information, you MUST make a decision. Your options are:
1. Propose a transfer
    Note: if you are proposing a transfer make sure to output JSON in the following format:
    {{
        "transfer_proposal": {{
            "to_team": "team_name",
            "from_team": "team_name",
            "player_name": "player_name",
            "proposed_price": price,
            "reason": "reason for the proposed transfer"
        }}
    }}
    IMPORTANT: Only propose transfers to teams in the valid teams list. Make sure that the player_name is a valid player on the from_team. Ensure that the the from_team name is a valid team name.
2. Do nothing (output the following JSON object)
    {{
        "no_transfer": "reason for not proposing a transfer"
    }}

Do not ask for more information or clarification. Make a decision based on what you know.

{agent_scratchpad}""")

    async def team_agent_function(state: SimulationState) -> Dict:
        agent = create_openai_functions_agent(llm, tools, prompt)
        executor = AgentExecutor(
            agent=agent, tools=[get_team_roster, search_player_info], verbose=True
        )
        team_data = state['teams'][team_name]

        result = await executor.ainvoke(
            {
                'team_name': team_name,
                'event': state['event'],
                'budget': team_data['budget'],
                'valid_teams': ', '.join(VALID_TEAMS),
            }
        )

        json_result = json.loads(result['output'])
        messages = []
        transfer_offer = None
        if 'transfer_proposal' in json_result:
            transfer_offer = json_result['transfer_proposal']
            logger.info(f'Transfer proposal made by {team_name}: {transfer_offer["reason"]}')
            messages.append(f'Transfer proposal made by {team_name}: {transfer_offer["reason"]}')
            if (
                transfer_offer['to_team'] not in VALID_TEAMS
                or transfer_offer['from_team'] not in VALID_TEAMS
            ):
                logger.warning(f'Invalid transfer proposal: {transfer_offer}. Ignoring.')
                transfer_offer = None
        if 'no_transfer' in json_result:
            logger.info(f'No transfer proposal made by {team_name}: {json_result["no_transfer"]}')
            messages.append(
                f'No transfer proposal made by {team_name}: {json_result["no_transfer"]}'
            )
        return {
            'transfer_offers': [transfer_offer] if transfer_offer else [],
            'messages': messages,
        }

    return team_agent_function


async def process_event(state: SimulationState) -> SimulationState:
    await add_episode(state['event'])
    return {
        **state,
        'messages': [f"Event processed: {state['event']}"],
        'transfer_offers': [],
    }


async def process_transfers(state: SimulationState) -> SimulationState:
    if not state['transfer_offers']:
        return state

    # Group offers by player
    offers_by_player = {}
    for offer in state['transfer_offers']:
        player = offer['player_name']
        if player not in offers_by_player:
            offers_by_player[player] = []
        offers_by_player[player].append(offer)

    for player, offers in offers_by_player.items():
        # Sort offers by price, highest first
        best_offer = max(offers, key=lambda x: x['proposed_price'])

        logger.info(
            f"Best offer for {player}: {best_offer['to_team']} wants to buy from {best_offer['from_team']} for ${best_offer['proposed_price']:,}"
        )

        # Execute the transfer
        transfer_result = await execute_transfer.ainvoke(
            {
                'player_name': best_offer['player_name'],
                'from_team': best_offer['from_team'],
                'to_team': best_offer['to_team'],
                'price': best_offer['proposed_price'],
                'reason': best_offer['reason'],
            }
        )
        # Add the transfer result message to the state
        state['messages'] = [transfer_result['messages']]

        # Update team rosters and budgets
        from_team = best_offer['from_team']
        to_team = best_offer['to_team']
        price = best_offer['proposed_price']

        if from_team in state['teams'] and to_team in state['teams']:
            state['teams'][from_team]['budget'] += price
            state['teams'][to_team]['budget'] -= price
        else:
            logger.warning(f'Cannot process transfer: {from_team} or {to_team} not in simulation.')

    # Clear all processed offers
    state['transfer_offers'] = []

    return state


def create_simulator_agent():
    llm = ChatOpenAI(
        temperature=0.7, model=DEFAULT_MODEL
    )  # Higher temperature for more creative events
    prompt = ChatPromptTemplate.from_template("""
    You are an NBA event simulator. 
    Your role is to generate realistic events based on the current state of NBA teams and players. 
    Use the provided team and player information to create engaging and plausible scenarios.
    Ensure that you use as much as possible from the teams_context to create the event. 
    Use the existing events to get a sense of the narrative unfolding.
    Current NBA landscape:
    {teams_context}

    Generate a single, specific event involving one or more teams or players. The event should be impactful enough to potentially influence team decisions. Examples include outstanding performances, injuries, trade rumors, or off-court incidents.

    Output the event as a brief, news-like statement.

    Event:
    """)

    return prompt, llm


simulator_prompt, simulator_llm = create_simulator_agent()


def create_analyzer_agent():
    llm = ChatOpenAI(temperature=0.7, model='gpt-4o')  # Higher temperature for more creative events
    prompt = ChatPromptTemplate.from_template("""
    You are an NBA Simulation Turn analyzer. 
    Your task is to compare the previous state of the graph (before the last turn) and the current state of the graph (after the last turn) and provide a brief analysis of the changes that occurred.
    Make sure to put more emphasis on the transfers or reasons why transfers were not made.

    Previous state of the graph:
    {previous_state}

    Current state of the graph:
    {current_state}
                                              
    Changes that occurred during the last turn:
    {changes}
    
    Output the analysis as a brief summary of the changes that occurred.

    Analysis:
    """)

    return prompt, llm


analyzer_prompt, analyzer_llm = create_analyzer_agent()


async def simulate_event(state: SimulationState) -> SimulationState:
    teams_context = await fetch_all_teams_context.ainvoke({'teams': VALID_TEAMS})
    result = await simulator_llm.ainvoke(
        simulator_prompt.format_prompt(teams_context=json.dumps(teams_context, indent=2))
    )

    new_event = result.content
    existing_events = state['all_events'] or []
    existing_events.append(new_event)
    return {
        **state,
        'event': new_event,
        'teams_context': teams_context,
        'all_events': existing_events,
        'transfer_offers': [],
        'current_iteration': state['current_iteration'] + 1,
    }


async def analyze_simulation_turn(state: SimulationState) -> SimulationState:
    current_teams_context = state['teams_context']
    updated_team_context = await fetch_all_teams_context.ainvoke({'teams': VALID_TEAMS})
    result = await analyzer_llm.ainvoke(
        analyzer_prompt.format_prompt(
            previous_state=json.dumps(current_teams_context, indent=2),
            current_state=json.dumps(updated_team_context, indent=2),
            changes=state['messages'],
        )
    )

    summary = result.content
    logger.info(f'Analysis of the last turn: {summary}')
    return state


# Create the graph
workflow = StateGraph(SimulationState)

# Add nodes
workflow.add_node('simulate_event', simulate_event)
workflow.add_node('process_event', process_event)
for team in VALID_TEAMS:
    workflow.add_node(f'agent_{team}', create_team_agent(team))
workflow.add_node('process_transfers', process_transfers)
workflow.add_node('analyze_simulation_turn', analyze_simulation_turn)
# Add edges
workflow.add_edge(START, 'simulate_event')
workflow.add_edge('simulate_event', 'process_event')

# Add edges from process_event to all agent nodes
for team in VALID_TEAMS:
    workflow.add_edge('process_event', f'agent_{team}')

for team in VALID_TEAMS:
    workflow.add_edge(f'agent_{team}', 'process_transfers')


def routing_function(state: SimulationState) -> str:
    if state['current_iteration'] >= state['max_iterations']:
        return END
    else:
        return 'simulate_event'


workflow.add_edge('process_transfers', 'analyze_simulation_turn')

workflow.add_conditional_edges(
    'analyze_simulation_turn',
    routing_function,
)

# Compile the graph
app = workflow.compile()

print(app.get_graph().draw_mermaid())


async def run_simulation():
    num_iterations = int(input('Enter the number of simulation iterations: '))
    teams = {}
    for t in VALID_TEAMS:
        teams[t] = {'budget': 100_000_000}
    initial_state = SimulationState(
        messages=[],
        teams=teams,
        event='',
        transfer_offers=[],
        current_iteration=0,
        max_iterations=num_iterations,
    )

    final_state = await app.ainvoke(initial_state, {'recursion_limit': 200})

    print('\nFinal team states:')
    for team_name, team_data in final_state['teams'].items():
        print(f"{team_name} - Budget: ${team_data['budget']:,}")
    print(f'Steps taken: {final_state["current_iteration"]}')
    for event in final_state['all_events']:
        print('\n')
        print(event)
        print('\n')


if __name__ == '__main__':
    asyncio.run(run_simulation())
