import asyncio
import logging
import os
from typing import TypedDict, Dict, List, Optional, Any

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import PromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt.tool_executor import ToolExecutor
from datetime import datetime
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType

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

MAX_NEGOTIATION_ROUNDS = 5


def format_step_result(messages: List[str], **kwargs) -> Dict[str, Any]:
    return {'messages': messages, **kwargs}


class SimulationState(TypedDict):
    messages: List[str]  # Changed from HumanMessage to str for simplicity
    teams: Dict[str, Dict[str, Any]]  # Store team data as a dictionary
    event: str
    team_actions: Dict[str, str]
    transfer_offers: List[Dict[str, Any]]
    current_negotiation: Optional[Dict[str, Any]]
    negotiation_rounds: int
    negotiation_complete: bool


class TeamAgent:
    def __init__(self, name: str, tools: List[Any]):
        self.name = name
        self.roster: List[str] = []
        self.budget: int = 100_000_000
        self.tools = tools
        self.last_proposed_transfer: Optional[Dict[str, Any]] = None

        # Create the language model
        llm = ChatOpenAI(temperature=0.3)

        # Create the agent executor
        template = """You are the manager of the {team_name} NBA team. Make decisions to improve your team.

Current event: {event}

Your task is to decide on an action based on the event. Use the available tools to gather information and make decisions. Do not ask for further input. Instead, take action based on the information you have.

If you decide to propose a transfer, use the propose_transfer tool and include the exact output from the tool in your response, prefixed with "TRANSFER PROPOSAL:".

{agent_scratchpad}"""

        prompt = PromptTemplate(
            input_variables=['team_name', 'event', 'agent_scratchpad'], template=template
        )

        agent = create_openai_functions_agent(llm, self.tools, prompt)
        self.executor = AgentExecutor(agent=agent, tools=self.tools, verbose=True)

    async def process_event(self, event: str, relevant_offers: List[Dict[str, Any]]) -> str:
        logger.debug(f'{self.name}: Processing event: {event}')
        logger.debug(f'{self.name}: Current roster: {self.roster}')
        logger.debug(f'{self.name}: Current budget: {self.budget}')

        try:
            result = await self.executor.ainvoke(
                {
                    'team_name': self.name,
                    'event': event,
                    'roster': self.roster,
                    'budget': self.budget,
                    'relevant_offers': relevant_offers,
                }
            )
            logger.debug(f"{self.name}: Agent output: {result['output']}")

            # Check if a transfer was proposed
            if 'TRANSFER PROPOSAL:' in result['output']:
                # Parse the transfer details and set last_proposed_transfer
                transfer_details = result['output'].split('TRANSFER PROPOSAL:')[-1].strip()
                self.last_proposed_transfer = self.parse_transfer_proposal(transfer_details)
                logger.debug(f'{self.name}: Proposed transfer: {self.last_proposed_transfer}')

            return result['output']
        except Exception as e:
            logger.error(f'{self.name}: Error processing event: {str(e)}')
            return f'Error processing event: {str(e)}'

    def parse_transfer_proposal(self, proposal: str) -> Dict[str, Any]:
        # More flexible parsing
        parts = proposal.lower().replace(',', '').split()
        to_team = next(parts[i - 1] for i, word in enumerate(parts) if word == 'to')
        from_team = next(parts[i - 1] for i, word in enumerate(parts) if word == 'from')
        player_name = next(parts[i + 1] for i, word in enumerate(parts) if word == 'buy')
        proposed_price = int(''.join(filter(str.isdigit, parts[-1])))

        return {
            'to_team': to_team,
            'from_team': from_team,
            'player_name': player_name,
            'proposed_price': proposed_price,
        }

    async def propose_transfer(self, player_name: str, to_team: str, price: int):
        self.last_proposed_transfer = {
            'from_team': self.name,
            'to_team': to_team,
            'player_name': player_name,
            'proposed_price': price,
        }
        return f'Proposed transfer of {player_name} to {to_team} for ${price:,}'

    async def update_budget(self, amount: int):
        """Update the team's budget."""
        self.budget += amount
        logger.debug(f"{self.name}'s new budget: ${self.budget:,}")
        return f"{self.name}'s new budget: ${self.budget:,}"

    async def propose_transfers(self) -> List[Dict[str, Any]]:
        """Propose transfer offers based on the agent's strategy."""
        # This is a placeholder implementation. In a real scenario, this would involve more complex logic.
        if hasattr(self, 'last_proposed_transfer'):
            return [self.last_proposed_transfer]
        return []

    async def update_roster(self):
        """Update the team's roster using the get_team_roster tool."""
        roster_tool = next(tool for tool in self.tools if tool.name == 'get_team_roster')
        roster_string = await roster_tool.ainvoke(self.name)
        self.roster = roster_string.split(': ')[1].split(', ')

    def remove_player(self, player_name: str):
        """Remove a player from the team's roster."""
        if player_name in self.roster:
            self.roster.remove(player_name)
            logger.debug(f"{player_name} removed from {self.name}'s roster")
        else:
            logger.warning(f"{player_name} not found in {self.name}'s roster")

    def add_player(self, player_name: str):
        """Add a player to the team's roster."""
        if player_name not in self.roster:
            self.roster.append(player_name)
            logger.debug(f"{player_name} added to {self.name}'s roster")
        else:
            logger.warning(f"{player_name} already in {self.name}'s roster")

    async def get_transfer_offers(self):
        """Get the list of proposed transfers for this team."""
        return self.proposed_transfers

    async def submit_transfer_offer(self, player_name: str, to_team: str, proposed_price: int):
        """Submit a transfer offer for a player."""
        offer = {
            'from_team': self.name,
            'to_team': to_team,
            'player_name': player_name,
            'proposed_price': proposed_price,
        }
        self.proposed_transfers.append(offer)
        logger.debug(f'Transfer offer submitted: {offer}')
        return offer

    async def react_to_others(self, other_actions: List[str]) -> str:
        reaction_prompt = f"""Other teams have taken the following actions:
        {' '.join(other_actions)}
        
        How do you want to react to these actions? Consider if you need to adjust your strategy or make counter-moves."""

        result = await self.executor.ainvoke(
            {
                'input': reaction_prompt,
                'agent_scratchpad': [],
            }
        )
        return result['output']

    async def decide_on_transfer(self, offer: Dict[str, Any]) -> Dict[str, Any]:
        logger.debug(f'{self.name}: Deciding on transfer offer: {offer}')

        decision = await self.executor.ainvoke(
            {
                'input': f"Transfer offer received:\nPlayer: {offer['player_name']}\nFrom: {offer['from_team']}\nTo: {self.name}\nPrice: ${offer['proposed_price']:,}\n\nMake a decision to accept, reject, or counter-offer. Respond with a dictionary containing 'action' (accept/reject/counter) and 'counter_offer' (if applicable).",
                'agent_scratchpad': [],
            }
        )

        logger.debug(f"{self.name}: Decision on transfer offer: {decision['output']}")
        return eval(
            decision['output']
        )  # Convert the string representation of the dictionary to an actual dictionary

    async def handle_tool_use(self, response):
        if 'Action:' in response and 'Action Input:' in response:
            action = response.split('Action:')[1].split('Action Input:')[0].strip()
            action_input = response.split('Action Input:')[1].strip()
            try:
                tool = next(t for t in self.tools if t.name.lower() == action.lower())
                if tool.name == 'execute_transfer':
                    # Parse the action_input for execute_transfer
                    inputs = eval(action_input)
                    result = await tool.ainvoke(**inputs)
                else:
                    result = await tool.ainvoke(input=action_input)
                return f'Tool execution result: {result}'
            except Exception as e:
                return f'Error executing tool {action}: {e}'
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'roster': self.roster,
            'budget': self.budget,
            'last_proposed_transfer': self.last_proposed_transfer,
        }


async def add_episode(event_description: str):
    """Add a new episode to the Graphiti client."""
    result = await graphiti_client.add_episode(
        name='New Event',
        episode_body=event_description,
        source_description='User Input',
        reference_time=datetime.now(),
        source=EpisodeType.message,
    )
    return f"Episode '{event_description}' added successfully."


async def invoke_tool(tool_name: str, **kwargs):
    tool = next(t for t in tools if t.name == tool_name)
    return await tool.ainvoke(input=kwargs)


def get_fact_string(edge):
    return f'{edge.fact} {edge.valid_at or edge.created_at}'


@tool
async def get_team_roster(team_name: str):
    """Get the current roster for a specific team."""
    search_result = await graphiti_client.search(f'plays for {team_name}', num_results=30)
    roster = [
        edge.fact.split(' plays for ')[0]
        for edge in search_result
        if 'plays for' in edge.fact.lower()
    ]
    return f"{team_name}'s roster: {', '.join(roster)}"


@tool
async def search_player_info(player_name: str):
    """Search for information about a specific player."""
    search_result = await graphiti_client.search(f'{player_name}', num_results=30)
    player_info = {
        'name': player_name,
        'facts': [get_fact_string(edge) for edge in search_result],
    }
    return player_info


@tool
async def propose_transfer(player_name: str, from_team: str, to_team: str, proposed_price: int):
    """Propose a player transfer from one team to another with a proposed price."""
    return f'TRANSFER PROPOSAL: {to_team} wants to buy {player_name} from {from_team} for ${proposed_price:,}.'


@tool
async def respond_to_transfer(
    player_name: str, from_team: str, to_team: str, response: str, counter_offer: int = None
):
    """Respond to a transfer proposal with an accept, reject, or counter-offer."""
    response_message = f'{from_team} {response}s the transfer of {player_name} to {to_team}'
    if counter_offer:
        response_message += f' with a counter-offer of ${counter_offer:,}'
    return f'Transfer response: {response_message}.'


@tool
async def execute_transfer(
    player_name: str, from_team: str, to_team: str, price: int
) -> Dict[str, Any]:
    """Execute a transfer between two teams."""
    # This is a simplified version. In a real scenario, you'd need to handle this more robustly.
    return {
        'messages': [
            HumanMessage(
                content=f'Transfer executed: {player_name} moved from {from_team} to {to_team} for ${price:,}'
            )
        ],
    }


@tool
async def check_team_budget(team_name: str) -> Dict[str, Any]:
    """Check the current budget of a team."""
    # This is a placeholder. In a real scenario, you'd fetch the actual budget.
    return {
        'messages': [HumanMessage(content=f"Checking {team_name}'s budget...")],
    }


@tool
async def submit_transfer_offer(
    player_name: str, from_team: str, to_team: str, proposed_price: int
) -> Dict[str, Any]:
    """Submit a transfer offer for a player."""
    logger.debug(
        f'submit_transfer_offer called with args: player_name={player_name}, from_team={from_team}, to_team={to_team}, proposed_price={proposed_price}'
    )
    offer = {
        'from_team': from_team,
        'to_team': to_team,
        'player_name': player_name,
        'proposed_price': proposed_price,
    }
    logger.debug(f'Transfer offer created: {offer}')
    return {
        'messages': [
            HumanMessage(
                content=f'Transfer offer submitted: {from_team} offers to sell {player_name} to {to_team} for ${proposed_price:,}.'
            )
        ],
        'transfer_offers': [offer],
    }


# Update the tools list
tools = [
    get_team_roster,
    search_player_info,
    propose_transfer,
    respond_to_transfer,
    execute_transfer,
    check_team_budget,
    submit_transfer_offer,
]


def process_event(state: SimulationState) -> SimulationState:
    logger.debug('Entering process_event')
    new_message = f"Event processed: {state['event']}"
    return {
        **state,
        'messages': state.get('messages', []) + [new_message],
    }


async def parallel_agent_processing(state: SimulationState) -> SimulationState:
    logger.debug('Entering parallel_agent_processing')
    tasks = []
    team_agents = {}
    for team_name, team_data in state['teams'].items():
        team_agent = TeamAgent(team_data['name'], tools)
        team_agents[team_name] = team_agent
        tasks.append(
            asyncio.create_task(team_agent.process_event(state['event'], state['transfer_offers']))
        )

    results = await asyncio.gather(*tasks, return_exceptions=True)

    updated_state = state.copy()
    updated_state['transfer_offers'] = []
    for i, (team_name, team_data) in enumerate(state['teams'].items()):
        if isinstance(results[i], Exception):
            logger.error(f'Error processing event for {team_name}: {str(results[i])}')
            updated_state['team_actions'][team_name] = f'Error: {str(results[i])}'
        else:
            logger.debug(f'Team {team_name} action: {results[i]}')
            updated_state['team_actions'][team_name] = results[i]
            if team_agents[team_name].last_proposed_transfer:
                updated_state['transfer_offers'].append(
                    team_agents[team_name].last_proposed_transfer
                )
        updated_state['teams'][team_name] = team_agents[team_name].to_dict()

    logger.debug(f'Updated state after parallel processing: {updated_state}')
    return updated_state


async def collect_transfer_offers(state: SimulationState) -> SimulationState:
    logger.debug('Entering collect_transfer_offers')
    updated_state = state.copy()
    logger.debug(f'Collected transfer offers: {updated_state}')
    # The transfer offers are already collected in parallel_agent_processing
    logger.debug(f"Collected transfer offers: {updated_state['transfer_offers']}")
    return updated_state


def select_negotiation(state: SimulationState) -> SimulationState:
    logger.debug('Entering select_negotiation')
    if not state['transfer_offers']:
        return {**state, 'current_negotiation': None, 'negotiation_complete': True}
    return {**state, 'current_negotiation': state['transfer_offers'][0]}


async def negotiate_transfer(state: SimulationState) -> SimulationState:
    logger.debug('Entering negotiate_transfer')
    updated_state = state.copy()

    if not updated_state['transfer_offers']:
        logger.debug('No transfer offers to negotiate')
        return updated_state

    # Sort offers by proposed price (highest first)
    sorted_offers = sorted(
        updated_state['transfer_offers'], key=lambda x: x['proposed_price'], reverse=True
    )
    best_offer = sorted_offers[0]

    # Simulate negotiation
    from_team = updated_state['teams'][best_offer['from_team']]
    to_team = updated_state['teams'][best_offer['to_team']]

    # Simple negotiation logic: accept if the price is above a threshold
    threshold = 50  # This can be adjusted
    if best_offer['proposed_price'] > threshold:
        logger.info(
            f"Transfer accepted: {best_offer['player_name']} from {best_offer['from_team']} to {best_offer['to_team']} for ${best_offer['proposed_price']}"
        )
        # Update team rosters and budgets
        from_team['roster'].remove(best_offer['player_name'])
        to_team['roster'].append(best_offer['player_name'])
        from_team['budget'] += best_offer['proposed_price']
        to_team['budget'] -= best_offer['proposed_price']
        updated_state['negotiation_complete'] = True
    else:
        logger.info(
            f"Transfer rejected: {best_offer['player_name']} from {best_offer['from_team']} to {best_offer['to_team']} for ${best_offer['proposed_price']}"
        )

    updated_state['current_negotiation'] = None
    updated_state['transfer_offers'] = []
    return updated_state


async def execute_transfer(state: SimulationState, offer: Dict[str, Any]) -> None:
    from_agent = TeamAgent(
        state['teams'][offer['from_team']]['name'], tools
    )  # Recreate TeamAgent from data
    to_agent = TeamAgent(
        state['teams'][offer['to_team']]['name'], tools
    )  # Recreate TeamAgent from data

    from_agent.roster.remove(offer['player_name'])
    to_agent.roster.append(offer['player_name'])
    from_agent.budget += offer['proposed_price']
    to_agent.budget -= offer['proposed_price']

    transfer_message = f"{offer['player_name']} transferred from {offer['from_team']} to {offer['to_team']} for ${offer['proposed_price']:,}"
    state['messages'].append(transfer_message)


def should_continue(state: SimulationState) -> List[str]:
    if state['negotiation_complete'] and not state['transfer_offers']:
        return [END]
    return ['select_negotiation']


# Define the graph
workflow = StateGraph(SimulationState)

# Add nodes
workflow.add_node('process_event', process_event)
workflow.add_node('parallel_agent_processing', parallel_agent_processing)
workflow.add_node('collect_transfer_offers', collect_transfer_offers)
workflow.add_node('select_negotiation', select_negotiation)
workflow.add_node('negotiate_transfer', negotiate_transfer)

# Add edges
workflow.add_edge('process_event', 'parallel_agent_processing')
workflow.add_edge('parallel_agent_processing', 'collect_transfer_offers')
workflow.add_edge('collect_transfer_offers', 'select_negotiation')
workflow.add_edge('select_negotiation', 'negotiate_transfer')

# Add conditional edge
workflow.add_conditional_edges(
    'negotiate_transfer', should_continue, {'select_negotiation': 'select_negotiation', END: END}
)

# Set the entrypoint
workflow.set_entry_point('process_event')

# Compile the graph
app = workflow.compile()


async def run_simulation():
    while True:
        event = input("Enter an event (or 'quit' to exit): ")
        if event.lower() == 'quit':
            break

        initial_state = SimulationState(
            messages=[],
            teams={
                'Toronto Raptors': TeamAgent('Toronto Raptors', tools).to_dict(),
                'Boston Celtics': TeamAgent('Boston Celtics', tools).to_dict(),
                'Golden State Warriors': TeamAgent('Golden State Warriors', tools).to_dict(),
            },
            event=event,
            team_actions={},
            transfer_offers=[],
            current_negotiation=None,
            negotiation_rounds=0,
            negotiation_complete=False,
        )

        async for state in app.astream(initial_state):
            if 'messages' in state:
                for message in state['messages']:
                    print(message)

            if 'transfer_offers' in state:
                print(f"Current transfer offers: {state['transfer_offers']}")

            if 'current_negotiation' in state:
                print(f"Current negotiation: {state['current_negotiation']}")

        print('\nFinal team states:')
        for team_name, team_data in initial_state['teams'].items():
            print(f"{team_name} - Roster: {team_data['roster']}, Budget: ${team_data['budget']:,}")

        print('\n' + '=' * 50 + '\n')


if __name__ == '__main__':
    asyncio.run(run_simulation())
