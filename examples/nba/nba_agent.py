import asyncio
import logging
import os
from datetime import datetime
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict, Union

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

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


class State(TypedDict):
    messages: List[HumanMessage]
    teams: Dict[str, Any]
    event: str
    team_actions: Dict[str, str]
    transfer_offers: List[Dict[str, Any]]
    current_negotiation: Optional[Dict[str, Any]]
    negotiation_rounds: int
    negotiation_complete: bool


def format_step_result(messages: List[str], **kwargs) -> Dict[str, Any]:
    return {'messages': messages, **kwargs}


class TeamAgent:
    def __init__(
        self, team_name: str, tools: List[Any], budget: int = 100000000, roster: List[str] = []
    ):
        # logger.debug(f'Initializing TeamAgent for {team_name}')
        self.team_name = team_name
        self.tools = tools
        self.budget = budget
        self.roster = []
        self.proposed_transfers = []
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    'system',
                    f"""You are an AI assistant managing the {team_name} NBA team. 
            Your role is to make strategic decisions for your team, react to events, and interact with other teams.
            Use the available tools to gather information and perform actions.
            When an event occurs, decide how to react. You can:
            1. Use tools to gather more information about players or team situations.
            2. Propose player transfers (buy or sell) based on events and your team's needs.
            3. Set transfer prices based on player performance and your available budget.
            4. Negotiate with other teams on transfer prices.
            Always consider what's best for your team in the long term. Be strategic and competitive.
            The only teams in this simulation are: Toronto Raptors, Boston Celtics, and Golden State Warriors.
            Only interact with these teams and do not mention or propose transfers to any other teams.""",
                ),
                ('human', '{input}'),
                MessagesPlaceholder(variable_name='agent_scratchpad'),
            ]
        )
        self.llm = ChatOpenAI(temperature=0.2)
        self.agent = create_openai_functions_agent(self.llm, self.tools, self.prompt)
        self.executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)

    async def update_budget(self, amount: int):
        """Update the team's budget."""
        self.budget += amount
        logger.debug(f"{self.team_name}'s new budget: ${self.budget:,}")
        return f"{self.team_name}'s new budget: ${self.budget:,}"

    async def propose_transfers(self) -> List[Dict[str, Any]]:
        """Propose transfer offers based on the agent's strategy."""
        # This is a placeholder implementation. In a real scenario, this would involve more complex logic.
        if hasattr(self, 'last_proposed_transfer'):
            return [self.last_proposed_transfer]
        return []

    async def update_roster(self):
        """Update the team's roster using the get_team_roster tool."""
        roster_tool = next(tool for tool in self.tools if tool.name == 'get_team_roster')
        roster_string = await roster_tool.ainvoke(self.team_name)
        self.roster = roster_string.split(': ')[1].split(', ')

    def remove_player(self, player_name: str):
        """Remove a player from the team's roster."""
        if player_name in self.roster:
            self.roster.remove(player_name)
            logger.debug(f"{player_name} removed from {self.team_name}'s roster")
        else:
            logger.warning(f"{player_name} not found in {self.team_name}'s roster")

    def add_player(self, player_name: str):
        """Add a player to the team's roster."""
        if player_name not in self.roster:
            self.roster.append(player_name)
            logger.debug(f"{player_name} added to {self.team_name}'s roster")
        else:
            logger.warning(f"{player_name} already in {self.team_name}'s roster")

    async def get_transfer_offers(self):
        """Get the list of proposed transfers for this team."""
        return self.proposed_transfers

    async def submit_transfer_offer(self, player_name: str, to_team: str, proposed_price: int):
        """Submit a transfer offer for a player."""
        offer = {
            'from_team': self.team_name,
            'to_team': to_team,
            'player_name': player_name,
            'proposed_price': proposed_price,
        }
        self.proposed_transfers.append(offer)
        logger.debug(f'Transfer offer submitted: {offer}')
        return offer

    async def process_event(
        self, state: Dict[str, Any], event: str, relevant_offers: List[Dict[str, Any]] = []
    ):
        logger.debug(f'{self.team_name}: Processing event: {event}')
        logger.debug(f'{self.team_name}: Current roster: {self.roster}')
        logger.debug(f'{self.team_name}: Current budget: {self.budget}')

        try:
            result = await self.executor.ainvoke(
                {
                    'input': f'Event: {event}\n\nCurrent team: {self.team_name}\nCurrent budget: ${self.budget:,}\nCurrent roster: {", ".join(self.roster)}\nRelevant transfer offers: {relevant_offers}\n\nReact to this event and consider the transfer offers. Make decisions and take actions without asking for confirmation.',
                    'agent_scratchpad': [],
                }
            )
            logger.debug(f"{self.team_name}: Finished processing event. Result: {result['output']}")

            # Check if propose_transfer was called
            if 'propose_transfer' in result['output']:
                # Extract transfer details
                transfer_details = result['output'].split('propose_transfer(')[1].split(')')[0]
                transfer_dict = eval(f'dict({transfer_details})')
                self.last_proposed_transfer = transfer_dict
                logger.debug(
                    f'{self.team_name}: Set last_proposed_transfer to {self.last_proposed_transfer}'
                )

            return result['output']
        except Exception as e:
            logger.error(f'{self.team_name}: Error processing event: {str(e)}', exc_info=True)
            raise

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
        logger.debug(f'{self.team_name}: Deciding on transfer offer: {offer}')

        decision = await self.executor.ainvoke(
            {
                'input': f"Transfer offer received:\nPlayer: {offer['player_name']}\nFrom: {offer['from_team']}\nTo: {self.team_name}\nPrice: ${offer['proposed_price']:,}\n\nMake a decision to accept, reject, or counter-offer. Respond with a dictionary containing 'action' (accept/reject/counter) and 'counter_offer' (if applicable).",
                'agent_scratchpad': [],
            }
        )

        logger.debug(f"{self.team_name}: Decision on transfer offer: {decision['output']}")
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
    return f'Transfer proposal: {to_team} wants to buy {player_name} from {from_team} for ${proposed_price:,}.'


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


def root(state: dict) -> dict:
    logger.debug(f'Entering root node. State: {state}')
    return {
        **state,
        'messages': state['messages']
        + [HumanMessage(content=f"Processing event: {state['event']}")],
    }


async def add_episode_node(state: dict) -> dict:
    logger.debug(f'Entering add_episode_node. State: {state}')
    logger.debug(f"Adding episode: {state['event']}")
    result = await add_episode(state['event'])
    logger.debug(f'Episode added successfully: {result}')
    return {**state, 'messages': state['messages'] + [HumanMessage(content=result)]}


async def process_event(state: dict) -> dict:
    logger.debug('Entering process_event')
    return {
        **state,
        'messages': state['messages']
        + [HumanMessage(content=f"Event processed: {state['event']}")],
    }


async def parallel_agent_processing(state: Dict[str, Any]) -> Dict[str, Any]:
    logger.debug('Entering parallel_agent_processing')
    logger.debug(f'Current state: {state}')
    tasks = []
    for team_name, agent in state['teams'].items():
        logger.debug(f"{team_name}: Starting to process event: {state['event']}")
        relevant_offers = [
            offer for offer in state.get('transfer_offers', []) if offer['to_team'] == team_name
        ]
        logger.debug(f'{team_name}: Relevant offers: {relevant_offers}')
        tasks.append(agent.process_event(state, state['event'], relevant_offers))
    results = await asyncio.gather(*tasks)
    logger.debug(f'Parallel processing results: {results}')

    # Merge transfer offers from all agents
    all_transfer_offers = state.get('transfer_offers', [])
    for agent in state['teams'].values():
        if hasattr(agent, 'last_proposed_transfer'):
            all_transfer_offers.append(agent.last_proposed_transfer)
            logger.debug(f'Added transfer offer: {agent.last_proposed_transfer}')

    updated_state = {
        **state,
        'team_actions': {team: action for team, action in zip(state['teams'].keys(), results)},
        'transfer_offers': all_transfer_offers,
    }
    logger.debug(f'Updated state after parallel processing: {updated_state}')
    return updated_state


async def agent_reaction_phase(state: dict) -> dict:
    logger.debug('Entering agent_reaction_phase')
    messages = []
    for team_name, action in state['team_actions'].items():
        messages.append(f'{team_name}: {action}')
    return {
        **state,
        'messages': state['messages'] + [HumanMessage(content=msg) for msg in messages],
    }


async def collect_transfer_offers(state: State) -> State:
    logger.debug('Entering collect_transfer_offers')
    transfer_offers = []
    for team_name, agent in state['teams'].items():
        offers = await agent.propose_transfers()
        transfer_offers.extend(offers)
    return {**state, 'transfer_offers': state['transfer_offers'] + transfer_offers}


async def select_negotiation(state: Dict[str, Any]) -> Dict[str, Any]:
    logger.debug('Entering select_negotiation')
    logger.debug(f'Current state in select_negotiation: {state}')
    if not state.get('transfer_offers', []):
        return {
            **state,
            'messages': state['messages']
            + [HumanMessage(content='No transfer offers to negotiate.')],
            'current_negotiation': None,
        }

    selected_offer = state['transfer_offers'][0]
    return {
        **state,
        'messages': state['messages']
        + [
            HumanMessage(
                content=f"Selected negotiation: {selected_offer['player_name']} from {selected_offer['from_team']} to {selected_offer['to_team']}"
            )
        ],
        'current_negotiation': selected_offer,
    }


async def negotiate_transfer(state: dict) -> dict:
    logger.debug('Entering negotiate_transfer')
    if not state['transfer_offers']:
        logger.debug('No transfer offers to negotiate')
        return {
            **state,
            'messages': state['messages']
            + [HumanMessage(content='No transfer offers to negotiate.')],
            'negotiation_complete': True,
        }

    offer = state['transfer_offers'][0]  # Take the first offer to negotiate
    from_agent = state['teams'][offer['from_team']]
    to_agent = state['teams'][offer['to_team']]

    logger.debug(f'Negotiating transfer: {offer}')

    # Ask the receiving team to make a decision
    decision = await to_agent.decide_on_transfer(offer)
    logger.debug(f'Decision from {to_agent.team_name}: {decision}')

    if decision['action'] == 'accept':
        # Execute the transfer
        execute_transfer_tool = next(tool for tool in tools if tool.name == 'execute_transfer')
        transfer_result = await execute_transfer_tool.ainvoke(
            player_name=offer['player_name'],
            from_team=offer['from_team'],
            to_team=offer['to_team'],
            price=offer['proposed_price'],
        )

        # Update team rosters and budgets
        from_agent.remove_player(offer['player_name'])
        to_agent.add_player(offer['player_name'])
        await from_agent.update_budget(offer['proposed_price'])
        await to_agent.update_budget(-offer['proposed_price'])

        transfer_message = f"{offer['player_name']} transferred from {offer['from_team']} to {offer['to_team']} for ${offer['proposed_price']:,}"
        logger.debug(transfer_message)

        return {
            **state,
            'messages': state['messages'] + [HumanMessage(content=transfer_message)],
            'transfer_offers': state['transfer_offers'][1:],  # Remove the processed offer
            'negotiation_complete': True,
        }
    elif decision['action'] == 'reject':
        reject_message = f"{offer['to_team']} rejected the offer for {offer['player_name']}."
        logger.debug(reject_message)
        return {
            **state,
            'messages': state['messages'] + [HumanMessage(content=reject_message)],
            'transfer_offers': state['transfer_offers'][1:],  # Remove the rejected offer
            'negotiation_complete': True,
        }
    elif decision['action'] == 'counter':
        counter_offer = decision['counter_offer']
        counter_message = (
            f"{offer['to_team']} counter-offered for {offer['player_name']} at ${counter_offer:,}."
        )
        logger.debug(counter_message)
        # Update the offer with the new price
        updated_offer = {**offer, 'proposed_price': counter_offer}
        return {
            **state,
            'messages': state['messages'] + [HumanMessage(content=counter_message)],
            'transfer_offers': [updated_offer] + state['transfer_offers'][1:],
            'negotiation_complete': False,  # Continue negotiation
        }

    return state


async def handle_tool_use(state: Dict[str, Any]) -> Dict[str, Any]:
    logger.debug('Entering handle_tool_use')
    logger.debug(f'Current state in handle_tool_use: {state}')

    executed_transfers = []
    for team_name, agent in state['teams'].items():
        logger.debug(f'Checking for last_proposed_transfer on {team_name}')
        if hasattr(agent, 'last_proposed_transfer'):
            transfer = agent.last_proposed_transfer
            logger.debug(f'Processing transfer: {transfer}')

            from_agent = state['teams'][transfer['from_team']]
            to_agent = state['teams'][transfer['to_team']]

            # Execute the transfer
            execute_transfer_tool = next(tool for tool in tools if tool.name == 'execute_transfer')
            transfer_result = await execute_transfer_tool.ainvoke(
                player_name=transfer['player_name'],
                from_team=transfer['from_team'],
                to_team=transfer['to_team'],
                price=transfer['proposed_price'],
            )

            logger.debug(f'Transfer result: {transfer_result}')

            # Update team rosters and budgets
            logger.debug(f'Before transfer - {from_agent.team_name} roster: {from_agent.roster}')
            logger.debug(f'Before transfer - {to_agent.team_name} roster: {to_agent.roster}')

            from_agent.remove_player(transfer['player_name'])
            to_agent.add_player(transfer['player_name'])

            logger.debug(f'After transfer - {from_agent.team_name} roster: {from_agent.roster}')
            logger.debug(f'After transfer - {to_agent.team_name} roster: {to_agent.roster}')

            await from_agent.update_budget(transfer['proposed_price'])
            await to_agent.update_budget(-transfer['proposed_price'])

            logger.debug(f'Updated {from_agent.team_name} budget: {from_agent.budget}')
            logger.debug(f'Updated {to_agent.team_name} budget: {to_agent.budget}')

            # Add transfer result to messages
            state['messages'].append(HumanMessage(content=transfer_result['messages'][0].content))
            executed_transfers.append(transfer)

            # Clear the last proposed transfer
            delattr(agent, 'last_proposed_transfer')
        else:
            logger.debug(f'No last_proposed_transfer found for {team_name}')

    # Add executed transfers to the state
    state['executed_transfers'] = executed_transfers
    logger.debug(f'Executed transfers: {executed_transfers}')

    return state


def should_continue(x: dict):
    if x.get('negotiation_complete', False):
        return END
    if len(x.get('transfer_offers', [])) == 0 and x.get('current_negotiation') is None:
        return END
    return 'collect_transfer_offers'


# Define the workflow
workflow = StateGraph(State)

workflow.add_node('root', root)
workflow.add_node('add_episode', add_episode_node)
workflow.add_node('process_event', process_event)
workflow.add_node('parallel_agent_processing', parallel_agent_processing)
workflow.add_node('agent_reaction_phase', agent_reaction_phase)
workflow.add_node('collect_transfer_offers', collect_transfer_offers)
workflow.add_node('select_negotiation', select_negotiation)
workflow.add_node('negotiate_transfer', negotiate_transfer)
workflow.add_node('handle_tool_use', handle_tool_use)

workflow.set_entry_point('root')

workflow.add_edge('root', 'add_episode')
workflow.add_edge('add_episode', 'process_event')
workflow.add_edge('process_event', 'parallel_agent_processing')
workflow.add_edge('parallel_agent_processing', 'agent_reaction_phase')
workflow.add_edge('agent_reaction_phase', 'collect_transfer_offers')
workflow.add_edge('collect_transfer_offers', 'select_negotiation')
workflow.add_edge('select_negotiation', 'negotiate_transfer')
workflow.add_edge('negotiate_transfer', 'handle_tool_use')

workflow.add_conditional_edges(
    'handle_tool_use',
    should_continue,
    {END: END, 'collect_transfer_offers': 'collect_transfer_offers'},
)

app = workflow.compile()

teams = {
    'Toronto Raptors': TeamAgent('Toronto Raptors', tools),
    'Boston Celtics': TeamAgent('Boston Celtics', tools),
    'Golden State Warriors': TeamAgent('Golden State Warriors', tools),
}

# Update the initial_state
initial_state: State = {
    'messages': [],
    'teams': teams,
    'event': '',
    'team_actions': {},
    'transfer_offers': [],
    'current_negotiation': None,
    'negotiation_rounds': 0,
    'negotiation_complete': False,
}


async def run_simulation():
    state = {
        'messages': [],
        'teams': teams,
        'event': None,
        'team_actions': {},
        'transfer_offers': [],
        'current_negotiation': None,
        'negotiation_rounds': 0,
        'negotiation_complete': False,
    }

    while True:
        event = input("Enter an event (or 'quit' to exit): ")
        if event.lower() == 'quit':
            break

        state['event'] = event
        state['messages'] = []
        state['team_actions'] = {}
        state['transfer_offers'] = []
        state['current_negotiation'] = None
        state['negotiation_rounds'] = 0
        state['negotiation_complete'] = False

        # Process the event
        state = await process_event(state)

        # Parallel agent processing
        state = await parallel_agent_processing(state)

        # Collect transfer offers
        state = await collect_transfer_offers(state)

        # Negotiate transfers
        while (
            not state['negotiation_complete']
            and state['negotiation_rounds'] < MAX_NEGOTIATION_ROUNDS
        ):
            state = await negotiate_transfer(state)
            state['negotiation_rounds'] += 1

        # Handle tool use (execute transfers, etc.)
        state = await handle_tool_use(state)

        # Display results
        for message in state['messages']:
            print(message.content)

        print('\nFinal team states:')
        for team_name, team_agent in state['teams'].items():
            print(f'{team_name} - Roster: {team_agent.roster}, Budget: ${team_agent.budget:,}')

    print('Simulation ended.')


def main():
    asyncio.run(run_simulation())


if __name__ == '__main__':
    main()
