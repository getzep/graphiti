import asyncio
import logging
import os
from datetime import datetime
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolInvocation, ToolNode

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger('nba_agent')
for name in logging.root.manager.loggerDict:
    if name != 'nba_agent':
        logging.getLogger(name).setLevel(logging.WARNING)

# Initialize Graphiti client
neo4j_uri = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
neo4j_user = os.environ.get('NEO4J_USER', 'neo4j')
neo4j_password = os.environ.get('NEO4J_PASSWORD', 'password')
graphiti_client = Graphiti(neo4j_uri, neo4j_user, neo4j_password)


class TeamAgent:
    def __init__(self, team_name: str, tools: List[Any], budget: int = 100000000):
        self.team_name = team_name
        self.tools = tools
        self.budget = budget
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
            Always consider what's best for your team in the long term. Be strategic and competitive.""",
                ),
                ('human', '{input}'),
                MessagesPlaceholder(variable_name='agent_scratchpad'),
            ]
        )
        self.llm = ChatOpenAI(temperature=0.2)
        self.agent = create_openai_functions_agent(self.llm, self.tools, self.prompt)
        self.executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)

    async def update_budget(self, amount: int):
        self.budget += amount
        return f"{self.team_name}'s new budget: ${self.budget:,}"

    async def process_event(self, event: str):
        result = await self.executor.ainvoke(
            {
                'input': f'Event: {event}\n\nCurrent team: {self.team_name}\nCurrent budget: ${self.budget:,}\n\nReact to this event. If there are transfer proposals, consider them and respond appropriately. Make decisions and take actions without asking for confirmation. Ensure transfer prices are realistic (in millions of dollars).',
                'agent_scratchpad': [],
            }
        )
        return result['output']

    async def handle_tool_use(self, response):
        if 'Action:' in response and 'Action Input:' in response:
            action = response.split('Action:')[1].split('Action Input:')[0].strip()
            action_input = response.split('Action Input:')[1].strip()
            try:
                tool = next(t for t in self.tools if t.name.lower() == action.lower())
                result = await tool.ainvoke(**eval(action_input))
                return f'Tool execution result: {result}'
            except Exception as e:
                return f'Error executing tool {action}: {e}'
        return None


class AgentManager:
    def __init__(self):
        self.agents = {}

    def add_agent(self, team_name: str, budget: int):
        if team_name not in self.agents:
            self.agents[team_name] = TeamAgent(team_name, tools, budget)

    async def process_event(self, event: str):
        responses = []
        for team_name, agent in self.agents.items():
            response = await agent.process_event(event)
            responses.append(f'{team_name}: {response}')
        return responses


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
async def execute_transfer(player_name: str, from_team: str, to_team: str, final_price: int):
    """Execute a player transfer from one team to another with the final agreed price."""
    from_agent = agent_manager.agents.get(from_team)
    to_agent = agent_manager.agents.get(to_team)

    if not from_agent or not to_agent:
        return 'One or both teams not found.'

    if to_agent.budget < final_price:
        return f"{to_team} doesn't have enough budget for this transfer."

    # Update budgets
    await from_agent.update_budget(final_price)
    await to_agent.update_budget(-final_price)

    # Add the transfer as an episode
    await add_episode(
        event_description=f'{player_name} transferred from {from_team} to {to_team} for ${final_price:,}'
    )
    return f'Transfer executed: {player_name} has been transferred from {from_team} to {to_team} for ${final_price:,}.'


@tool
async def check_team_budget(team_name: str):
    """Check the current budget of a team."""
    agent = agent_manager.agents.get(team_name)
    if agent:
        return f"{team_name}'s current budget: ${agent.budget:,}"
    return f'Team {team_name} not found.'


# Update the tools list
tools = [
    get_team_roster,
    search_player_info,
    propose_transfer,
    respond_to_transfer,
    execute_transfer,
    check_team_budget,
]

agent_manager = AgentManager()

# Add your teams here
agent_manager.add_agent('Toronto Raptors', budget=100000000)
agent_manager.add_agent('Boston Celtics', budget=100000000)
agent_manager.add_agent('Golden State Warriors', budget=100000000)


# Main loop
async def main():
    print('Welcome to the NBA Team Management Simulation!')
    print('Enter events, and watch how the teams react.')
    print("Type 'quit' to exit the simulation.\n")

    while True:
        user_input = input("Enter an event (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break

        try:
            # Add the event as an episode
            result = await add_episode(event_description=user_input)
            print(result)
        except Exception as e:
            print(f'Error adding episode: {e}')

        # Process the event with all team agents for multiple rounds
        transfer_proposals = []
        for round in range(3):  # You can adjust the number of rounds as needed
            print(f'\nRound {round + 1}:')
            for team_name, agent in agent_manager.agents.items():
                print(f'\n{team_name} reaction:')
                try:
                    response = await agent.process_event(user_input)
                    print(response)

                    # Handle tool use
                    tool_result = await agent.handle_tool_use(response)
                    if tool_result:
                        print(tool_result)

                        # If a transfer was proposed or responded to, add it to the list
                        if (
                            'Transfer proposal:' in tool_result
                            or 'Transfer response:' in tool_result
                        ):
                            transfer_proposals.append(tool_result)

                except Exception as e:
                    print(f'Error processing event for {team_name}: {e}')

            # After each round, update the user_input to include transfer proposals
            if transfer_proposals:
                user_input = (
                    f'Previous event: {user_input}\nTransfer proposals and responses:\n'
                    + '\n'.join(transfer_proposals)
                )
            else:
                break  # If no new proposals or responses, end the rounds

        print('\n')


if __name__ == '__main__':
    asyncio.run(main())
