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


async def invoke_tool(tool_name: str, **kwargs):
    tool = next(t for t in tools if t.name == tool_name)
    return await tool.ainvoke(input=kwargs)


@tool
async def get_team_roster(team_name: str):
    """Get the current roster for a specific team."""
    search_result = await graphiti_client.search(f'{team_name.lower()}', num_results=1)
    print(search_result)
    print(team_name.lower())
    roster = []
    for fact in search_result:
        roster.append(fact)
    return roster


@tool
async def search_player_info(player_name: str):
    """Search for information about a specific player."""
    search_result = await graphiti_client.search(f'{player_name}')
    player_info = {
        'name': player_name,
        'facts': search_result,
    }
    return player_info


@tool
async def verify_transfer_conditions(player_name: str, from_team: str, to_team: str):
    """Verify conditions for a player transfer."""
    from_roster = await invoke_tool('get_team_roster', team_name=from_team)
    to_roster = await invoke_tool('get_team_roster', team_name=to_team)
    player_info = await invoke_tool('search_player_info', player_name=player_name)

    # Prepare context for LLM
    context = f"""
    Player: {player_name}
    From Team: {from_team}
    To Team: {to_team}

    From Team Roster:
    {from_roster}

    To Team Roster:
    {to_roster}

    Player Info:
    {player_info}
    """

    # Use LLM to evaluate transfer conditions
    llm = ChatOpenAI(temperature=0)
    prompt = f"""
    Based on the following information, determine if the transfer conditions are met for {player_name} to move from {from_team} to {to_team}.

    Context:
    {context}

    Please consider the following conditions:
    1. Is {from_team} a valid NBA team?
    2. Is {to_team} a valid NBA team?
    3. Is {player_name} currently on the roster of {from_team}?
    4. Is there enough information about {player_name}?

    Important: Players can be transferred multiple times, including back to teams they've played for before.

    Provide a detailed analysis of each condition and conclude whether all conditions are met or not.
    Your response should end with one of these two statements:
    - TRANSFER APPROVED: All conditions are met.
    - TRANSFER DENIED: [Reason for denial]
    """

    response = await llm.ainvoke(prompt)

    return response.content


@tool
async def transfer_player(player_name: str, from_team: str, to_team: str):
    """Transfer a player from one team to another."""
    try:
        # Verify transfer conditions
        verification_result = await invoke_tool(
            'verify_transfer_conditions',
            player_name=player_name,
            from_team=from_team,
            to_team=to_team,
        )

        # Check if transfer is approved
        if 'TRANSFER APPROVED' in verification_result:
            logger.info(f'Transfer initiated: {player_name} from {from_team} to {to_team}')

            # Add episode
            await graphiti_client.add_episode(
                name=f'Transfer {player_name}',
                episode_body=f'{player_name} transferred from {from_team} to {to_team}',
                source_description='Player Transfer',
                reference_time=datetime.now(),
                source=EpisodeType.message,
            )

            return f'Player {player_name} has been successfully transferred from {from_team} to {to_team}.'
        else:
            return f'Transfer denied: {verification_result}'
    except Exception as e:
        logger.error(f'Error in transfer_player: {str(e)}')
        return 'An error occurred while transferring the player. Please try again later or contact support.'


# Main agent setup
tools = [get_team_roster, search_player_info, verify_transfer_conditions, transfer_player]

prompt = ChatPromptTemplate.from_messages(
    [
        (
            'system',
            """You are an AI assistant for NBA team management. Your role is to help users manage team rosters, transfer players, and provide information about players and teams. Use the available tools to gather information and perform actions.

    When transferring players, always verify the following before proceeding:
    1. The player is currently on the roster of the 'from' team.
    2. Both the 'from' and 'to' teams are valid NBA teams.
    
    Use the get_team_roster and search_player_info tools to verify this information. Only proceed with the transfer if all conditions are met.

    IMPORTANT: Only use the information retrieved from the tools. Do not make assumptions or use information that hasn't been explicitly provided by the tools or the user. If you're unsure about any information, use the appropriate tool to verify it.""",
        ),
        MessagesPlaceholder(variable_name='chat_history'),
        ('human', '{input}'),
        MessagesPlaceholder(variable_name='agent_scratchpad'),
    ]
)

llm = ChatOpenAI(temperature=0)
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True)

# Graph setup
workflow = StateGraph(MessagesState)


async def agent_node(state):
    user_input = state['messages'][-1].content if state['messages'] else ''
    chat_history = state['messages'][:-1]  # All messages except the last one
    result = await agent_executor.ainvoke({'input': user_input, 'chat_history': chat_history})
    return {
        'messages': state['messages'] + [AIMessage(content=result['output'], name='Manager')],
        'agent_scratchpad': [],
    }


workflow.add_node('agent', agent_node)
workflow.set_entry_point('agent')
workflow.add_edge('agent', '__end__')

app = workflow.compile()


# Run function
async def run_workflow(input_text: str, chat_history: List[Dict[str, Any]] = []):
    result = await app.ainvoke(
        {
            'messages': [
                *[
                    HumanMessage(content=msg['content'])
                    if msg['type'] == 'human'
                    else AIMessage(content=msg['content'])
                    for msg in chat_history
                ],
                HumanMessage(content=input_text),
            ],
        },
    )

    # Log only the latest human input and AI response
    logger.info(f'Human: {input_text}\n')

    latest_ai_message = next(
        (message for message in reversed(result['messages']) if isinstance(message, AIMessage)),
        None,
    )
    if latest_ai_message:
        logger.info(f'AI: {latest_ai_message.content}\n')

    return result['messages']


# Main loop
async def main():
    chat_history = []
    while True:
        user_input = input("Enter your request (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break
        messages = await run_workflow(user_input, chat_history)
        # Update chat history with only the latest human input and AI response
        chat_history = [
            {'type': 'human', 'content': user_input},
            {
                'type': 'ai',
                'content': messages[-1].content
                if isinstance(messages[-1], AIMessage)
                else messages[-2].content,
            },
        ]


if __name__ == '__main__':
    import asyncio

    asyncio.run(main())
