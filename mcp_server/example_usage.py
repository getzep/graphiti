#!/usr/bin/env python3
"""
Example script demonstrating how to use the Graphiti MCP server with Claude.
This script simulates a conversation with Claude where Graphiti functionality is used.
It showcases the various tools available in the Graphiti MCP server, including the
new search_facts and search_nodes tools.
"""

import json
import uuid

# Sample conversation with Claude using Graphiti MCP server
conversation = [
    {
        'role': 'user',
        'content': 'I want to add some information about Kamala Harris to the knowledge graph.',
    },
    {
        'role': 'assistant',
        'content': "I'd be happy to help you add information about Kamala Harris to the knowledge graph. What specific information would you like to add?",
    },
    {
        'role': 'user',
        'content': 'Add that Kamala Harris was the Attorney General of California from January 3, 2011 to January 3, 2017.',
    },
    {
        'role': 'assistant',
        'content': "I'll add that information about Kamala Harris to the knowledge graph for you.",
    },
    {
        'role': 'assistant',
        'tool_calls': [
            {
                'name': 'add_episode',
                'parameters': {
                    'name': 'Kamala Harris Career',
                    'episode_body': 'Kamala Harris was the Attorney General of California. She was in office from January 3, 2011 to January 3, 2017.',
                    'source': 'text',
                    'source_description': 'user provided information',
                    'group_id': 'politics_figures',
                    'uuid': str(uuid.uuid4()),
                },
            }
        ],
    },
    {
        'role': 'assistant',
        'content': "I've added the information about Kamala Harris being the Attorney General of California from January 3, 2011 to January 3, 2017 to the knowledge graph. The information has been stored with the group ID 'politics_figures'. Is there anything else you'd like to add or would you like to search for this information?",
    },
    {
        'role': 'user',
        'content': 'Now search for information about Kamala Harris in the knowledge graph.',
    },
    {
        'role': 'assistant',
        'content': "I'll search for information about Kamala Harris in the knowledge graph.",
    },
    {
        'role': 'assistant',
        'tool_calls': [
            {
                'name': 'search_facts',
                'parameters': {
                    'query': 'Kamala Harris',
                    'group_ids': ['politics_figures'],
                    'max_facts': 5,
                },
            }
        ],
    },
    {
        'role': 'assistant',
        'content': 'I found the following information about Kamala Harris in the knowledge graph:\n\n- Kamala Harris was the Attorney General of California from January 3, 2011 to January 3, 2017.\n\nThis fact was extracted from the episode you provided earlier. The knowledge graph has stored this information with temporal awareness, noting both when she began and ended her term as Attorney General.',
    },
    {
        'role': 'user',
        'content': 'Add that before being Attorney General, she was the District Attorney of San Francisco.',
    },
    {
        'role': 'assistant',
        'content': "I'll add that information about Kamala Harris's role as District Attorney of San Francisco before becoming Attorney General.",
    },
    {
        'role': 'assistant',
        'tool_calls': [
            {
                'name': 'add_episode',
                'parameters': {
                    'name': 'Kamala Harris San Francisco DA',
                    'episode_body': 'Before becoming Attorney General of California, Kamala Harris was the District Attorney of San Francisco from 2004 to 2011.',
                    'source': 'text',
                    'source_description': 'user provided information',
                    'group_id': 'politics_figures',
                    'uuid': str(uuid.uuid4()),
                },
            }
        ],
    },
    {
        'role': 'assistant',
        'content': "I've added the information that Kamala Harris was the District Attorney of San Francisco from 2004 to 2011 before becoming Attorney General of California. This information has been added to the same group 'politics_figures' in the knowledge graph. Would you like to search again to see both pieces of information?",
    },
    {'role': 'user', 'content': 'Yes, search for all information about Kamala Harris now.'},
    {
        'role': 'assistant',
        'content': "I'll search for all information about Kamala Harris in the knowledge graph.",
    },
    {
        'role': 'assistant',
        'tool_calls': [
            {
                'name': 'search_facts',
                'parameters': {
                    'query': 'Kamala Harris',
                    'group_ids': ['politics_figures'],
                    'max_facts': 10,
                },
            }
        ],
    },
    {
        'role': 'assistant',
        'content': "I found the following information about Kamala Harris in the knowledge graph:\n\n1. Kamala Harris was the Attorney General of California from January 3, 2011 to January 3, 2017.\n\n2. Before becoming Attorney General of California, Kamala Harris was the District Attorney of San Francisco from 2004 to 2011.\n\nThe knowledge graph has successfully stored both pieces of information with their temporal relationships preserved. This demonstrates how Graphiti maintains the chronology of events and can represent a person's career progression over time.",
    },
    {
        'role': 'user',
        'content': 'Can you also show me a summary of entities related to Kamala Harris?',
    },
    {
        'role': 'assistant',
        'content': "I'll search for node summaries related to Kamala Harris to show you the entities and their relationships.",
    },
    {
        'role': 'assistant',
        'tool_calls': [
            {
                'name': 'search_nodes',
                'parameters': {
                    'query': 'Kamala Harris',
                    'group_ids': ['politics_figures'],
                    'max_nodes': 5,
                },
            }
        ],
    },
    {
        'role': 'assistant',
        'content': "I found the following entity summaries related to Kamala Harris:\n\n1. **Kamala Harris**: American politician who served as the Attorney General of California (2011-2017) and District Attorney of San Francisco (2004-2011). She is connected to entities like 'California', 'San Francisco', and 'Attorney General'.\n\n2. **California**: A state in the United States where Kamala Harris served as Attorney General. Connected to entities like 'United States' and 'San Francisco'.\n\n3. **San Francisco**: A city in California where Kamala Harris served as District Attorney. Connected to 'California' and 'District Attorney'.\n\nThese node summaries provide a high-level view of the entities in the knowledge graph and how they relate to each other. The summaries are generated based on the relationships between entities.",
    },
]

# Print the conversation as a demonstration
print('Example Conversation Using Graphiti MCP Server with Claude\n')
print('=' * 80)

for message in conversation:
    if message['role'] == 'user':
        print(f"\nUser: {message['content']}\n")
    elif message['role'] == 'assistant':
        if 'tool_calls' in message:
            print(f"Claude (using tool): {json.dumps(message['tool_calls'], indent=2)}\n")
        else:
            print(f"Claude: {message['content']}\n")

    print('-' * 80)

print('\nThis example demonstrates how Claude can use the Graphiti MCP server to:')
print('1. Add episodes to the knowledge graph')
print('2. Search for information in the knowledge graph')
print('3. Build a temporal understanding of facts and relationships')
print('\nIn a real implementation, Claude would receive the actual results from the tool calls')
print('and incorporate them into its responses.')
