import pandas as pd
from datetime import datetime, timedelta
import json
from graphiti_core.nodes import EpisodicNode, EpisodeType, EntityNode

def create_episodes_from_messages(input_message, input_previous_messages):
    """
    Create an episode and a list of previous episodes from input messages.
    """
    # Current time for the episode
    current_time = datetime.now()

    # Create the current episode
    role = input_message["role"]
    content = input_message["content"]
    message_content = f"{role}: {content}"
    episode = EpisodicNode(
        name="",
        group_id="",
        source=EpisodeType.message,
        type=EpisodeType.message,
        source_description="",
        content=message_content,
        valid_at=current_time,
    )

    # Create previous episodes
    num_previous_messages = len(input_previous_messages)
    previous_times = [current_time - timedelta(minutes=num_previous_messages - i) for i in range(num_previous_messages)]
    previous_episodes = [
        EpisodicNode(
            name="",
            group_id="",
            source=EpisodeType.message,
            source_description="",
            content=f"{message['role']}: {message['content']}",
            valid_at=previous_time,
        )
        for message, previous_time in zip(input_previous_messages, previous_times)
    ]

    return episode, previous_episodes

async def ingest_and_label_snippet(llm_client, snippet_df, output_column_name):
    # Import necessary functions
    from graphiti_core.utils.maintenance.node_operations import extract_nodes, resolve_extracted_nodes
    from graphiti_core.utils.maintenance.edge_operations import extract_edges

    # Loop through each unique message_index_within_snippet in sorted order
    for message_index in sorted(snippet_df['message_index_within_snippet'].unique()):
        message_df = snippet_df[snippet_df['message_index_within_snippet'] == message_index]

        #### Process 'extract_nodes' task
        extract_nodes_row = message_df[message_df['task_name'] == 'extract_nodes']
        assert len(extract_nodes_row) == 1, f"There should be exactly one row for 'extract_nodes' but there are {len(extract_nodes_row)}"
        input_message = json.loads(extract_nodes_row.iloc[0]['input_message'])
        input_previous_messages = json.loads(extract_nodes_row.iloc[0]['input_previous_messages'])
        episode, previous_episodes = create_episodes_from_messages(input_message, input_previous_messages)
        extracted_nodes = await extract_nodes(llm_client, episode, previous_episodes)
        snippet_df.at[extract_nodes_row.index[0], output_column_name] = json.dumps([entity_to_dict(node) for node in extracted_nodes])
        
        #### Process 'dedupe_nodes' task
        dedupe_nodes_row = message_df[message_df['task_name'] == 'dedupe_nodes']
        assert len(dedupe_nodes_row) == 1, "There should be exactly one row for 'dedupe_nodes' but there are {len(dedupe_nodes_row)}"

        # Calculate existing nodes list
        existing_nodes = []
        for prev_message_index in sorted(snippet_df['message_index_within_snippet'].unique()):
            if prev_message_index >= message_index:
                break

            # Filter for previous messages with 'extract_nodes' task
            prev_message_df = snippet_df[
                (snippet_df['message_index_within_snippet'] == prev_message_index) &
                (snippet_df['task_name'] == 'extract_nodes')
            ]

            # Retrieve and deserialize the nodes
            serialized_nodes = prev_message_df.iloc[0][output_column_name]
            node_dicts = json.loads(serialized_nodes)
            nodes = [dict_to_entity(node_dict, EntityNode) for node_dict in node_dicts]
            existing_nodes.extend(nodes)

        existing_nodes_lists = [existing_nodes for _ in range(len(extracted_nodes))]
        resolved_nodes, uuid_map = await resolve_extracted_nodes(llm_client, extracted_nodes, existing_nodes_lists, episode, previous_episodes)
        snippet_df.at[dedupe_nodes_row.index[0], output_column_name] = json.dumps([entity_to_dict(node) for node in resolved_nodes])

        #### Process 'extract_edges' task
        extract_edges_row = message_df[message_df['task_name'] == 'extract_edges']
        assert len(extract_edges_row) == 1, f"There should be exactly one row for 'extract_edges' but there are {len(extract_edges_row)}"
        extracted_edges = await extract_edges(
            llm_client,
            episode,
            extracted_nodes,
            previous_episodes,
            group_id='',
        )
        snippet_df.at[extract_edges_row.index[0], output_column_name] = json.dumps([entity_to_dict(edge) for edge in extracted_edges])

        ########## TODO: Complete the implementation of the below

        #### Process 'dedupe_edges' task
        # dedupe_edges_row = message_df[message_df['task_name'] == 'dedupe_edges']
        # assert len(dedupe_edges_row) == 1, "There should be exactly one row for 'dedupe_edges'"
        # output = dedupe_extracted_edge(
        #     llm_client,
        #     extracted_edge,
        #     related_edges,
        # )
        # snippet_df.at[dedupe_edges_row.index[0], output_column_name] = output

        #### Process 'extract_edge_dates' task
        # extract_edge_dates_row = message_df[message_df['task_name'] == 'extract_edge_dates']
        # assert len(extract_edge_dates_row) == 1, "There should be exactly one row for 'extract_edge_dates'"
        # output = extract_edge_dates(extract_edge_dates_row.iloc[0]['input_extracted_edge_dates'])
        # snippet_df.at[extract_edge_dates_row.index[0], output_column_name] = output

        #### Process 'edge_invalidation' task
        # edge_invalidation_row = message_df[message_df['task_name'] == 'edge_invalidation']
        # assert len(edge_invalidation_row) == 1, "There should be exactly one row for 'edge_invalidation'"
        # output = edge_invalidation(edge_invalidation_row.iloc[0]['input_edge_invalidation'])
        # snippet_df.at[edge_invalidation_row.index[0], output_column_name] = output

    return snippet_df


async def ingest_and_label_minidataset(llm_client, minidataset_df, output_column_name):
    # Add a new column with the specified name, initialized with empty values
    minidataset_df[output_column_name] = None

    minidataset_labelled_df = None
    for snippet_index in sorted(minidataset_df['snippet_index'].unique()):
        snippet_df = minidataset_df[minidataset_df['snippet_index'] == snippet_index]
        
        # Pass the output column name to the ingest_and_label_snippet function
        snippet_df_labelled = await ingest_and_label_snippet(llm_client, snippet_df, output_column_name)
        
        if minidataset_labelled_df is None:
            minidataset_labelled_df = snippet_df_labelled
        else:
            minidataset_labelled_df = pd.concat([minidataset_labelled_df, snippet_df_labelled])
    
    return minidataset_labelled_df

def entity_to_dict(entity):
    """
    Convert an entity object to a dictionary, handling datetime serialization.
    """
    entity_dict = vars(entity)
    for key, value in entity_dict.items():
        if isinstance(value, datetime):
            entity_dict[key] = value.isoformat()  # Convert datetime to ISO 8601 string
    return entity_dict

def dict_to_entity(entity_dict, entity_class):
    """
    Convert a dictionary back to an entity object, handling datetime deserialization.
    """
    for key, value in entity_dict.items():
        try:
            # Attempt to parse strings back to datetime objects
            entity_dict[key] = datetime.fromisoformat(value)
        except (ValueError, TypeError):
            # If parsing fails, keep the original value
            pass
    return entity_class(**entity_dict)