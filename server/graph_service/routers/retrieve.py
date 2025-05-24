from datetime import datetime, timezone

from fastapi import APIRouter, status
import logging

from graph_service.dto import (
    GetMemoryRequest,
    GetMemoryResponse,
    Message,
    SearchQuery,
    SearchResults,
)
from graph_service.dto.retrieve import GetRelationsRequest, RelationsResponse, RelationItem
from graph_service.zep_graphiti import ZepGraphitiDep, get_fact_result_from_edge

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post('/search', status_code=status.HTTP_200_OK)
async def search(query: SearchQuery, graphiti: ZepGraphitiDep):
    relevant_edges = await graphiti.search(
        group_ids=query.group_ids,
        query=query.query,
        num_results=query.max_facts,
    )
    facts = [get_fact_result_from_edge(edge) for edge in relevant_edges]
    return SearchResults(
        facts=facts,
    )


@router.get('/entity-edge/{uuid}', status_code=status.HTTP_200_OK)
async def get_entity_edge(uuid: str, graphiti: ZepGraphitiDep):
    entity_edge = await graphiti.get_entity_edge(uuid)
    return get_fact_result_from_edge(entity_edge)


@router.get('/episodes/{group_id}', status_code=status.HTTP_200_OK)
async def get_episodes(group_id: str, last_n: int, graphiti: ZepGraphitiDep):
    episodes = await graphiti.retrieve_episodes(
        group_ids=[group_id], last_n=last_n, reference_time=datetime.now(timezone.utc)
    )
    return episodes


@router.post('/get-memory', status_code=status.HTTP_200_OK)
async def get_memory(
    request: GetMemoryRequest,
    graphiti: ZepGraphitiDep,
):
    combined_query = compose_query_from_messages(request.messages)
    result = await graphiti.search(
        group_ids=[request.group_id],
        query=combined_query,
        num_results=request.max_facts,
    )
    facts = [get_fact_result_from_edge(edge) for edge in result]
    return GetMemoryResponse(facts=facts)


@router.post('/relations', status_code=status.HTTP_200_OK)
async def get_relations(
    request: GetRelationsRequest,
    graphiti: ZepGraphitiDep,
):
    print(f"[get_relations] called with group_id={request.group_id}, relation_types={request.relation_types}")
    relations_dict: dict[str, list[RelationItem]] = {rt: [] for rt in request.relation_types}
    # mapping of type keys to (relationship type, node label)
    mapping = {
        'emotions': ('HAS_EMOTION', 'Emotion'),
        'relations': ('HAS_RELATION', 'Relation'),
        'facts': ('IS_FACT', 'Fact'),
        'memories': ('HAS_MEMORY', 'Memory'),
    }
    async with graphiti.driver.session() as session:
        for rt in request.relation_types:
            print(f"[get_relations] Processing relation type: {rt}")
            rel_info = mapping.get(rt)
            if not rel_info:
                print(f"[get_relations] Warning: Unknown relation type: {rt}")
                continue
            rel_type, node_label = rel_info
            print(f"[get_relations] Running query for rel_type={rel_type}, node_label={node_label}")
            # use generic relationship match to avoid warnings for unknown types
            query = f'''
                MATCH (e:Episodic {{group_id: $group_id}})-[r]->(n:{node_label})
                WHERE type(r) = $rel_type
                RETURN e.uuid AS episodic_id, n.text AS text
            '''
            result = await session.run(query, group_id=request.group_id, rel_type=rel_type)
            records = await result.data()
            print(f"[get_relations] Found {len(records)} records for {rt}")
            for rec in records:
                print(f"[get_relations] Record for {rt}: {rec}")
                relations_dict[rt].append(
                    RelationItem(episodic_id=rec['episod_id'], text=rec['text'])
                )
    return RelationsResponse(relations=relations_dict)


def compose_query_from_messages(messages: list[Message]):
    combined_query = ''
    for message in messages:
        combined_query += f'{message.role_type or ""}({message.role or ""}): {message.content}\n'
    return combined_query
