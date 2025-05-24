from datetime import datetime, timezone

from fastapi import APIRouter, status

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
    relations_dict: dict[str, list[RelationItem]] = {rt: [] for rt in request.relation_types}
    async with graphiti.driver.session() as session:
        for rt in request.relation_types:
            if rt == 'emotions':
                query = '''
                    MATCH (e:Episodic {group_id: $group_id})-[r:HAS_EMOTION]->(n:Emotion)
                    RETURN e.uuid AS episodic_id, n.text AS text
                '''
            elif rt == 'relations':
                query = '''
                    MATCH (e:Episodic {group_id: $group_id})-[r:HAS_RELATION]->(n:Relation)
                    RETURN e.uuid AS episodic_id, n.text AS text
                '''
            elif rt == 'facts':
                query = '''
                    MATCH (e:Episodic {group_id: $group_id})-[r:IS_FACT]->(n:Fact)
                    RETURN e.uuid AS episodic_id, n.text AS text
                '''
            elif rt == 'memories':
                query = '''
                    MATCH (e:Episodic {group_id: $group_id})-[r:HAS_MEMORY]->(n:Memory)
                    RETURN e.uuid AS episodic_id, n.text AS text
                '''
            else:
                continue
            result = await session.run(query, group_id=request.group_id)
            records = await result.data()
            for rec in records:
                relations_dict[rt].append(
                    RelationItem(episodic_id=rec['episodic_id'], text=rec['text'])
                )
    return RelationsResponse(relations=relations_dict)


def compose_query_from_messages(messages: list[Message]):
    combined_query = ''
    for message in messages:
        combined_query += f'{message.role_type or ""}({message.role or ""}): {message.content}\n'
    return combined_query
