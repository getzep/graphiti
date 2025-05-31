from datetime import datetime, timezone

from fastapi import APIRouter, status
import logging

from graph_service.dto import (
    GetEntityRequest,
    GetEntityResponse,
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


@router.post('/get-entity', status_code=status.HTTP_200_OK)
async def get_entity(
    request: GetEntityRequest,
    graphiti: ZepGraphitiDep,
):
    combined_query = compose_query_from_messages(request.messages)
    result = await graphiti.search(
        group_ids=[request.group_id],
        query=combined_query,
        num_results=request.max_facts,
    )
    facts = [get_fact_result_from_edge(edge) for edge in result]
    return GetEntityResponse(facts=facts)


@router.post('/relations', status_code=status.HTTP_200_OK)
async def get_relations(
    request: GetRelationsRequest,
    graphiti: ZepGraphitiDep,
):
    # przygotuj słownik wyników dla żądanych typów relacji
    relations_dict = {rt: [] for rt in request.relation_types}
    # mapa Cypher → klucz w odpowiedzi
    rel_map = {
        'HAS_EMOTION': 'emotions',
        'HAS_MEMORY':  'memories',
        'IS_FACT':     'facts',
    }
    query = """
    MATCH (e:Episodic {group_id: $group_id})-[r]->(n)
    WHERE r.group_id = $group_id
      AND type(r) IN $rel_types
    RETURN
      type(r)      AS rel_type,
      e.uuid       AS episodic_id,
      n.text       AS text
    """
    async with graphiti.driver.session() as session:
        result = await session.run(
            query,
            group_id=request.group_id,
            rel_types=list(rel_map.keys())
        )
        records = await result.data()
    for rec in records:
        key = rel_map.get(rec['rel_type'])
        if key in relations_dict:
            relations_dict[key].append(
                RelationItem(
                    episodic_id=rec['episodic_id'],
                    text=rec['text']
                )
            )
    return RelationsResponse(relations=relations_dict)



def compose_query_from_messages(messages: list[Message]):
    combined_query = ''
    for message in messages:
        combined_query += f'{message.role_type or ""}({message.role or ""}): {message.content}\n'
    return combined_query
