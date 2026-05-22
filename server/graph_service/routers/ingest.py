import asyncio
import logging
from functools import partial
from typing import Any, cast

from fastapi import APIRouter, HTTPException, status
from graphiti_core.nodes import EpisodeType  # type: ignore
from graphiti_core.utils.maintenance.graph_data_operations import clear_data  # type: ignore

from graph_service.dto import AddEntityNodeRequest, AddMessagesRequest, Message, Result
from graph_service.ontologies import is_known_schema_id, resolve_ontology
from graph_service.zep_graphiti import ZepGraphitiDep

logger = logging.getLogger(__name__)


class AsyncWorker:
    def __init__(self):
        self.queue = asyncio.Queue()
        self.task = None

    async def worker(self):
        while True:
            job = None
            try:
                job = await self.queue.get()
                await job()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception('Graphiti background job failed.')
            finally:
                if job is not None:
                    self.queue.task_done()

    async def start(self):
        self.task = asyncio.create_task(self.worker())

    async def stop(self):
        if self.task:
            self.task.cancel()
            await self.task
        while not self.queue.empty():
            self.queue.get_nowait()


async_worker = AsyncWorker()

router = APIRouter()


@router.post('/messages', status_code=status.HTTP_202_ACCEPTED)
async def add_messages(
    request: AddMessagesRequest,
    graphiti: ZepGraphitiDep,
):
    if request.schema_id is not None and not is_known_schema_id(request.schema_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f'Unknown schema_id: {request.schema_id}',
        )

    async def add_messages_task(m: Message):
        ontology = resolve_ontology(request.schema_id, m.content)
        await graphiti.add_episode(
            uuid=m.uuid,
            group_id=request.group_id,
            name=m.name,
            episode_body=f'{m.role or ""}({m.role_type}): {m.content}',
            reference_time=m.timestamp,
            source=EpisodeType.message,
            source_description=m.source_description,
            entity_types=cast(Any, ontology.entity_types) if ontology else None,
            edge_types=cast(Any, ontology.edge_types) if ontology else None,
            edge_type_map=ontology.edge_type_map if ontology else None,
        )

    for m in request.messages:
        await async_worker.queue.put(partial(add_messages_task, m))

    return Result(message='Messages added to processing queue', success=True)


@router.post('/entity-node', status_code=status.HTTP_201_CREATED)
async def add_entity_node(
    request: AddEntityNodeRequest,
    graphiti: ZepGraphitiDep,
):
    node = await graphiti.save_entity_node(
        uuid=request.uuid,
        group_id=request.group_id,
        name=request.name,
        summary=request.summary,
    )
    return node


@router.delete('/entity-edge/{uuid}', status_code=status.HTTP_200_OK)
async def delete_entity_edge(uuid: str, graphiti: ZepGraphitiDep):
    await graphiti.delete_entity_edge(uuid)
    return Result(message='Entity Edge deleted', success=True)


@router.delete('/group/{group_id}', status_code=status.HTTP_200_OK)
async def delete_group(group_id: str, graphiti: ZepGraphitiDep):
    await graphiti.delete_group(group_id)
    return Result(message='Group deleted', success=True)


@router.delete('/episode/{uuid}', status_code=status.HTTP_200_OK)
async def delete_episode(uuid: str, graphiti: ZepGraphitiDep):
    await graphiti.delete_episodic_node(uuid)
    return Result(message='Episode deleted', success=True)


@router.post('/clear', status_code=status.HTTP_200_OK)
async def clear(
    graphiti: ZepGraphitiDep,
):
    await clear_data(graphiti.driver)
    await graphiti.build_indices_and_constraints()
    return Result(message='Graph cleared', success=True)
