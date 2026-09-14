import asyncio
import logging
from functools import partial

from fastapi import APIRouter, status
from graphiti_core.nodes import EpisodeType  # type: ignore

from graph_service.dto import AddMessagesRequest, Message, Result, AddEntityNodeRequest
from graph_service.zep_graphiti import ZepGraphiti, ZepGraphitiDep
from graph_service.config import get_settings
from graphiti_core.utils.maintenance.graph_data_operations import clear_data # type: ignore

logger = logging.getLogger(__name__)


class AsyncWorker:
    def __init__(self):
        self.queue = asyncio.Queue()
        self.task = None
        self.graphiti = None

    async def worker(self):
        while True:
            try:
                job = await self.queue.get()
                await job()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f'AsyncWorker error in worker loop: {e}', exc_info=True)
            finally:
                if 'job' in locals():
                    self.queue.task_done()

    async def start(self):
        settings = get_settings()
        self.graphiti = ZepGraphiti(
            uri=settings.neo4j_uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password
        )
        if settings.openai_base_url:
            self.graphiti.llm_client.config.base_url = settings.openai_base_url
        if settings.openai_api_key:
            self.graphiti.llm_client.config.api_key = settings.openai_api_key
        if settings.model_name:
            self.graphiti.llm_client.model = settings.model_name
            
        self.task = asyncio.create_task(self.worker())

    async def stop(self):
        if self.task:
            self.task.cancel()
            await self.task
        if self.graphiti:
            await self.graphiti.close()
        while not self.queue.empty():
            self.queue.get_nowait()


async_worker = AsyncWorker()


router = APIRouter()


@router.post('/messages', status_code=status.HTTP_202_ACCEPTED)
async def add_messages(
    request: AddMessagesRequest,
):
    async def add_messages_task(m: Message):
        if async_worker.graphiti:
            await async_worker.graphiti.add_episode(
                uuid=m.uuid,
                group_id=request.group_id,
                name=m.name,
                episode_body=f'{m.role or ""}({m.role_type}): {m.content}',
                reference_time=m.timestamp,
                source=EpisodeType.message,
                source_description=m.source_description,
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
