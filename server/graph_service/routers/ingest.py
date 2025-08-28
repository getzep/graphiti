import asyncio
import json
from contextlib import asynccontextmanager
from functools import partial
import logging

from fastapi import APIRouter, FastAPI, status
from graphiti_core.nodes import EpisodeType  # type: ignore
from graphiti_core.utils.bulk_utils import RawEpisode  # type: ignore
from graphiti_core.utils.maintenance.graph_data_operations import clear_data  # type: ignore

from graph_service.dto import AddEntityNodeRequest, AddEpisodesRequest, AddMessagesRequest, Episode, Message, Result
from graph_service.zep_graphiti import ZepGraphitiDep
from graph_service.models import entity_types, edge_types, edge_type_map


# init our logger
log = logging.getLogger("ingest")

class AsyncWorker:
    def __init__(self):
        self.queue = asyncio.Queue()
        self.task = None

    async def worker(self):
        while True:
            try:
                log.info(f'Got a job: (size of remaining queue: {self.get_queue_size()})')
                job = await self.queue.get()
                await job()
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f'Error in async worker: {e}')
                pass

    async def start(self):
        self.task = asyncio.create_task(self.worker())

    async def stop(self):
        if self.task:
            self.task.cancel()
            await self.task
        while not self.queue.empty():
            self.queue.get_nowait()

    def get_status(self) -> str:
        """Get the current status of the async worker."""
        if self.task is None:
            return "not_started"
        elif self.task.done():
            if self.task.cancelled():
                return "cancelled"
            elif self.task.exception():
                return "error"
            else:
                return "stopped"
        else:
            return "running"

    def get_queue_size(self) -> int:
        """Get the current queue size."""
        return self.queue.qsize()


async_worker = AsyncWorker()


@asynccontextmanager
async def lifespan(_: FastAPI):
    log.info(f'Starting async worker')
    await async_worker.start()
    yield
    log.info(f'Stopping async worker')
    await async_worker.stop()


router = APIRouter(lifespan=lifespan)


@router.post('/messages', status_code=status.HTTP_202_ACCEPTED)
async def add_messages(
    request: AddMessagesRequest,
    graphiti: ZepGraphitiDep,
):
    async def add_messages_task(m: Message):
        await graphiti.add_episode(
            uuid=m.uuid,
            group_id=request.group_id,
            name=m.name,
            episode_body=f'{m.role or ""}({m.role_type}): {m.content}',
            reference_time=m.timestamp,
            source=EpisodeType.message,
            source_description=m.source_description,
            edge_type_map=edge_type_map,
            edge_types=edge_types,
            entity_types=entity_types
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


@router.post('/episodes', status_code=status.HTTP_202_ACCEPTED)
async def add_episodes(
    request: AddEpisodesRequest,
    graphiti: ZepGraphitiDep,
):
    async def add_episode_task(episode: Episode):
        # episode_body is already normalized to a string by the Episode model validator
        await graphiti.add_episode(
            uuid=episode.uuid,
            group_id=request.group_id,
            name=episode.name,
            episode_body=episode.episode_body,
            reference_time=episode.reference_time,
            source=EpisodeType.from_str(episode.source),
            source_description=episode.source_description,
            update_communities=episode.update_communities,
            edge_type_map=edge_type_map,
            edge_types=edge_types,
            entity_types=entity_types
        )

    for episode in request.episodes:
        log.info(f'Adding episode to queue: {episode.name}')
        await async_worker.queue.put(partial(add_episode_task, episode))

    return Result(message=F'Episodes added to processing queue (size of remaining queue: {async_worker.get_queue_size()})', success=True)


@router.post('/episodes_bulk', status_code=status.HTTP_202_ACCEPTED)
async def add_episodes_bulk(
    request: AddEpisodesRequest,
    graphiti: ZepGraphitiDep,
):
    # Convert Episode objects to RawEpisode objects for bulk processing
    raw_episodes: list[RawEpisode] = []
    
    for episode in request.episodes:
        # episode_body is already normalized to a string by the Episode model validator
        raw_episode = RawEpisode(
            uuid=episode.uuid,
            name=episode.name,
            content=episode.episode_body,
            source_description=episode.source_description,
            reference_time=episode.reference_time,
            source=EpisodeType.from_str(episode.source),
        )
        raw_episodes.append(raw_episode)
    
    log.info(f'Processing {len(raw_episodes)} episodes in bulk for group {request.group_id}')
    
    # Use the bulk method directly (not queued) since it's already optimized for batch processing
    await graphiti.add_episode_bulk(
        bulk_episodes=raw_episodes,
        group_id=request.group_id,
        edge_type_map=edge_type_map,
        edge_types=edge_types,
        entity_types=entity_types
    )
    
    return Result(message=f'Successfully processed {len(raw_episodes)} episodes in bulk', success=True)


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
