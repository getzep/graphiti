import asyncio
import logging
from contextlib import asynccontextmanager
from functools import partial

from fastapi import APIRouter, FastAPI, status
from graphiti_core.nodes import EpisodeType  # type: ignore
from graphiti_core.utils.maintenance.graph_data_operations import clear_data  # type: ignore

from graph_service.dto import AddEntityNodeRequest, AddMessagesRequest, Message, Result
from graph_service.zep_graphiti import ZepGraphitiDep

logger = logging.getLogger(__name__)

MAX_RETRIES = 5
BASE_BACKOFF = 5  # seconds

# Errors that are permanent — retrying the same data won't help, skip immediately
PERMANENT_ERRORS = (
    IndexError,   # list index out of range in entity resolution
    KeyError,     # UUID not found in graph result set
    ValueError,   # data format / schema issues
)

# Error type names for permanent HTTP errors (provider-specific subclasses)
PERMANENT_ERROR_NAMES = {
    'BadRequestError',           # HTTP 400
    'NotFoundError',             # HTTP 404
    'UnprocessableEntityError',
}


def is_permanent(e: Exception) -> bool:
    if isinstance(e, PERMANENT_ERRORS):
        return True
    if type(e).__name__ in PERMANENT_ERROR_NAMES:
        return True
    if hasattr(e, 'status_code') and e.status_code == 400:
        return True
    return False


class AsyncWorker:
    def __init__(self):
        self.queue = asyncio.Queue()
        self.task = None

    async def worker(self):
        while True:
            try:
                print(f'Got a job: (size of remaining queue: {self.queue.qsize()})')
                job = await self.queue.get()
                for attempt in range(MAX_RETRIES):
                    try:
                        await job()
                        break  # success — move to next job
                    except asyncio.CancelledError:
                        raise
                    except Exception as e:
                        err_name = type(e).__name__

                        # Permanent errors: skip immediately, no point retrying
                        if is_permanent(e):
                            print(f'[worker] permanent error ({err_name}), skipping: {e}')
                            logger.error('Job skipped (permanent error): %s', e, exc_info=True)
                            break

                        # Transient errors: retry with exponential backoff
                        if attempt < MAX_RETRIES - 1:
                            wait = BASE_BACKOFF * (2 ** attempt)  # 5, 10, 20, 40s
                            print(
                                f'[worker] attempt {attempt + 1}/{MAX_RETRIES} failed '
                                f'({err_name}), retry in {wait}s: {e}'
                            )
                            logger.warning(
                                'Job attempt %d/%d failed, retrying in %ds: %s',
                                attempt + 1, MAX_RETRIES, wait, e,
                            )
                            await asyncio.sleep(wait)
                        else:
                            print(
                                f'[worker] job failed after {MAX_RETRIES} attempts '
                                f'({err_name}), skipping: {e}'
                            )
                            logger.error(
                                'Job failed after %d attempts, skipping: %s',
                                MAX_RETRIES, e, exc_info=True,
                            )
            except asyncio.CancelledError:
                break

    async def start(self):
        self.task = asyncio.create_task(self.worker())

    async def stop(self):
        if self.task:
            self.task.cancel()
            await self.task
        while not self.queue.empty():
            self.queue.get_nowait()


async_worker = AsyncWorker()


@asynccontextmanager
async def lifespan(_: FastAPI):
    await async_worker.start()
    yield
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
