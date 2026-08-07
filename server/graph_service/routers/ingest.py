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

# Retry policy for transient ingestion failures (rate limits, transient
# network/DB errors). Permanent failures are skipped without retry.
MAX_RETRIES = 5
BASE_BACKOFF_SECONDS = 5  # exponential backoff: 5, 10, 20, 40s

# Built-in exceptions that indicate bad input/data — retrying the same job will
# never succeed, so skip immediately rather than burn the backoff schedule.
PERMANENT_ERROR_TYPES = (IndexError, KeyError, ValueError)
# Provider HTTP errors (e.g. OpenAI) aren't imported here; match them by class
# name and by client-error status codes that won't change on retry. Note that
# rate limits (429) and timeouts (408) are deliberately excluded — those are
# transient and should be retried.
PERMANENT_ERROR_NAMES = {'BadRequestError', 'NotFoundError', 'UnprocessableEntityError'}
PERMANENT_STATUS_CODES = {400, 404, 422}


def _is_permanent_error(exc: Exception) -> bool:
    """Whether ``exc`` is a permanent (non-retryable) failure."""
    if isinstance(exc, PERMANENT_ERROR_TYPES):
        return True
    if type(exc).__name__ in PERMANENT_ERROR_NAMES:
        return True
    return getattr(exc, 'status_code', None) in PERMANENT_STATUS_CODES


class AsyncWorker:
    def __init__(self):
        self.queue = asyncio.Queue()
        self.task = None

    async def _process_job(self, job) -> None:
        """Run a single job, retrying transient failures with exponential backoff.

        Previously ``worker()`` ran ``await job()`` directly, so any exception
        other than cancellation propagated out of the loop and killed the worker
        coroutine — leaving every queued job permanently unprocessed while the
        endpoint had already returned ``202 Accepted``. Catch and classify
        failures here so a single bad job can never take down the worker.
        """
        for attempt in range(MAX_RETRIES):
            try:
                await job()
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                err_name = type(exc).__name__
                if _is_permanent_error(exc):
                    logger.error('Job skipped (permanent error: %s)', err_name, exc_info=True)
                    return
                if attempt >= MAX_RETRIES - 1:
                    logger.error(
                        'Job failed after %d attempts, skipping (%s)',
                        MAX_RETRIES,
                        err_name,
                        exc_info=True,
                    )
                    return
                wait = BASE_BACKOFF_SECONDS * (2**attempt)
                logger.warning(
                    'Job attempt %d/%d failed (%s); retrying in %ds',
                    attempt + 1,
                    MAX_RETRIES,
                    err_name,
                    wait,
                )
                await asyncio.sleep(wait)

    async def worker(self):
        while True:
            try:
                print(f'Got a job: (size of remaining queue: {self.queue.qsize()})')
                job = await self.queue.get()
                await self._process_job(job)
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
