import asyncio
from contextlib import asynccontextmanager
from functools import partial

from fastapi import APIRouter, FastAPI, status
from graphiti_core.nodes import EpisodeType  # type: ignore
from graphiti_core.utils.maintenance.graph_data_operations import clear_data  # type: ignore

from graph_service.config import get_settings
from graph_service.dto import AddEntityNodeRequest, AddMessagesRequest, Message, Result
from graph_service.zep_graphiti import ZepGraphiti, ZepGraphitiDep, _create_graphiti_client


class AsyncWorker:
    def __init__(self, graphiti_client: ZepGraphiti):
        self.queue = asyncio.Queue()
        self.task = None
        self.graphiti_client = graphiti_client

    async def worker(self):
        import logging
        logger = logging.getLogger(__name__)
        
        while True:
            try:
                logger.info(f'Async worker waiting for job (queue size: {self.queue.qsize()})')
                job = await self.queue.get()
                logger.info(f'Async worker processing job (remaining queue size: {self.queue.qsize()})')
                await job()
                logger.info(f'Async worker completed job successfully')
            except Exception as ex:
                logger.error(f'Async worker error processing job: {ex}', exc_info=True)
            except asyncio.CancelledError:
                logger.info('Async worker cancelled')
                break

    async def start(self):
        self.task = asyncio.create_task(self.worker())

    async def stop(self):
        if self.task:
            self.task.cancel()
            await self.task
        while not self.queue.empty():
            self.queue.get_nowait()


async_worker: AsyncWorker | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Create a shared graphiti client for the async worker
    settings = get_settings()
    shared_client = _create_graphiti_client(settings)
    
    # Configure the shared client
    api_key = settings.openai_api_key.strip() if settings.openai_api_key else None
    if settings.openai_base_url is not None:
        shared_client.llm_client.config.base_url = settings.openai_base_url
    if api_key:
        shared_client.llm_client.config.api_key = api_key
        if hasattr(shared_client.embedder, 'client') and hasattr(shared_client.embedder.client, 'api_key'):
            shared_client.embedder.client.api_key = api_key
        if hasattr(shared_client.embedder, 'config') and hasattr(shared_client.embedder.config, 'api_key'):
            shared_client.embedder.config.api_key = api_key
    if settings.model_name is not None:
        shared_client.llm_client.model = settings.model_name
    if settings.embedding_model_name is not None and hasattr(shared_client.embedder, 'config'):
        if hasattr(shared_client.embedder.config, 'embedding_model'):
            shared_client.embedder.config.embedding_model = settings.embedding_model_name
    
    global async_worker
    async_worker = AsyncWorker(shared_client)
    await async_worker.start()
    
    yield
    
    await async_worker.stop()
    await shared_client.close()


router = APIRouter(lifespan=lifespan)


@router.post('/messages', status_code=status.HTTP_202_ACCEPTED)
async def add_messages(
    request: AddMessagesRequest,
    graphiti: ZepGraphitiDep,
):
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info(f"Add messages request - group_id: {request.group_id}, message count: {len(request.messages)}")
    for idx, m in enumerate(request.messages):
        logger.debug(f"Message {idx + 1}: uuid={m.uuid}, name={m.name}, role_type={m.role_type}, content_length={len(m.content)}, source_description={m.source_description[:100] if m.source_description else 'N/A'}")
    
    async def add_messages_task(m: Message):
        if async_worker is None:
            raise RuntimeError('Async worker not initialized')
        logger.info(f"Processing message - uuid: {m.uuid}, group_id: {request.group_id}, name: {m.name}")
        try:
            # Get organization-specific Graphiti client (uses group_id as database name for FalkorDB)
            from graph_service.zep_graphiti import get_graphiti_for_group
            org_graphiti = get_graphiti_for_group(request.group_id, async_worker.graphiti_client)
            
            await org_graphiti.add_episode(
                uuid=m.uuid,
                group_id=request.group_id,
                name=m.name,
                episode_body=f'{m.role or ""}({m.role_type}): {m.content}',
                reference_time=m.timestamp,
                source=EpisodeType.message,
                source_description=m.source_description,
            )
            logger.info(f"Successfully added episode - uuid: {m.uuid}, group_id: {request.group_id}")
        except Exception as ex:
            logger.error(f"Failed to add episode - uuid: {m.uuid}, group_id: {request.group_id}, error: {ex}", exc_info=True)
            raise

    for m in request.messages:
        await async_worker.queue.put(partial(add_messages_task, m))

    logger.info(f"Queued {len(request.messages)} messages for processing")
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
