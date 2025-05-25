import asyncio
import uuid  # add UUID generation
from contextlib import asynccontextmanager
from functools import partial

from fastapi import APIRouter, FastAPI, status
from graphiti_core.nodes import EpisodeType  # type: ignore
from graphiti_core.utils.maintenance.graph_data_operations import clear_data  # type: ignore

from graph_service.dto import AddEntityNodeRequest, AddMessagesRequest, Message, Result
from graph_service.zep_graphiti import ZepGraphitiDep

from graph_service.routers.ai import extractFactsAndStore
# from graph_service.routers.ai.emotionExtractor import extractEmotionsAndStore
# from graph_service.routers.ai.memoryExtractor import extractMemoriesAndStore
# from graph_service.routers.ai.relationshipExtractor import extractRelationsAndStore
# from graph_service.routers.ai.presenceExtractor import extractPresenceAndStore


class AsyncWorker:
    def __init__(self):
        self.task = None

    async def worker(self):
        while True:
            try:
                print(f'Got a job: (size of remaining queue: {self.queue.qsize()})')
                job = await self.queue.get()
                await job()
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
    request: AddMessagesRequest, # Assume this DTO might contain 'chat_history' and 'shirt_slug'
    graphiti: ZepGraphitiDep,
):
    token_usages = []

    chat_history_from_request = getattr(request, 'chat_history', "")
    shirt_slug_from_request = getattr(request, 'shirt_slug', None)

    async def add_messages_task(m: Message): # Inner function, can access chat_history_from_request and shirt_slug_from_request
        # Pomiń, jeśli wiadomość nie ma treści
        if not hasattr(m, 'content') or not m.content or not m.content.strip():
            print(f"[Graphiti] Pomijam pustą wiadomość (uuid: {getattr(m, 'uuid', None)})")
            return None
        # Ensure message has UUID to avoid null merges
        if not getattr(m, 'uuid', None):
            m.uuid = str(uuid.uuid4())
        print("[Graphiti] Zaczynam dodawanie do Neo4j")
        try:
            await graphiti.add_episode(
                uuid=m.uuid,
                group_id=request.group_id,
                name=m.name,
                episode_body=f'{m.role or ""}({m.role_type}): {m.content}',
                reference_time=m.timestamp,
                source=EpisodeType.message,
                source_description=m.source_description,
            )
            print("[Graphiti] DONE dodane")

            if m.role == "user":
                token_usage = await extractFactsAndStore(
                    graphiti, 
                    m, 
                    request.group_id, 
                    chat_history_from_request, # Use history from request
                    shirt_slug_from_request    # Use shirt_slug from request
                )
                if token_usage:
                    print(f"[Graphiti] Token usage for message {m.uuid}: {token_usage}")
                    token_usages.append({"uuid": m.uuid, **token_usage})

        except Exception as e:
            print(f"[Graphiti] ERROR: {e}")

    for m_message in request.messages: # Renamed loop variable for clarity
        await add_messages_task(m_message)

    print(f"[Graphiti] FINAL token_usages: {token_usages}")

    # Extract only token usage information (not facts/emotions/entities)
    tokens_only = []
    if token_usages:
        for usage in token_usages:
            tokens_only.append({
                "uuid": usage.get("uuid"),
                "tokens": usage.get("tokens", {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0
                })
            })

    return Result(
        message='Messages added to processing queue',
        success=True,
        tokens=tokens_only
    )


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
