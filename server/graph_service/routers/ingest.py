import asyncio
import uuid  # add UUID generation
from contextlib import asynccontextmanager
from functools import partial

from fastapi import APIRouter, FastAPI, status
from graphiti_core.nodes import EpisodeType  # type: ignore
from graphiti_core.utils.maintenance.graph_data_operations import clear_data  # type: ignore

from graph_service.dto import AddEntityNodeRequest, AddMessagesRequest, Message, Result, TokenUsage
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
    message_results = []

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
                extraction_result = await extractFactsAndStore(
                    graphiti, 
                    m, 
                    request.group_id, 
                    chat_history_from_request, # Use history from request
                    shirt_slug_from_request    # Use shirt_slug from request
                )
                if extraction_result:
                    print(f"[Graphiti] Extraction result for message {m.uuid}: {extraction_result}")
                    # Store extraction result directly
                    message_results.append(extraction_result)

        except Exception as e:
            print(f"[Graphiti] ERROR: {e}")

    for m_message in request.messages: # Renamed loop variable for clarity
        await add_messages_task(m_message)

    print(f"[Graphiti] FINAL message_results: {message_results}")

    # Combine all message results into tokens structure  
    combined_tokens = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "model": None,
        "temperature": None
    }

    data = {
        "facts": [],
        "emotions": [],
        "entities": [],
    }
    
    for result in message_results:
        combined_tokens["input_tokens"] += result.get("input_tokens", 0)
        combined_tokens["output_tokens"] += result.get("output_tokens", 0)
        combined_tokens["total_tokens"] += result.get("total_tokens", 0)
        
        # Set model and temperature from the first result (they should be the same for all)
        # Extract from models/temperatures dictionaries if they exist
        models_dict = result.get("models", {})
        temperatures_dict = result.get("temperatures", {})
        
        if combined_tokens["model"] is None and models_dict:
            # Use the facts model as the primary model (they should all be the same)
            combined_tokens["model"] = models_dict.get("facts") or next(iter(models_dict.values()), None)
        if combined_tokens["temperature"] is None and temperatures_dict:
            # Use the facts temperature as the primary temperature 
            combined_tokens["temperature"] = temperatures_dict.get("facts")
            if combined_tokens["temperature"] is None:
                combined_tokens["temperature"] = next(iter(temperatures_dict.values()), None)

    for result in message_results:
        data["facts"].extend(result.get("facts", []))
        data["emotions"].extend(result.get("emotions", []))
        data["entities"].extend(result.get("entities", []))

    return Result(
        message='Messages added to processing queue',
        success=True,
        tokens=TokenUsage(**combined_tokens),
        data=data
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
