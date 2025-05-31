# New router for /messages2 endpoint
from fastapi import APIRouter, status, HTTPException
from graph_service.dto import AddMessagesRequest
from graph_service.zep_graphiti import ZepGraphitiDep
from graphiti_core.nodes import EpisodeType

from graph_service.routers.ai.extraction import extract_facts_emotions_entities

router = APIRouter()

@router.post('/messages2', status_code=status.HTTP_202_ACCEPTED)
async def add_messages_echo(
    request: AddMessagesRequest,
    graphiti: ZepGraphitiDep,
):
    try:
        # Wyciągnij ostatnią wiadomość usera i ustaw chat_history bez niej
        user_messages = [m for m in request.messages if getattr(m, 'role', None) == 'user']
        last_user_message = user_messages[-1] if user_messages else None
        content = getattr(last_user_message, 'content', None) if last_user_message is not None else None
        group_id = getattr(request, 'group_id', None)
        
        if not content or not str(content).strip():
            raise HTTPException(status_code=400, detail="No user message content provided.")

        # chat_history = wszystkie wiadomości przed ostatnią (nie tylko usera)
        last_user_idx = max(i for i, m in enumerate(request.messages) if getattr(m, 'role', None) == 'user')
        chat_history = request.messages[:last_user_idx]
        print("DEBUG chat_history:", chat_history)

        # Pobierz znane entities z bazy po group_id
        existing_entities = []
        if group_id:
            async with graphiti.driver.session() as session:
                from graph_service.routers.ai.neo4j_operations import get_existing_data
                existing_data = await get_existing_data(session, group_id)
                existing_entities = existing_data.get("entities", [])

        # --- Coreference & entities & emotions ---
        extraction_result = await extract_facts_emotions_entities(
            message_content=content,
            chat_history=chat_history,
            existing_entities=existing_entities,
            extract_emotions=True  # Włącz ekstrakcję emocji
        )
        resolved_entities = extraction_result.get("entities", [])
        resolved_text = extraction_result.get("resolved_text", "")
        resolved_emotions = extraction_result.get("emotions", [])

        # Dodaj nowe entities do bazy
        import uuid
        new_entities = [e for e in resolved_entities if e not in existing_entities]
        entity_uuids = {}
        if group_id:
            # Najpierw pobierz już istniejące entities z bazy (z nazwą i uuid)
            all_entities = []
            from graphiti_core.nodes import EntityNode
            db_entities = await EntityNode.get_by_group_ids(graphiti.driver, [group_id])
            for ent in db_entities:
                entity_uuids[ent.name] = ent.uuid
                all_entities.append(ent.name)
            # Dodaj nowe entities
            for entity in new_entities:
                new_uuid = str(uuid.uuid4())
                await graphiti.save_entity_node(
                    name=entity,
                    uuid=new_uuid,
                    group_id=group_id,
                    summary=""
                )
                entity_uuids[entity] = new_uuid
        # Ustal uuid dla wszystkich encji powiązanych z epizodem
        for entity in resolved_entities:
            if entity not in entity_uuids:
                # fallback: generuj uuid, jeśli nie znaleziono
                entity_uuids[entity] = str(uuid.uuid4())

        # Dodaj epizod (po coref, z resolved_text)
        from datetime import datetime
        episode_uuid = str(uuid.uuid4())
        await graphiti.add_episode(
            uuid=episode_uuid,
            group_id=group_id,
            name="user_message",
            episode_body=resolved_text,
            reference_time=datetime.utcnow().isoformat(),
            source_description="user message after coref",
            source=EpisodeType.message
        )
        # Dodaj relacje MENTIONS (epizod -> entity)
        from graphiti_core.nodes import EpisodicNode
        from graphiti_core.utils.maintenance.edge_operations import build_episodic_edges
        episode_node = await EpisodicNode.get_by_uuid(graphiti.driver, episode_uuid)
        from graphiti_core.nodes import EntityNode
        entity_nodes = [await EntityNode.get_by_uuid(graphiti.driver, entity_uuids[e]) for e in resolved_entities]
        from datetime import datetime as dt
        episodic_edges = build_episodic_edges(entity_nodes, episode_node, dt.utcnow())
        for edge in episodic_edges:
            await edge.save(graphiti.driver)

        # --- Zapisz emocje do bazy i powiąż z epizodem oraz encjami ---
        if resolved_emotions:
            from neo4j import AsyncDriver
            async with graphiti.driver.session() as session:
                for emotion in resolved_emotions:
                    emotion_uuid = str(uuid.uuid4())
                    # Utwórz węzeł emocji jeśli nie istnieje
                    await session.run(
                        """
                        MERGE (e:Emotion {name: $name})
                        ON CREATE SET e.uuid = $uuid, e.created_at = datetime()
                        """,
                        name=emotion,
                        uuid=emotion_uuid
                    )
                    # Powiąż epizod z emocją
                    await session.run(
                        """
                        MATCH (ep:Episodic {uuid: $episode_uuid}), (emo:Emotion {name: $name})
                        MERGE (ep)-[:HAS_EMOTION]->(emo)
                        """,
                        episode_uuid=episode_uuid,
                        name=emotion
                    )
                    # Powiąż każdą encję z emocją
                    for entity in resolved_entities:
                        await session.run(
                            """
                            MATCH (en:Entity {uuid: $entity_uuid}), (emo:Emotion {name: $name})
                            MERGE (en)-[:HAS_EMOTION]->(emo)
                            """,
                            entity_uuid=entity_uuids[entity],
                            name=emotion
                        )

        return {
            "entities": resolved_entities,
            "resolved_text": resolved_text,
            "emotions": resolved_emotions,
            "episode_uuid": episode_uuid
        }
    except Exception as e:
        import traceback
        import sys
        print("[ERROR /messages2]", traceback.format_exc(), file=sys.stderr)
        return {"error": str(e), "traceback": traceback.format_exc()}
