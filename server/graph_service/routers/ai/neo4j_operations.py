"""
Neo4j database operations for fact extraction.
"""
from typing import Dict, List, Any, Optional

async def get_existing_data(session, group_id: str) -> Dict[str, List[str]]:
    """
    Retrieve existing facts, emotions, and entities for a given group_id.
    
    Args:
        session: Neo4j database session
        group_id: Group ID to fetch data for
        
    Returns:
        Dictionary containing lists of existing facts, emotions, and entities
    """
    # Initialize results
    existing_facts = []
    existing_emotions = []
    existing_entities = []
    
    # Fetch existing facts
    result_facts = await session.run(
        """
        MATCH (fact_node:Fact)<-[r:IS_FACT]-(:Episodic)
        WHERE r.group_id = $group_id
        RETURN collect(DISTINCT {text: fact_node.text, count: COALESCE(fact_node.count, 0)}) AS fact_nodes_for_group
        """,
        {"group_id": group_id}
    )
    record_facts = await result_facts.single()
    if record_facts and record_facts["fact_nodes_for_group"]:
        fact_nodes = record_facts["fact_nodes_for_group"]
        existing_facts = [fn["text"] for fn in fact_nodes if fn and fn.get("text")]
    
    # Fetch existing emotions
    result_emotions = await session.run(
        """
        MATCH (emotion_node:Emotion)<-[r:HAS_EMOTION]-(:Episodic)
        WHERE r.group_id = $group_id
        RETURN collect(DISTINCT {text: emotion_node.text, count: COALESCE(emotion_node.count, 0)}) AS emotion_nodes_for_group
        """,
        {"group_id": group_id}
    )
    record_emotions = await result_emotions.single()
    if record_emotions and record_emotions["emotion_nodes_for_group"]:
        emotion_nodes = record_emotions["emotion_nodes_for_group"]
        existing_emotions = [en["text"] for en in emotion_nodes if en and en.get("text")]

    # Fetch existing entities
    result_entities = await session.run(
        """
        MATCH (entity_node:Entity)<-[r:HAS_ENTITY]-(:Episodic)
        WHERE r.group_id = $group_id
        RETURN collect(DISTINCT {text: entity_node.text, count: COALESCE(entity_node.count, 0)}) AS entity_nodes_for_group
        """,
        {"group_id": group_id}
    )
    record_entities = await result_entities.single()
    if record_entities and record_entities["entity_nodes_for_group"]:
        entity_nodes = record_entities["entity_nodes_for_group"]
        existing_entities = [mn["text"] for mn in entity_nodes if mn and mn.get("text")]
        
    return {
        "facts": existing_facts,
        "emotions": existing_emotions,
        "entities": existing_entities
    }

async def store_extracted_data(
    session, 
    uuid: str, 
    group_id: str, 
    facts: List[str], 
    emotions: List[str], 
    entities: List[str], 
    shirt_slug: str
) -> None:
    """
    Store extracted facts, emotions, and entities in Neo4j.
    
    Args:
        session: Neo4j database session
        uuid: UUID of the message
        group_id: Group ID for the data
        facts: List of facts to store
        emotions: List of emotions to store
        entities: List of entities to store
        shirt_slug: Shirt slug to associate with the data
    """
    await session.run(
        """
        MERGE (e:Episodic {uuid: $uuid})
        ON CREATE SET e.group_id = $group_id
        WITH e
        MERGE (s:Shirt {slug: $shirt_slug})
        WITH e, s
        MERGE (e)-[:CONNECTED_TO]->(s)
        WITH e

        UNWIND $emotions AS emo
          MERGE (em:Emotion {text: emo})
          ON CREATE SET em.count = 1
          ON MATCH SET em.count = COALESCE(em.count, 0) + 1
          MERGE (e)-[rel:HAS_EMOTION {group_id: $group_id, shirt_slug: $shirt_slug}]->(em)
        WITH e

        UNWIND $facts AS fact
          MERGE (f:Fact {text: fact})
          ON CREATE SET f.count = 1
          ON MATCH SET f.count = COALESCE(f.count, 0) + 1
          MERGE (e)-[rel:IS_FACT {group_id: $group_id, shirt_slug: $shirt_slug}]->(f)
        WITH e

        UNWIND $entities AS ent
          MERGE (m:Entity {text: ent})
          ON CREATE SET m.count = 1
          ON MATCH SET m.count = COALESCE(m.count, 0) + 1
          MERGE (e)-[rel:HAS_ENTITY {group_id: $group_id, shirt_slug: $shirt_slug}]->(m)
        """,
        {
            "uuid": uuid,
            "group_id": group_id,
            "emotions": emotions,
            "facts": facts,
            "entities": entities,
            "shirt_slug": shirt_slug
        }
    )

async def get_relationships_data(session, group_id: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Get relationships between facts, emotions, and entities.
    
    Args:
        session: Neo4j database session
        group_id: Group ID to fetch relationships for
        
    Returns:
        Dictionary containing relationships between facts, emotions, and entities
    """
    facts_connected_to_entities = []
    facts_connected_to_emotions = []
    emotions_connected_to_entities = []
    
    # Facts connected to entities
    result_facts_entities = await session.run(
        """
        MATCH (fact_node:Fact)<-[rf:IS_FACT]-(ep:Episodic)-[re:HAS_ENTITY]->(entity_node:Entity)
        WHERE rf.group_id = $group_id AND re.group_id = $group_id
        RETURN collect(DISTINCT {
            fact: fact_node.text, 
            entity: entity_node.text, 
            fact_count: COALESCE(fact_node.count, 0),
            entity_count: COALESCE(entity_node.count, 0)
        }) AS facts_with_entities
        """,
        {"group_id": group_id}
    )
    record_facts_entities = await result_facts_entities.single()
    if record_facts_entities and record_facts_entities["facts_with_entities"]:
        facts_connected_to_entities = record_facts_entities["facts_with_entities"]

    # Facts connected to emotions
    result_facts_emotions = await session.run(
        """
        MATCH (fact_node:Fact)<-[rf:IS_FACT]-(ep:Episodic)-[rem:HAS_EMOTION]->(emotion_node:Emotion)
        WHERE rf.group_id = $group_id AND rem.group_id = $group_id
        RETURN collect(DISTINCT {
            fact: fact_node.text, 
            emotion: emotion_node.text,
            fact_count: COALESCE(fact_node.count, 0),
            emotion_count: COALESCE(emotion_node.count, 0)
        }) AS facts_with_emotions
        """,
        {"group_id": group_id}
    )
    record_facts_emotions = await result_facts_emotions.single()
    if record_facts_emotions and record_facts_emotions["facts_with_emotions"]:
        facts_connected_to_emotions = record_facts_emotions["facts_with_emotions"]

    # Emotions connected to entities
    result_emotions_entities = await session.run(
        """
        MATCH (emotion_node:Emotion)<-[rem:HAS_EMOTION]-(ep:Episodic)-[re:HAS_ENTITY]->(entity_node:Entity)
        WHERE rem.group_id = $group_id AND re.group_id = $group_id
        RETURN collect(DISTINCT {
            emotion: emotion_node.text, 
            entity: entity_node.text,
            emotion_count: COALESCE(emotion_node.count, 0),
            entity_count: COALESCE(entity_node.count, 0)
        }) AS emotions_with_entities
        """,
        {"group_id": group_id}
    )
    record_emotions_entities = await result_emotions_entities.single()
    if record_emotions_entities and record_emotions_entities["emotions_with_entities"]:
        emotions_connected_to_entities = record_emotions_entities["emotions_with_entities"]
    
    return {
        "facts_entities": facts_connected_to_entities,
        "facts_emotions": facts_connected_to_emotions,
        "emotions_entities": emotions_connected_to_entities
    }

async def get_top_items(session, group_id: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Get top facts, emotions, and entities by count.
    
    Args:
        session: Neo4j database session
        group_id: Group ID to fetch top items for
        
    Returns:
        Dictionary containing top facts, emotions, and entities
    """
    # Top facts
    result_top_facts = await session.run(
        """
        MATCH (fact_node:Fact)<-[r:IS_FACT]-(:Episodic)
        WHERE r.group_id = $group_id
        WITH fact_node.text AS text, MAX(fact_node.count) AS count
        ORDER BY count DESC LIMIT 5
        RETURN collect({text: text, count: count}) AS top_facts
        """,
        {"group_id": group_id}
    )
    record_top_facts = await result_top_facts.single()
    top_facts = record_top_facts["top_facts"] if record_top_facts else []
    
    # Top emotions
    result_top_emotions = await session.run(
        """
        MATCH (emotion_node:Emotion)<-[r:HAS_EMOTION]-(:Episodic)
        WHERE r.group_id = $group_id
        WITH emotion_node.text AS text, MAX(emotion_node.count) AS count
        ORDER BY count DESC LIMIT 5
        RETURN collect({text: text, count: count}) AS top_emotions
        """,
        {"group_id": group_id}
    )
    record_top_emotions = await result_top_emotions.single()
    top_emotions = record_top_emotions["top_emotions"] if record_top_emotions else []
    
    # Top entities
    result_top_entities = await session.run(
        """
        MATCH (entity_node:Entity)<-[r:HAS_ENTITY]-(:Episodic)
        WHERE r.group_id = $group_id
        WITH entity_node.text AS text, MAX(entity_node.count) AS count
        ORDER BY count DESC LIMIT 5
        RETURN collect({text: text, count: count}) AS top_entities
        """,
        {"group_id": group_id}
    )
    record_top_entities = await result_top_entities.single()
    top_entities = record_top_entities["top_entities"] if record_top_entities else []
    
    return {
        "top_facts": top_facts,
        "top_emotions": top_emotions,
        "top_entities": top_entities
    }

async def get_relationships_for_extracted_data(
    session, 
    group_id: str, 
    extracted_facts: List[str], 
    extracted_emotions: List[str], 
    extracted_entities: List[str]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Get relationships specifically for newly extracted facts, emotions, and entities.
    
    Args:
        session: Neo4j database session
        group_id: Group ID to fetch relationships for
        extracted_facts: List of newly extracted facts
        extracted_emotions: List of newly extracted emotions  
        extracted_entities: List of newly extracted entities
        
    Returns:
        Dictionary containing relationships between newly extracted items
    """
    results = {
        "new_facts_to_entities": [],
        "new_facts_to_emotions": [],
        "new_emotions_to_entities": [],
        "all_related_to_new_extractions": []
    }
    
    if not any([extracted_facts, extracted_emotions, extracted_entities]):
        return results
    
    # Build lists for Cypher queries
    all_extracted_texts = extracted_facts + extracted_emotions + extracted_entities
    
    # Get relationships where newly extracted items are involved
    if all_extracted_texts:
        result_new_relationships = await session.run(
            """
            WITH $extracted_texts AS new_items
            MATCH (source_node)<-[r1]-(ep:Episodic)-[r2]->(target_node)
            WHERE r1.group_id = $group_id AND r2.group_id = $group_id
                AND (source_node.text IN new_items OR target_node.text IN new_items)
            RETURN DISTINCT {
                source_type: labels(source_node)[0],
                source_text: source_node.text,
                target_type: labels(target_node)[0], 
                target_text: target_node.text,
                episodic_uuid: ep.uuid,
                is_source_new: source_node.text IN new_items,
                is_target_new: target_node.text IN new_items
            } AS relationship
            """,
            {"group_id": group_id, "extracted_texts": all_extracted_texts}
        )
        
        records = await result_new_relationships.data()
        for record in records:
            rel = record["relationship"]
            results["all_related_to_new_extractions"].append(rel)
            
            # Categorize by type combinations
            source_type = rel["source_type"]
            target_type = rel["target_type"]
            
            if source_type == "Fact" and target_type == "Entity":
                results["new_facts_to_entities"].append(rel)
            elif source_type == "Fact" and target_type == "Emotion":  
                results["new_facts_to_emotions"].append(rel)
            elif source_type == "Emotion" and target_type == "Entity":
                results["new_emotions_to_entities"].append(rel)
    
    return results

async def get_current_message_data(
    session, 
    message_uuid: str,
    group_id: str
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Get facts, emotions, and entities specifically associated with the current message.
    
    Args:
        session: Neo4j database session
        message_uuid: UUID of the current message
        group_id: Group ID for filtering
        
    Returns:
        Dictionary containing facts, emotions, and entities for the current message    """
    results = {
        "message_facts": [],
        "message_emotions": [], 
        "message_entities": []
    }
    
    # Get facts associated with this specific message
    result_facts = await session.run(
        """
        MATCH (ep:Episodic {uuid: $message_uuid})-[rf:IS_FACT]->(fact:Fact)
        WHERE rf.group_id = $group_id
        WITH fact.text AS text, COALESCE(fact.count, 0) AS count
        ORDER BY count DESC
        LIMIT 3
        RETURN collect({
            text: text,
            count: count
        }) AS message_facts
        """,
        {"message_uuid": message_uuid, "group_id": group_id}
    )
    record_facts = await result_facts.single()
    if record_facts and record_facts["message_facts"]:
        results["message_facts"] = record_facts["message_facts"]
    
    # Get emotions associated with this specific message
    result_emotions = await session.run(
        """
        MATCH (ep:Episodic {uuid: $message_uuid})-[re:HAS_EMOTION]->(emotion:Emotion)
        WHERE re.group_id = $group_id
        WITH emotion.text AS text, COALESCE(emotion.count, 0) AS count
        ORDER BY count DESC
        LIMIT 3
        RETURN collect({
            text: text,
            count: count
        }) AS message_emotions
        """,
        {"message_uuid": message_uuid, "group_id": group_id}
    )
    record_emotions = await result_emotions.single()
    if record_emotions and record_emotions["message_emotions"]:
        results["message_emotions"] = record_emotions["message_emotions"]    # Get entities associated with this specific message with related emotions and facts
    result_entities = await session.run(
        """
        MATCH (ep:Episodic {uuid: $message_uuid})-[rent:HAS_ENTITY]->(entity:Entity)
        WHERE rent.group_id = $group_id
        
        OPTIONAL MATCH (entity)<-[re:HAS_ENTITY]-(ep2:Episodic)-[rem:HAS_EMOTION]->(emotion:Emotion)
        WHERE re.group_id = $group_id AND rem.group_id = $group_id
        WITH entity, emotion, COALESCE(emotion.count, 0) AS emotion_count
        ORDER BY emotion_count DESC
        WITH entity, collect(emotion.text)[0..3] AS top_emotions
        
        OPTIONAL MATCH (entity)<-[re2:HAS_ENTITY]-(ep3:Episodic)-[rf:IS_FACT]->(fact:Fact)
        WHERE re2.group_id = $group_id AND rf.group_id = $group_id
        WITH entity, top_emotions, fact, COALESCE(fact.count, 0) AS fact_count
        ORDER BY fact_count DESC
        WITH entity, top_emotions, collect(fact.text)[0..3] AS top_facts
        
        RETURN collect({
            text: entity.text,
            related_emotions: top_emotions,
            related_facts: top_facts
        }) AS message_entities
        """,
        {"message_uuid": message_uuid, "group_id": group_id}
    )
    record_entities = await result_entities.single()
    if record_entities and record_entities["message_entities"]:
        results["message_entities"] = record_entities["message_entities"]
    
    return results
