EPISODIC_EDGE_SAVE = """
        MATCH (episode:Episodic {uuid: $episode_uuid}) 
        MATCH (node:Entity {uuid: $entity_uuid}) 
        MERGE (episode)-[r:MENTIONS {uuid: $uuid}]->(node)
        SET r = {uuid: $uuid, group_id: $group_id, created_at: $created_at}
        RETURN r.uuid AS uuid"""

ENTITY_EDGE_SAVE = """
        MATCH (source:Entity {uuid: $source_uuid}) 
        MATCH (target:Entity {uuid: $target_uuid}) 
        MERGE (source)-[r:RELATES_TO {uuid: $uuid}]->(target)
        SET r = {uuid: $uuid, name: $name, group_id: $group_id, fact: $fact, episodes: $episodes, 
        created_at: $created_at, expired_at: $expired_at, valid_at: $valid_at, invalid_at: $invalid_at}
        WITH r CALL db.create.setRelationshipVectorProperty(r, "fact_embedding", $fact_embedding)
        RETURN r.uuid AS uuid"""

COMMUNITY_EDGE_SAVE = """
        MATCH (community:Community {uuid: $community_uuid}) 
        MATCH (node:Entity | Community {uuid: $entity_uuid}) 
        MERGE (community)-[r:HAS_MEMBER {uuid: $uuid}]->(node)
        SET r = {uuid: $uuid, group_id: $group_id, created_at: $created_at}
        RETURN r.uuid AS uuid"""
