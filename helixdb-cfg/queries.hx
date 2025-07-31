// #########################################################
//                         Entity
// #########################################################

QUERY createEntity (name: String, name_embedding: [F64], group_id: String, summary: String, created_at: Date, labels: [String], attributes: String, uuid: String) =>
    entity <- AddN<Entity>({name: name, group_id: group_id, summary: summary, created_at: created_at, labels: labels, attributes: attributes, uuid: uuid})
    embedding <- AddV<Entity_Embedding>(name_embedding, {name_embedding: name_embedding})
    edge <- AddE<Entity_to_Embedding>({group_id: group_id})::From(entity)::To(embedding)
    RETURN entity

QUERY updateEntity (entity_id: ID, name: String, name_embedding: [F64], group_id: String, summary: String, created_at: Date, labels: [String], attributes: String, uuid: String) =>
    entity <- N<Entity>(entity_id)::UPDATE({name: name, group_id: group_id, summary: summary, created_at: created_at, labels: labels, attributes: attributes, uuid: uuid})
    DROP N<Entity>(entity_id)::Out<Entity_to_Embedding>
    embedding <- AddV<Entity_Embedding>(name_embedding, {name_embedding: name_embedding})
    edge <- AddE<Entity_to_Embedding>({group_id: group_id})::From(entity)::To(embedding)
    RETURN entity

QUERY getEntity (uuid: String) =>
    entity <- N<Entity>({uuid: uuid})
    RETURN entity

QUERY getEntitiesbyGroup (group_id: String) =>
    entities <- N<Entity>::WHERE(_::{group_id}::EQ(group_id))
    RETURN entities

QUERY getEntitiesbyGroupLimit (group_id: String, limit: I64) =>
    entities <- N<Entity>::WHERE(_::{group_id}::EQ(group_id))::RANGE(0, limit)
    RETURN entities

QUERY loadEntityEmbedding (uuid: String) =>
    embedding <- N<Entity>({uuid: uuid})::Out<Entity_to_Embedding>
    RETURN embedding

QUERY deleteEntity (uuid: String) =>
    DROP N<Entity>({uuid: uuid})::Out<Entity_to_Embedding>
    DROP N<Entity>({uuid: uuid})
    RETURN "SUCCESS"

// #########################################################
//                      Entity Search
// #########################################################

QUERY getMentionedEntities (episode_uuid: String) =>
    entities <- N<Episode>({uuid: episode_uuid})::Out<Episode_Entity>
    RETURN entities

QUERY getEntityEpisodeCount (entity_uuid: String) =>
    episode_count <- N<Entity>({uuid: entity_uuid})::In<Episode_Entity>::COUNT
    RETURN episode_count

// #########################################################
//                           Fact
// #########################################################

QUERY createFact (name: String, fact: String, fact_embedding: [F64], group_id: String, created_at: Date, source_uuid: String, target_uuid: String, episodes: [String], valid_at: Date, invalid_at: Date, expired_at: Date, attributes: String, uuid: String) =>
    fact_node <- AddN<Fact>({name: name, fact: fact, group_id: group_id, created_at: created_at, episodes: episodes, valid_at: valid_at, invalid_at: invalid_at, expired_at: expired_at, attributes: attributes, uuid: uuid})

    embedding <- AddV<Fact_Embedding>(fact_embedding, {fact_embedding: fact_embedding})
    edge <- AddE<Fact_to_Embedding>({group_id: group_id})::From(fact_node)::To(embedding)

    source <- N<Entity>({uuid: source_uuid})
    target <- N<Entity>({uuid: target_uuid})
    source_fact_edge <- AddE<Entity_Fact>({group_id: group_id})::From(source)::To(fact_node)
    fact_target_edge <- AddE<Fact_Entity>({group_id: group_id})::From(fact_node)::To(target)
    RETURN fact_node

QUERY updateFact (fact_id: ID, name: String, fact: String, fact_embedding: [F64], group_id: String, created_at: Date, source_uuid: String, target_uuid: String, episodes: [String], valid_at: Date, invalid_at: Date, expired_at: Date, attributes: String, uuid: String) =>
    fact_node <- N<Fact>(fact_id)::UPDATE({name: name, fact: fact, group_id: group_id, created_at: created_at, episodes: episodes, valid_at: valid_at, invalid_at: invalid_at, expired_at: expired_at, attributes: attributes, uuid: uuid})

    DROP N<Fact>(fact_id)::Out<Fact_to_Embedding>
    embedding <- AddV<Fact_Embedding>(fact_embedding, {fact_embedding: fact_embedding})
    edge <- AddE<Fact_to_Embedding>({group_id: group_id})::From(fact_node)::To(embedding)

    DROP N<Fact>(fact_id)::InE<Entity_Fact>
    DROP N<Fact>(fact_id)::OutE<Fact_Entity>
    source <- N<Entity>({uuid: source_uuid})
    target <- N<Entity>({uuid: target_uuid})
    source_fact_edge <- AddE<Entity_Fact>({group_id: group_id})::From(source)::To(fact_node)
    fact_target_edge <- AddE<Fact_Entity>({group_id: group_id})::From(fact_node)::To(target)
    RETURN fact_node

QUERY getFact (uuid: String) =>
    fact <- N<Fact>({uuid: uuid})
    source <- fact::In<Entity_Fact>
    target <- fact::Out<Fact_Entity>
    RETURN fact, source, target

QUERY getFactsbyGroup (group_id: String) =>
    facts <- N<Fact>::WHERE(_::{group_id}::EQ(group_id))
    sources <- facts::In<Entity_Fact>
    targets <- facts::Out<Fact_Entity>
    RETURN facts, sources, targets

QUERY getFactsbyGroupLimit (group_id: String, limit: I64) =>
    facts <- N<Fact>::WHERE(_::{group_id}::EQ(group_id))::RANGE(0, limit)
    sources <- facts::In<Entity_Fact>
    targets <- facts::Out<Fact_Entity>
    RETURN facts, sources, targets

QUERY getFactsbyEntity (uuid: String) =>
    facts <- N<Entity>({uuid: uuid})::Out<Entity_Fact>
    sources <- facts::In<Entity_Fact>
    targets <- facts::Out<Fact_Entity>
    RETURN facts, sources, targets

QUERY loadFactEmbedding (uuid: String) =>
    embedding <- N<Fact>({uuid: uuid})::Out<Fact_to_Embedding>
    RETURN embedding

QUERY deleteFact (uuid: String) =>
    DROP N<Fact>({uuid: uuid})::Out<Fact_to_Embedding>
    DROP N<Fact>({uuid: uuid})
    RETURN "SUCCESS"

// #########################################################
//                      Fact Operations
// #########################################################

QUERY checkDuplicateFact (source_uuid: String, target_uuid: String) =>
    fact <- N<Fact>::WHERE(
        AND(
            EXISTS(_::In<Entity_Fact>::WHERE(_::{uuid}::EQ(source_uuid))),
            EXISTS(_::Out<Fact_Entity>::WHERE(_::{uuid}::EQ(target_uuid))),
            _::{name}::EQ("IS_DUPLICATE_OF")
        )
    )
    source <- N<Entity>({uuid: source_uuid})
    target <- N<Entity>({uuid: target_uuid})
    RETURN fact, source, target

// #########################################################
//                          Episode
// #########################################################

QUERY createEpisode (name: String, group_id: String, source_description: String, content: String, entity_edges: [String], created_at: Date, source: String, valid_at: Date, labels: [String], uuid: String) =>
    episode <- AddN<Episode>({name: name, group_id: group_id, source_description: source_description, content: content, entity_edges: entity_edges, created_at: created_at, source: source, valid_at: valid_at, labels: labels, uuid: uuid})
    RETURN episode

QUERY updateEpisode (episode_id: ID, name: String, group_id: String, source_description: String, content: String, entity_edges: [String], created_at: Date, source: String, valid_at: Date, labels: [String], uuid: String) =>
    episode <- N<Episode>(episode_id)::UPDATE({name: name, group_id: group_id, source_description: source_description, content: content, entity_edges: entity_edges, created_at: created_at, source: source, valid_at: valid_at, labels: labels, uuid: uuid})
    RETURN episode

QUERY getEpisode (uuid: String) =>
    episode <- N<Episode>({uuid: uuid})
    RETURN episode

QUERY getEpisodesbyGroup (group_id: String) =>
    episodes <- N<Episode>::WHERE(_::{group_id}::EQ(group_id))
    RETURN episodes

QUERY getEpisodesbyGroupLimit (group_id: String, limit: I64) =>
    episodes <- N<Episode>::WHERE(_::{group_id}::EQ(group_id))::RANGE(0, limit)
    RETURN episodes

QUERY getEpisodebyEntity (uuid: String) =>
    episodes <- N<Entity>({uuid: uuid})::In<Episode_Entity>
    RETURN episodes

QUERY deleteEpisode (uuid: String) =>
    DROP N<Episode>({uuid: uuid})
    RETURN "SUCCESS"

QUERY createEpisodeEdge (uuid: String, episode_uuid: String, entity_uuid: String, group_id: String, created_at: Date) =>
    episode <- N<Episode>({uuid: episode_uuid})
    entity <- N<Entity>({uuid: entity_uuid})
    episode_edge <- AddE<Episode_Entity>({group_id: group_id, created_at: created_at, uuid: uuid})::From(episode)::To(entity)
    RETURN episode_edge

QUERY updateEpisodeEdge (episodeEdge_id: ID, uuid: String, episode_uuid: String, entity_uuid: String, group_id: String, created_at: Date) =>
    DROP E<Episode_Entity>(episodeEdge_id)
    episode <- N<Episode>({uuid: episode_uuid})
    entity <- N<Entity>({uuid: entity_uuid})
    episode_edge <- AddE<Episode_Entity>({group_id: group_id, created_at: created_at, uuid: uuid})::From(episode)::To(entity)
    RETURN episode_edge

QUERY getEpisodeEdge (uuid: String) =>
    episode_edge <- E<Episode_Entity>::WHERE(_::{uuid}::EQ(uuid))
    episode <- episode_edge::FromN
    entity <- episode_edge::ToN
    RETURN episode_edge, episode, entity

QUERY getEpisodeEdgesbyGroup (group_id: String) =>
    episode_edges <- E<Episode_Entity>::WHERE(_::{group_id}::EQ(group_id))
    episodes <- episode_edges::FromN
    entities <- episode_edges::ToN
    RETURN episode_edges, episodes, entities

QUERY getEpisodeEdgesbyGroupLimit (group_id: String, limit: I64) =>
    episode_edges <- E<Episode_Entity>::WHERE(_::{group_id}::EQ(group_id))::RANGE(0, limit)
    episodes <- episode_edges::FromN
    entities <- episode_edges::ToN
    RETURN episode_edges, episodes, entities

QUERY deleteEpisodeEdge (uuid: String) =>
    DROP E<Episode_Entity>::WHERE(_::{uuid}::EQ(uuid))
    RETURN "SUCCESS"

// #########################################################
//                          Community
// #########################################################

QUERY createCommunity (name: String, group_id: String, summary: String, created_at: Date, labels: [String], name_embedding: [F64], uuid: String) =>
    community <- AddN<Community>({name: name, group_id: group_id, summary: summary, created_at: created_at, labels: labels, uuid: uuid})
    embedding <- AddV<Community_Embedding>(name_embedding, {name_embedding: name_embedding})
    edge <- AddE<Community_to_Embedding>({group_id: group_id})::From(community)::To(embedding)
    RETURN community

QUERY updateCommunity (community_id: ID, name: String, group_id: String, summary: String, created_at: Date, labels: [String], name_embedding: [F64], uuid: String) =>
    community <- N<Community>(community_id)::UPDATE({name: name, group_id: group_id, summary: summary, created_at: created_at, labels: labels, uuid: uuid})
    DROP N<Community>(community_id)::Out<Community_to_Embedding>
    embedding <- AddV<Community_Embedding>(name_embedding, {name_embedding: name_embedding})
    edge <- AddE<Community_to_Embedding>({group_id: group_id})::From(community)::To(embedding)
    RETURN community

QUERY getCommunity (uuid: String) =>
    community <- N<Community>({uuid: uuid})
    RETURN community

QUERY getCommunitiesbyGroup (group_id: String) =>
    communities <- N<Community>::WHERE(_::{group_id}::EQ(group_id))
    RETURN communities

QUERY getCommunitiesbyGroupLimit (group_id: String, limit: I64) =>
    communities <- N<Community>::WHERE(_::{group_id}::EQ(group_id))::RANGE(0, limit)
    RETURN communities

QUERY loadCommunityEmbedding (uuid: String) =>
    embedding <- N<Community>({uuid: uuid})::Out<Community_to_Embedding>
    RETURN embedding

QUERY deleteCommunity (uuid: String) =>
    DROP N<Community>({uuid: uuid})::Out<Community_to_Embedding>
    DROP N<Community>({uuid: uuid})
    RETURN "SUCCESS"

QUERY createCommunityEdge (community_uuid: String, entity_uuid: String, group_id: String, created_at: Date, uuid: String) =>
    community <- N<Community>({uuid: community_uuid})
    entity <- N<Entity>({uuid: entity_uuid})
    community_edge <- AddE<Community_Entity>({group_id: group_id, created_at: created_at, uuid: uuid})::From(community)::To(entity)
    RETURN community_edge

QUERY updateCommunityEdge (communityEdge_id: ID, community_uuid: String, entity_uuid: String, group_id: String, created_at: Date, uuid: String) =>
    DROP E<Community_Entity>(communityEdge_id)
    community <- N<Community>({uuid: community_uuid})
    entity <- N<Entity>({uuid: entity_uuid})
    community_edge <- AddE<Community_Entity>({group_id: group_id, created_at: created_at, uuid: uuid})::From(community)::To(entity)
    RETURN community_edge

QUERY getCommunityEdge (uuid: String) =>
    community_edge <- E<Community_Entity>::WHERE(_::{uuid}::EQ(uuid))
    community <- community_edge::FromN
    entity <- community_edge::ToN
    RETURN community_edge, community, entity

QUERY getCommunityEdgesbyGroup (group_id: String) =>
    community_edges <- E<Community_Entity>::WHERE(_::{group_id}::EQ(group_id))
    communities <- community_edges::FromN
    entities <- community_edges::ToN
    RETURN community_edges, communities, entities

QUERY getCommunityEdgesbyGroupLimit (group_id: String, limit: I64) =>
    community_edges <- E<Community_Entity>::WHERE(_::{group_id}::EQ(group_id))::RANGE(0, limit)
    communities <- community_edges::FromN
    entities <- community_edges::ToN
    RETURN community_edges, communities, entities

QUERY deleteCommunityEdge (uuid: String) =>
    DROP E<Community_Entity>::WHERE(_::{uuid}::EQ(uuid))
    RETURN "SUCCESS"

// #########################################################
//                    Community Search
// #########################################################

QUERY getCommunitybyEntity (entity_uuid: String) =>
    communities <- N<Entity>({uuid: entity_uuid})::In<Community_Entity>
    RETURN communities

// #########################################################
//                          Global
// #########################################################

QUERY deleteGroup (group_id: String) =>
    DROP N<Entity>::WHERE(_::{group_id}::EQ(group_id))
    DROP N<Episode>::WHERE(_::{group_id}::EQ(group_id))
    DROP N<Community>::WHERE(_::{group_id}::EQ(group_id))
    DROP N<Fact>::WHERE(_::{group_id}::EQ(group_id))
    RETURN "SUCCESS"