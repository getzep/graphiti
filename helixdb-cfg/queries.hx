// #########################################################
//                         Entity
// #########################################################

QUERY createEntity (name: String, name_embedding: [F64], group_id: String, summary: String, created_at: Date, labels: [String], attributes: String) =>
    entity <- AddN<Entity>({name: name, group_id: group_id, summary: summary, created_at: created_at, labels: labels, attributes: attributes})
    embedding <- AddV<Entity_Embedding>(name_embedding, {name_embedding: name_embedding})
    edge <- AddE<Entity_to_Embedding>({group_id: group_id})::From(entity)::To(embedding)
    RETURN entity

QUERY updateEntity (entity_id: ID, name: String, name_embedding: [F64], group_id: String, summary: String, created_at: Date, labels: [String], attributes: String) =>
    entity <- N<Entity>(entity_id)::UPDATE({name: name, group_id: group_id, summary: summary, created_at: created_at, labels: labels, attributes: attributes})
    DROP N<Entity>(entity_id)::Out<Entity_to_Embedding>
    embedding <- AddV<Entity_Embedding>(name_embedding, {name_embedding: name_embedding})
    edge <- AddE<Entity_to_Embedding>({group_id: group_id})::From(entity)::To(embedding)
    RETURN entity

QUERY getEntity (entity_id: ID) =>
    entity <- N<Entity>(entity_id)
    RETURN entity

QUERY getEntitiesbyGroup (group_id: String) =>
    entities <- N<Entity>::WHERE(_::{group_id}::EQ(group_id))
    RETURN entities

QUERY getEntitiesbyGroupLimit (group_id: String, limit: I64) =>
    entities <- N<Entity>::WHERE(_::{group_id}::EQ(group_id))::RANGE(0, limit)
    RETURN entities

QUERY loadEntityEmbedding (entity_id: ID) =>
    embedding <- N<Entity>(entity_id)::Out<Entity_to_Embedding>
    RETURN embedding

QUERY deleteEntity (entity_id: ID) =>
    DROP N<Entity>(entity_id)::Out<Entity_to_Embedding>
    DROP N<Entity>(entity_id)::OutE<Entity_Fact>
    DROP N<Entity>(entity_id)
    RETURN "SUCCESS"

// #########################################################
//                           Fact
// #########################################################

QUERY loadFactEmbedding (fact_id: ID) =>
    embedding <- N<Fact>(fact_id)::Out<Fact_to_Embedding>
    RETURN embedding

QUERY deleteFact (fact_id: ID) =>
    DROP N<Fact>(fact_id)::Out<Fact_to_Embedding>
    DROP N<Fact>(fact_id)::OutE<Fact_Entity>
    DROP N<Fact>(fact_id)
    RETURN "SUCCESS"

// #########################################################
//                          Episode
// #########################################################

QUERY createEpisode (name: String, group_id: String, source_description: String, content: String, entity_edges: [String], created_at: Date, source: String, valid_at: Date, labels: [String]) =>
    episode <- AddN<Episode>({name: name, group_id: group_id, source_description: source_description, content: content, entity_edges: entity_edges, created_at: created_at, source: source, valid_at: valid_at, labels: labels})
    RETURN episode

QUERY updateEpisode (episode_id: ID, name: String, group_id: String, source_description: String, content: String, entity_edges: [String], created_at: Date, source: String, valid_at: Date, labels: [String]) =>
    episode <- N<Episode>(episode_id)::UPDATE({name: name, group_id: group_id, source_description: source_description, content: content, entity_edges: entity_edges, created_at: created_at, source: source, valid_at: valid_at, labels: labels})
    RETURN episode

QUERY getEpisode (episode_id: ID) =>
    episode <- N<Episode>(episode_id)
    RETURN episode

QUERY getEpisodesbyGroup (group_id: String) =>
    episodes <- N<Episode>::WHERE(_::{group_id}::EQ(group_id))
    RETURN episodes

QUERY getEpisodesbyGroupLimit (group_id: String, limit: I64) =>
    episodes <- N<Episode>::WHERE(_::{group_id}::EQ(group_id))::RANGE(0, limit)
    RETURN episodes

QUERY getEpisodebyEntity (entity_id: ID) =>
    episodes <- N<Entity>(entity_id)::In<Episode_Entity>
    RETURN episodes

QUERY deleteEpisode (episode_id: ID) =>
    DROP N<Episode>(episode_id)::OutE<Episode_Entity>
    DROP N<Episode>(episode_id)
    RETURN "SUCCESS"

QUERY createEpisodeEdge (episode_id: ID, entity_id: ID, group_id: String, created_at: Date) =>
    episode <- N<Episode>(episode_id)
    entity <- N<Entity>(entity_id)
    episode_edge <- AddE<Episode_Entity>({group_id: group_id, created_at: created_at})::From(episode)::To(entity)
    RETURN episode_edge

QUERY updateEpisodeEdge (episodeEdge_id: ID, episode_id: ID, entity_id: ID, group_id: String, created_at: Date) =>
    DROP E<Episode_Entity>(episodeEdge_id)
    episode <- N<Episode>(episode_id)
    entity <- N<Entity>(entity_id)
    episode_edge <- AddE<Episode_Entity>({group_id: group_id, created_at: created_at})::From(episode)::To(entity)
    RETURN episode_edge

QUERY getEpisodeEdge (episodeEdge_id: ID) =>
    episode_edge <- E<Episode_Entity>(episodeEdge_id)
    RETURN episode_edge

QUERY getEpisodeEdgesbyGroup (group_id: String) =>
    episode_edges <- E<Episode_Entity>::WHERE(_::{group_id}::EQ(group_id))
    RETURN episode_edges

QUERY getEpisodeEdgesbyGroupLimit (group_id: String, limit: I64) =>
    episode_edges <- E<Episode_Entity>::WHERE(_::{group_id}::EQ(group_id))::RANGE(0, limit)
    RETURN episode_edges

QUERY deleteEpisodeEdge (episodeEdge_id: ID) =>
    DROP E<Episode_Entity>(episodeEdge_id)
    RETURN "SUCCESS"

// #########################################################
//                          Community
// #########################################################

QUERY createCommunity (name: String, group_id: String, summary: String, created_at: Date, labels: [String], name_embedding: [F64]) =>
    community <- AddN<Community>({name: name, group_id: group_id, summary: summary, created_at: created_at, labels: labels})
    embedding <- AddV<Community_Embedding>(name_embedding, {name_embedding: name_embedding})
    edge <- AddE<Community_to_Embedding>({group_id: group_id})::From(community)::To(embedding)
    RETURN community

QUERY updateCommunity (community_id: ID, name: String, group_id: String, summary: String, created_at: Date, labels: [String], name_embedding: [F64]) =>
    community <- N<Community>(community_id)::UPDATE({name: name, group_id: group_id, summary: summary, created_at: created_at, labels: labels})
    DROP N<Community>(community_id)::Out<Community_to_Embedding>
    embedding <- AddV<Community_Embedding>(name_embedding, {name_embedding: name_embedding})
    edge <- AddE<Community_to_Embedding>({group_id: group_id})::From(community)::To(embedding)
    RETURN community

QUERY getCommunity (community_id: ID) =>
    community <- N<Community>(community_id)
    RETURN community

QUERY getCommunitiesbyGroup (group_id: String) =>
    communities <- N<Community>::WHERE(_::{group_id}::EQ(group_id))
    RETURN communities

QUERY getCommunitiesbyGroupLimit (group_id: String, limit: I64) =>
    communities <- N<Community>::WHERE(_::{group_id}::EQ(group_id))::RANGE(0, limit)
    RETURN communities

QUERY loadCommunityEmbedding (community_id: ID) =>
    embedding <- N<Community>(community_id)::Out<Community_to_Embedding>
    RETURN embedding

QUERY deleteCommunity (community_id: ID) =>
    DROP N<Community>(community_id)::Out<Community_to_Embedding>
    DROP N<Community>(community_id)::OutE<Community_Fact>
    DROP N<Community>(community_id)::OutE<Community_Entity>
    DROP N<Community>(community_id)::OutE<Community_Community>
    DROP N<Community>(community_id)
    RETURN "SUCCESS"

// #########################################################
//                          Global
// #########################################################

QUERY deleteGroup(group_id: String) =>
    DROP N<Entity>::WHERE(_::{group_id}::EQ(group_id))
    DROP N<Episode>::WHERE(_::{group_id}::EQ(group_id))
    DROP N<Community>::WHERE(_::{group_id}::EQ(group_id))
    DROP N<Fact>::WHERE(_::{group_id}::EQ(group_id))
    RETURN "SUCCESS"