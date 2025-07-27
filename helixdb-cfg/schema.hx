// #########################################################
//                         Entity
// #########################################################

N::Entity {
    name: String,
    group_id: String,
    labels: [String],
    created_at: Date DEFAULT NOW,
    summary: String,
    attributes: String
}

E::Entity_to_Embedding {
    From: Entity,
    To: Entity_Embedding,
    Properties: {
        group_id: String
    }
}

V::Entity_Embedding {
    name_embedding: [F64],
}

E::Entity_Fact {
    From: Entity,
    To: Fact,
    Properties: {
        group_id: String,
        created_at: Date DEFAULT NOW
    }
}

// #########################################################
//                           Fact
// #########################################################

N::Fact {
    name: String,
    fact: String,
    group_id: String,
    labels: [String],
    created_at: Date DEFAULT NOW,
    source_id: String,
    target_id: String,
    episodes: [String],
    valid_at: Date DEFAULT NOW,
    invalid_at: Date DEFAULT NOW,
    expired_at: Date DEFAULT NOW,
    attributes: String
}

E::Fact_to_Embedding {
    From: Fact,
    To: Fact_Embedding,
    Properties: {
        group_id: String
    }
}

V::Fact_Embedding {
    fact: [F64],
}

E::Fact_Entity {
    From: Fact,
    To: Entity,
    Properties: {
        group_id: String,
        created_at: Date DEFAULT NOW
    }
}

// #########################################################
//                          Episode
// #########################################################

N::Episode {
    name: String,
    group_id: String,
    labels: [String],
    created_at: Date DEFAULT NOW,
    source: String,
    source_description: String,
    content: String,
    valid_at: Date DEFAULT NOW,
    entity_edges: [String]
}

E::Episode_Entity {
    From: Episode,
    To: Entity,
    Properties: {
        group_id: String,
        created_at: Date DEFAULT NOW
    }
}

// #########################################################
//                           Community
// #########################################################

N::Community {
    name: String,
    group_id: String,
    labels: [String],
    created_at: Date DEFAULT NOW,
    summary: String
}

E::Community_to_Embedding {
    From: Community,
    To: Community_Embedding,
    Properties: {
        group_id: String
    }
}

V::Community_Embedding {
    name_embedding: [F64],
}

E::Community_Entity {
    From: Community,
    To: Entity,
    Properties: {
        group_id: String,
        created_at: Date DEFAULT NOW
    }
}

E::Community_Community {
    From: Community,
    To: Community,
    Properties: {
        group_id: String,
        created_at: Date DEFAULT NOW
    }
}

E::Community_Fact {
    From: Community,
    To: Fact,
    Properties: {
        group_id: String,
        created_at: Date DEFAULT NOW
    }
}