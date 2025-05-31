// Neo4j constraints for Graphiti entities

CREATE CONSTRAINT emotion_unique_text IF NOT EXISTS
FOR (e:Emotion)
REQUIRE e.text IS UNIQUE;

CREATE CONSTRAINT fact_unique_text IF NOT EXISTS
FOR (f:Fact)
REQUIRE f.text IS UNIQUE;

CREATE CONSTRAINT entity_unique_text IF NOT EXISTS
FOR (m:Entity)
REQUIRE m.text IS UNIQUE;
