#!/usr/bin/env python3
"""Verify migration data in Neo4j."""

from neo4j import GraphDatabase
import os
import json

NEO4J_URI = "bolt://192.168.1.25:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = '!"MiTa1205'

TARGET_DATABASE = "graphiti"
TARGET_GROUP_ID = "6910959f2128b5c4faa22283"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

print("=" * 70)
print("Verifying Migration Data")
print("=" * 70)

with driver.session(database=TARGET_DATABASE) as session:
    # Check total nodes
    result = session.run("""
        MATCH (n {group_id: $group_id})
        RETURN count(n) as total
    """, group_id=TARGET_GROUP_ID)

    total = result.single()['total']
    print(f"\n✓ Total nodes with group_id '{TARGET_GROUP_ID}': {total}")

    # Check node labels and properties
    result = session.run("""
        MATCH (n {group_id: $group_id})
        RETURN DISTINCT labels(n) as labels, count(*) as count
        ORDER BY count DESC
    """, group_id=TARGET_GROUP_ID)

    print(f"\n✓ Node types:")
    for record in result:
        labels = ':'.join(record['labels'])
        count = record['count']
        print(f"   {labels}: {count}")

    # Sample some episodic nodes
    result = session.run("""
        MATCH (n:Episodic {group_id: $group_id})
        RETURN n.uuid as uuid, n.name as name, n.content as content, n.created_at as created_at
        LIMIT 5
    """, group_id=TARGET_GROUP_ID)

    print(f"\n✓ Sample Episodic nodes:")
    episodes = list(result)
    if episodes:
        for record in episodes:
            print(f"   - {record['name']}")
            print(f"     UUID: {record['uuid']}")
            print(f"     Created: {record['created_at']}")
            print(f"     Content: {record['content'][:100] if record['content'] else 'None'}...")
    else:
        print("   ⚠️  No episodic nodes found!")

    # Sample some entity nodes
    result = session.run("""
        MATCH (n:Entity {group_id: $group_id})
        RETURN n.uuid as uuid, n.name as name, labels(n) as labels, n.summary as summary
        LIMIT 10
    """, group_id=TARGET_GROUP_ID)

    print(f"\n✓ Sample Entity nodes:")
    entities = list(result)
    if entities:
        for record in entities:
            labels = ':'.join(record['labels'])
            print(f"   - {record['name']} ({labels})")
            print(f"     UUID: {record['uuid']}")
            if record['summary']:
                print(f"     Summary: {record['summary'][:80]}...")
    else:
        print("   ⚠️  No entity nodes found!")

    # Check relationships
    result = session.run("""
        MATCH (n {group_id: $group_id})-[r]->(m {group_id: $group_id})
        RETURN type(r) as rel_type, count(*) as count
        ORDER BY count DESC
        LIMIT 10
    """, group_id=TARGET_GROUP_ID)

    print(f"\n✓ Relationship types:")
    rels = list(result)
    if rels:
        for record in rels:
            print(f"   {record['rel_type']}: {record['count']}")
    else:
        print("   ⚠️  No relationships found!")

    # Check if nodes have required properties
    result = session.run("""
        MATCH (n:Episodic {group_id: $group_id})
        RETURN
            count(n) as total,
            count(n.uuid) as has_uuid,
            count(n.name) as has_name,
            count(n.content) as has_content,
            count(n.created_at) as has_created_at,
            count(n.valid_at) as has_valid_at
    """, group_id=TARGET_GROUP_ID)

    props = result.single()
    print(f"\n✓ Episodic node properties:")
    print(f"   Total: {props['total']}")
    print(f"   Has uuid: {props['has_uuid']}")
    print(f"   Has name: {props['has_name']}")
    print(f"   Has content: {props['has_content']}")
    print(f"   Has created_at: {props['has_created_at']}")
    print(f"   Has valid_at: {props['has_valid_at']}")

    # Check Entity properties
    result = session.run("""
        MATCH (n:Entity {group_id: $group_id})
        RETURN
            count(n) as total,
            count(n.uuid) as has_uuid,
            count(n.name) as has_name,
            count(n.summary) as has_summary,
            count(n.created_at) as has_created_at
    """, group_id=TARGET_GROUP_ID)

    props = result.single()
    print(f"\n✓ Entity node properties:")
    print(f"   Total: {props['total']}")
    print(f"   Has uuid: {props['has_uuid']}")
    print(f"   Has name: {props['has_name']}")
    print(f"   Has summary: {props['has_summary']}")
    print(f"   Has created_at: {props['has_created_at']}")

driver.close()
print("\n" + "=" * 70)
