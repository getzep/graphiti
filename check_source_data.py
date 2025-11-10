#!/usr/bin/env python3
"""Check what's in the source database."""

from neo4j import GraphDatabase
import os

NEO4J_URI = "bolt://192.168.1.25:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = '!"MiTa1205'

SOURCE_DATABASE = "neo4j"
SOURCE_GROUP_ID = "lvarming73"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

print("=" * 70)
print("Checking Source Database")
print("=" * 70)

with driver.session(database=SOURCE_DATABASE) as session:
    # Check total nodes
    result = session.run("""
        MATCH (n {group_id: $group_id})
        RETURN count(n) as total
    """, group_id=SOURCE_GROUP_ID)

    total = result.single()['total']
    print(f"\n✓ Total nodes with group_id '{SOURCE_GROUP_ID}': {total}")

    # Check date range
    result = session.run("""
        MATCH (n:Episodic {group_id: $group_id})
        WHERE n.created_at IS NOT NULL
        RETURN
            min(n.created_at) as earliest,
            max(n.created_at) as latest,
            count(n) as total
    """, group_id=SOURCE_GROUP_ID)

    dates = result.single()
    if dates and dates['total'] > 0:
        print(f"\n✓ Episodic date range:")
        print(f"   Earliest: {dates['earliest']}")
        print(f"   Latest: {dates['latest']}")
        print(f"   Total episodes: {dates['total']}")
    else:
        print("\n⚠️  No episodic nodes with dates found")

    # Sample episodic nodes by date
    result = session.run("""
        MATCH (n:Episodic {group_id: $group_id})
        RETURN n.name as name, n.created_at as created_at
        ORDER BY n.created_at
        LIMIT 10
    """, group_id=SOURCE_GROUP_ID)

    print(f"\n✓ Oldest episodic nodes:")
    for record in result:
        print(f"   - {record['name']}: {record['created_at']}")

    # Check for other group_ids in neo4j database
    result = session.run("""
        MATCH (n)
        WHERE n.group_id IS NOT NULL
        RETURN DISTINCT n.group_id as group_id, count(n) as count
        ORDER BY count DESC
    """)

    print(f"\n✓ All group_ids in '{SOURCE_DATABASE}' database:")
    for record in result:
        print(f"   {record['group_id']}: {record['count']} nodes")

driver.close()
print("\n" + "=" * 70)
