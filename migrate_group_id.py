#!/usr/bin/env python3
"""
Migrate Graphiti data between databases and group_ids.

Usage:
    python migrate_group_id.py

This script migrates data from:
    Source: neo4j database, group_id='lvarming73'
    Target: graphiti database, group_id='6910959f2128b5c4faa22283'
"""

from neo4j import GraphDatabase
import os


# Configuration
NEO4J_URI = "bolt://192.168.1.25:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", '!"MiTa1205')

SOURCE_DATABASE = "neo4j"
SOURCE_GROUP_ID = "lvarming73"

TARGET_DATABASE = "graphiti"
TARGET_GROUP_ID = "6910959f2128b5c4faa22283"


def migrate_data():
    """Migrate all nodes and relationships from source to target."""

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    try:
        # Step 1: Export data from source database
        print(f"\n📤 Exporting data from {SOURCE_DATABASE} database (group_id: {SOURCE_GROUP_ID})...")

        with driver.session(database=SOURCE_DATABASE) as session:
            # Get all nodes with the source group_id
            nodes_result = session.run("""
                MATCH (n {group_id: $group_id})
                RETURN
                    id(n) as old_id,
                    labels(n) as labels,
                    properties(n) as props
                ORDER BY old_id
            """, group_id=SOURCE_GROUP_ID)

            nodes = list(nodes_result)
            print(f"   Found {len(nodes)} nodes to migrate")

            if len(nodes) == 0:
                print("   ⚠️  No nodes found. Nothing to migrate.")
                return

            # Get all relationships between nodes with the source group_id
            rels_result = session.run("""
                MATCH (n {group_id: $group_id})-[r]->(m {group_id: $group_id})
                RETURN
                    id(startNode(r)) as from_id,
                    id(endNode(r)) as to_id,
                    type(r) as rel_type,
                    properties(r) as props
            """, group_id=SOURCE_GROUP_ID)

            relationships = list(rels_result)
            print(f"   Found {len(relationships)} relationships to migrate")

        # Step 2: Create ID mapping (old Neo4j internal ID -> new node UUID)
        print(f"\n📥 Importing data to {TARGET_DATABASE} database (group_id: {TARGET_GROUP_ID})...")

        id_mapping = {}

        with driver.session(database=TARGET_DATABASE) as session:
            # Create nodes
            for node in nodes:
                old_id = node['old_id']
                labels = node['labels']
                props = dict(node['props'])

                # Update group_id
                props['group_id'] = TARGET_GROUP_ID

                # Get the uuid if it exists (for tracking)
                node_uuid = props.get('uuid', old_id)

                # Build labels string
                labels_str = ':'.join(labels)

                # Create node
                result = session.run(f"""
                    CREATE (n:{labels_str})
                    SET n = $props
                    RETURN id(n) as new_id, n.uuid as uuid
                """, props=props)

                record = result.single()
                id_mapping[old_id] = record['new_id']

            print(f"   ✅ Created {len(nodes)} nodes")

            # Create relationships
            rel_count = 0
            for rel in relationships:
                from_old_id = rel['from_id']
                to_old_id = rel['to_id']
                rel_type = rel['rel_type']
                props = dict(rel['props']) if rel['props'] else {}

                # Update group_id in relationship properties if it exists
                if 'group_id' in props:
                    props['group_id'] = TARGET_GROUP_ID

                # Get new node IDs
                from_new_id = id_mapping.get(from_old_id)
                to_new_id = id_mapping.get(to_old_id)

                if from_new_id is None or to_new_id is None:
                    print(f"   ⚠️  Skipping relationship: node mapping not found")
                    continue

                # Create relationship
                session.run(f"""
                    MATCH (a), (b)
                    WHERE id(a) = $from_id AND id(b) = $to_id
                    CREATE (a)-[r:{rel_type}]->(b)
                    SET r = $props
                """, from_id=from_new_id, to_id=to_new_id, props=props)

                rel_count += 1

            print(f"   ✅ Created {rel_count} relationships")

        # Step 3: Verify migration
        print(f"\n✅ Migration complete!")
        print(f"\n📊 Verification:")

        with driver.session(database=TARGET_DATABASE) as session:
            # Count nodes in target
            result = session.run("""
                MATCH (n {group_id: $group_id})
                RETURN count(n) as node_count
            """, group_id=TARGET_GROUP_ID)

            target_count = result.single()['node_count']
            print(f"   Target database now has {target_count} nodes with group_id={TARGET_GROUP_ID}")

            # Show node types
            result = session.run("""
                MATCH (n {group_id: $group_id})
                RETURN labels(n) as labels, count(*) as count
                ORDER BY count DESC
            """, group_id=TARGET_GROUP_ID)

            print(f"\n   Node types:")
            for record in result:
                labels = ':'.join(record['labels'])
                count = record['count']
                print(f"      {labels}: {count}")

        print(f"\n🎉 Done! Your data has been migrated successfully.")
        print(f"\nNext steps:")
        print(f"1. Verify the data in Neo4j Browser:")
        print(f"   :use graphiti")
        print(f"   MATCH (n {{group_id: '{TARGET_GROUP_ID}'}}) RETURN n LIMIT 25")
        print(f"2. Test in LibreChat to ensure everything works")
        print(f"3. Once verified, you can delete the old data:")
        print(f"   :use neo4j")
        print(f"   MATCH (n {{group_id: '{SOURCE_GROUP_ID}'}}) DETACH DELETE n")

    finally:
        driver.close()


if __name__ == "__main__":
    print("=" * 70)
    print("Graphiti Data Migration Script")
    print("=" * 70)
    print(f"\nSource: {SOURCE_DATABASE} database, group_id='{SOURCE_GROUP_ID}'")
    print(f"Target: {TARGET_DATABASE} database, group_id='{TARGET_GROUP_ID}'")
    print(f"\nNeo4j URI: {NEO4J_URI}")
    print("=" * 70)

    response = input("\n⚠️  Ready to migrate? This will copy all data. Type 'yes' to continue: ")

    if response.lower() == 'yes':
        migrate_data()
    else:
        print("\n❌ Migration cancelled.")
