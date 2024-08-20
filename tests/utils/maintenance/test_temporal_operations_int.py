import pytest
import asyncio
from datetime import datetime, timedelta
from core.utils.maintenance.temporal_operations import invalidate_edges, EdgeWithNodes
from core.edges import EntityEdge
from core.nodes import EntityNode
from core.llm_client import LLMClient, OpenAIClient, LLMConfig
from dotenv import load_dotenv
import os

load_dotenv()

# Set up the LLM client (TODO: in the future we might want to run tests using non-openai models as well)
llm_client = OpenAIClient(
    LLMConfig(
        api_key=os.getenv("TEST_OPENAI_API_KEY"),
        model=os.getenv("TEST_OPENAI_MODEL"),
        base_url="https://api.openai.com/v1",
    )
)


# Helper function to create test data
def create_test_data():
    now = datetime.now()

    # Create nodes
    node1 = EntityNode(uuid="1", name="Alice", labels=["Person"], created_at=now)
    node2 = EntityNode(uuid="2", name="Bob", labels=["Person"], created_at=now)

    # Create edges
    edge1 = EntityEdge(
        uuid="e1",
        source_node_uuid="1",
        target_node_uuid="2",
        name="LIKES",
        fact="Alice likes Bob",
        created_at=now - timedelta(days=1),
    )
    edge2 = EntityEdge(
        uuid="e2",
        source_node_uuid="1",
        target_node_uuid="2",
        name="DISLIKES",
        fact="Alice dislikes Bob",
        created_at=now,
    )

    # Create EdgeWithNodes objects
    existing_edge = EdgeWithNodes(edge=edge1, source_node=node1, target_node=node2)
    new_edge = EdgeWithNodes(edge=edge2, source_node=node1, target_node=node2)

    return existing_edge, new_edge


@pytest.mark.asyncio
async def test_invalidate_edges():
    existing_edge, new_edge = create_test_data()

    invalidated_edges = await invalidate_edges(llm_client, [existing_edge], [new_edge])

    assert len(invalidated_edges) == 1
    assert invalidated_edges[0].uuid == existing_edge.edge.uuid
    assert invalidated_edges[0].expired_at is not None


@pytest.mark.asyncio
async def test_invalidate_edges_no_invalidation():
    existing_edge, _ = create_test_data()

    invalidated_edges = await invalidate_edges(llm_client, [existing_edge], [])

    assert len(invalidated_edges) == 0


@pytest.mark.asyncio
async def test_invalidate_edges_multiple_existing():
    existing_edge1, new_edge = create_test_data()
    existing_edge2, _ = create_test_data()
    existing_edge2.edge.uuid = "e3"
    existing_edge2.edge.name = "KNOWS"
    existing_edge2.edge.fact = "Alice knows Bob"

    invalidated_edges = await invalidate_edges(
        llm_client, [existing_edge1, existing_edge2], [new_edge]
    )

    assert len(invalidated_edges) == 1
    assert invalidated_edges[0].uuid == existing_edge1.edge.uuid
    assert invalidated_edges[0].expired_at is not None


# Helper function to create more complex test data
def create_complex_test_data():
    now = datetime.now()

    # Create nodes
    node1 = EntityNode(uuid="1", name="Alice", labels=["Person"], created_at=now)
    node2 = EntityNode(uuid="2", name="Bob", labels=["Person"], created_at=now)
    node3 = EntityNode(uuid="3", name="Charlie", labels=["Person"], created_at=now)
    node4 = EntityNode(
        uuid="4", name="Company XYZ", labels=["Organization"], created_at=now
    )

    # Create edges
    edge1 = EntityEdge(
        uuid="e1",
        source_node_uuid="1",
        target_node_uuid="2",
        name="LIKES",
        fact="Alice likes Bob",
        created_at=now - timedelta(days=5),
    )
    edge2 = EntityEdge(
        uuid="e2",
        source_node_uuid="1",
        target_node_uuid="3",
        name="FRIENDS_WITH",
        fact="Alice is friends with Charlie",
        created_at=now - timedelta(days=3),
    )
    edge3 = EntityEdge(
        uuid="e3",
        source_node_uuid="2",
        target_node_uuid="4",
        name="WORKS_FOR",
        fact="Bob works for Company XYZ",
        created_at=now - timedelta(days=2),
    )

    # Create EdgeWithNodes objects
    existing_edge1 = EdgeWithNodes(edge=edge1, source_node=node1, target_node=node2)
    existing_edge2 = EdgeWithNodes(edge=edge2, source_node=node1, target_node=node3)
    existing_edge3 = EdgeWithNodes(edge=edge3, source_node=node2, target_node=node4)

    return [existing_edge1, existing_edge2, existing_edge3], [
        node1,
        node2,
        node3,
        node4,
    ]


@pytest.mark.asyncio
async def test_invalidate_edges_complex():
    existing_edges, nodes = create_complex_test_data()

    # Create a new edge that contradicts an existing one
    new_edge = EdgeWithNodes(
        edge=EntityEdge(
            uuid="e4",
            source_node_uuid="1",
            target_node_uuid="2",
            name="DISLIKES",
            fact="Alice dislikes Bob",
            created_at=datetime.now(),
        ),
        source_node=nodes[0],
        target_node=nodes[1],
    )

    invalidated_edges = await invalidate_edges(llm_client, existing_edges, [new_edge])

    assert len(invalidated_edges) == 1
    assert invalidated_edges[0].uuid == "e1"
    assert invalidated_edges[0].expired_at is not None


@pytest.mark.asyncio
async def test_invalidate_edges_temporal_update():
    existing_edges, nodes = create_complex_test_data()

    # Create a new edge that updates an existing one with new information
    new_edge = EdgeWithNodes(
        edge=EntityEdge(
            uuid="e5",
            source_node_uuid="2",
            target_node_uuid="4",
            name="LEFT_JOB",
            fact="Bob left his job at Company XYZ",
            created_at=datetime.now(),
        ),
        source_node=nodes[1],
        target_node=nodes[3],
    )

    invalidated_edges = await invalidate_edges(llm_client, existing_edges, [new_edge])

    assert len(invalidated_edges) == 1
    assert invalidated_edges[0].uuid == "e3"
    assert invalidated_edges[0].expired_at is not None


@pytest.mark.asyncio
async def test_invalidate_edges_multiple_invalidations():
    existing_edges, nodes = create_complex_test_data()

    # Create new edges that invalidate multiple existing edges
    new_edge1 = EdgeWithNodes(
        edge=EntityEdge(
            uuid="e6",
            source_node_uuid="1",
            target_node_uuid="2",
            name="ENEMIES_WITH",
            fact="Alice and Bob are now enemies",
            created_at=datetime.now(),
        ),
        source_node=nodes[0],
        target_node=nodes[1],
    )
    new_edge2 = EdgeWithNodes(
        edge=EntityEdge(
            uuid="e7",
            source_node_uuid="1",
            target_node_uuid="3",
            name="ENDED_FRIENDSHIP",
            fact="Alice ended her friendship with Charlie",
            created_at=datetime.now(),
        ),
        source_node=nodes[0],
        target_node=nodes[2],
    )

    invalidated_edges = await invalidate_edges(
        llm_client, existing_edges, [new_edge1, new_edge2]
    )

    assert len(invalidated_edges) == 2
    assert set(edge.uuid for edge in invalidated_edges) == {"e1", "e2"}
    for edge in invalidated_edges:
        assert edge.expired_at is not None


@pytest.mark.asyncio
async def test_invalidate_edges_no_effect():
    existing_edges, nodes = create_complex_test_data()

    # Create a new edge that doesn't invalidate any existing edges
    new_edge = EdgeWithNodes(
        edge=EntityEdge(
            uuid="e8",
            source_node_uuid="3",
            target_node_uuid="4",
            name="APPLIED_TO",
            fact="Charlie applied to Company XYZ",
            created_at=datetime.now(),
        ),
        source_node=nodes[2],
        target_node=nodes[3],
    )

    invalidated_edges = await invalidate_edges(llm_client, existing_edges, [new_edge])

    assert len(invalidated_edges) == 0


@pytest.mark.asyncio
async def test_invalidate_edges_partial_update():
    existing_edges, nodes = create_complex_test_data()

    # Create a new edge that partially updates an existing one
    new_edge = EdgeWithNodes(
        edge=EntityEdge(
            uuid="e9",
            source_node_uuid="2",
            target_node_uuid="4",
            name="CHANGED_POSITION",
            fact="Bob changed his position at Company XYZ",
            created_at=datetime.now(),
        ),
        source_node=nodes[1],
        target_node=nodes[3],
    )

    invalidated_edges = await invalidate_edges(llm_client, existing_edges, [new_edge])

    assert (
        len(invalidated_edges) == 0
    )  # The existing edge is not invalidated, just updated


@pytest.mark.asyncio
async def test_invalidate_edges_empty_inputs():
    invalidated_edges = await invalidate_edges(llm_client, [], [])

    assert len(invalidated_edges) == 0
