import asyncio
import logging
from collections import defaultdict

from pydantic import BaseModel

from graphiti_core.driver.driver import GraphDriver, GraphProvider
from graphiti_core.edges import CommunityEdge
from graphiti_core.embedder import EmbedderClient
from graphiti_core.helpers import semaphore_gather
from graphiti_core.llm_client import LLMClient
from graphiti_core.models.nodes.node_db_queries import COMMUNITY_NODE_RETURN
from graphiti_core.nodes import CommunityNode, EntityNode, get_community_node_from_record
from graphiti_core.prompts import prompt_library
from graphiti_core.prompts.summarize_nodes import Summary, SummaryDescription
from graphiti_core.utils.datetime_utils import utc_now
from graphiti_core.utils.maintenance.edge_operations import build_community_edges
from graphiti_core.utils.text_utils import MAX_SUMMARY_CHARS, truncate_at_sentence

MAX_COMMUNITY_BUILD_CONCURRENCY = 10

logger = logging.getLogger(__name__)


class Neighbor(BaseModel):
    node_uuid: str
    edge_count: int


async def _build_group_projection(
    driver: GraphDriver, group_id: str
) -> dict[str, list[Neighbor]]:
    """Fetch the RELATES_TO projection for all entities in a group.

    Returns a mapping from each node's uuid to its list of in-group neighbors
    with edge counts. Used by label propagation and by in-community degree
    computations for sampling.
    """
    projection: dict[str, list[Neighbor]] = {}
    nodes = await EntityNode.get_by_group_ids(driver, [group_id])
    for node in nodes:
        match_query = """
            MATCH (n:Entity {group_id: $group_id, uuid: $uuid})-[e:RELATES_TO]-(m: Entity {group_id: $group_id})
        """
        if driver.provider == GraphProvider.KUZU:
            match_query = """
            MATCH (n:Entity {group_id: $group_id, uuid: $uuid})-[:RELATES_TO]-(e:RelatesToNode_)-[:RELATES_TO]-(m: Entity {group_id: $group_id})
            """
        records, _, _ = await driver.execute_query(
            match_query
            + """
            WITH count(e) AS count, m.uuid AS uuid
            RETURN
                uuid,
                count
            """,
            uuid=node.uuid,
            group_id=group_id,
        )

        projection[node.uuid] = [
            Neighbor(node_uuid=record['uuid'], edge_count=record['count']) for record in records
        ]
    return projection


async def get_community_clusters(
    driver: GraphDriver,
    group_ids: list[str] | None,
    return_projection: bool = False,
) -> list[list[EntityNode]] | tuple[list[list[EntityNode]], dict[str, list[Neighbor]]]:
    """Compute community clusters via label propagation.

    Args:
        driver: Graph driver.
        group_ids: Optional list of group ids to scope clustering. If None,
            all groups are used.
        return_projection: When True, also return the combined projection
            (uuid → neighbors with edge counts) so callers can compute
            in-community degrees without a second pass over the graph.

    Returns:
        By default, just the list of clusters (each a list of EntityNode).
        When return_projection=True, returns (clusters, projection) tuple.
    """
    if driver.graph_operations_interface:
        try:
            clusters = await driver.graph_operations_interface.get_community_clusters(
                driver, group_ids
            )
            if return_projection:
                return clusters, {}
            return clusters
        except NotImplementedError:
            pass

    community_clusters: list[list[EntityNode]] = []
    combined_projection: dict[str, list[Neighbor]] = {}

    if group_ids is None:
        group_id_values, _, _ = await driver.execute_query(
            """
            MATCH (n:Entity)
            WHERE n.group_id IS NOT NULL
            RETURN
                collect(DISTINCT n.group_id) AS group_ids
            """
        )

        group_ids = group_id_values[0]['group_ids'] if group_id_values else []

    for group_id in group_ids:
        projection = await _build_group_projection(driver, group_id)
        if return_projection:
            combined_projection.update(projection)

        cluster_uuids = label_propagation(projection)

        community_clusters.extend(
            list(
                await semaphore_gather(
                    *[EntityNode.get_by_uuids(driver, cluster) for cluster in cluster_uuids]
                )
            )
        )

    if return_projection:
        return community_clusters, combined_projection
    return community_clusters


def label_propagation(projection: dict[str, list[Neighbor]]) -> list[list[str]]:
    # Implement the label propagation community detection algorithm.
    # 1. Start with each node being assigned its own community
    # 2. Each node will take on the community of the plurality of its neighbors
    # 3. Ties are broken by going to the largest community
    # 4. Continue until no communities change during propagation

    community_map = {uuid: i for i, uuid in enumerate(projection.keys())}

    while True:
        no_change = True
        new_community_map: dict[str, int] = {}

        for uuid, neighbors in projection.items():
            curr_community = community_map[uuid]

            community_candidates: dict[int, int] = defaultdict(int)
            for neighbor in neighbors:
                community_candidates[community_map[neighbor.node_uuid]] += neighbor.edge_count
            community_lst = [
                (count, community) for community, count in community_candidates.items()
            ]

            community_lst.sort(reverse=True)
            candidate_rank, community_candidate = community_lst[0] if community_lst else (0, -1)
            if community_candidate != -1 and candidate_rank > 1:
                new_community = community_candidate
            else:
                new_community = max(community_candidate, curr_community)

            new_community_map[uuid] = new_community

            if new_community != curr_community:
                no_change = False

        if no_change:
            break

        community_map = new_community_map

    community_cluster_map = defaultdict(list)
    for uuid, community in community_map.items():
        community_cluster_map[community].append(uuid)

    clusters = [cluster for cluster in community_cluster_map.values()]
    return clusters


async def summarize_pair(llm_client: LLMClient, summary_pair: tuple[str, str]) -> str:
    # Prepare context for LLM
    context = {
        'node_summaries': [{'summary': summary} for summary in summary_pair],
    }

    llm_response = await llm_client.generate_response(
        prompt_library.summarize_nodes.summarize_pair(context),
        response_model=Summary,
        prompt_name='summarize_nodes.summarize_pair',
    )

    pair_summary = llm_response.get('summary', '')

    return truncate_at_sentence(pair_summary, MAX_SUMMARY_CHARS)


async def generate_summary_description(llm_client: LLMClient, summary: str) -> str:
    context = {
        'summary': summary,
    }

    llm_response = await llm_client.generate_response(
        prompt_library.summarize_nodes.summary_description(context),
        response_model=SummaryDescription,
        prompt_name='summarize_nodes.summary_description',
    )

    description = llm_response.get('description', '')

    return description


def _select_representative_members(
    community_cluster: list[EntityNode],
    projection: dict[str, list[Neighbor]] | None,
    sample_size: int,
) -> list[EntityNode]:
    """Pick the top-K members most likely to characterize the community.

    Scoring key (descending): in-community weighted degree, then summary
    length, then name for deterministic ties. In-community degree uses the
    projection we already computed during clustering — no extra queries.

    When no projection is available (e.g. the graph_operations_interface
    returned clusters directly), falls back to summary length only.
    """
    if len(community_cluster) <= sample_size:
        return community_cluster

    member_uuids = {m.uuid for m in community_cluster}

    def in_community_degree(entity: EntityNode) -> int:
        if not projection:
            return 0
        neighbors = projection.get(entity.uuid, [])
        return sum(n.edge_count for n in neighbors if n.node_uuid in member_uuids)

    scored = sorted(
        community_cluster,
        key=lambda e: (in_community_degree(e), len(e.summary or ''), e.name),
        reverse=True,
    )
    return scored[:sample_size]


async def build_community(
    llm_client: LLMClient,
    community_cluster: list[EntityNode],
    *,
    projection: dict[str, list[Neighbor]] | None = None,
    sample_size: int | None = None,
) -> tuple[CommunityNode, list[CommunityEdge]]:
    """Build a community node from its member entities.

    Args:
        llm_client: LLM used to summarize pairs and generate the final name.
        community_cluster: Full list of member entities.
        projection: Optional {uuid -> neighbors} projection from the clustering
            step. Used to rank members by in-community weighted degree when
            sampling.
        sample_size: If set, only the top-K most representative members
            participate in the binary summary merge. The community still
            contains all members in its HAS_MEMBER edges — sampling only
            affects which summaries are fed into the LLM pipeline. This cuts
            LLM cost from O(N) per community to O(sample_size) and typically
            improves quality because hub nodes carry the community's signal.
    """
    summary_members = (
        _select_representative_members(community_cluster, projection, sample_size)
        if sample_size is not None
        else community_cluster
    )

    summaries = [entity.summary for entity in summary_members]
    length = len(summaries)
    while length > 1:
        odd_one_out: str | None = None
        if length % 2 == 1:
            odd_one_out = summaries.pop()
            length -= 1
        new_summaries: list[str] = list(
            await semaphore_gather(
                *[
                    summarize_pair(llm_client, (str(left_summary), str(right_summary)))
                    for left_summary, right_summary in zip(
                        summaries[: int(length / 2)], summaries[int(length / 2) :], strict=False
                    )
                ]
            )
        )
        if odd_one_out is not None:
            new_summaries.append(odd_one_out)
        summaries = new_summaries
        length = len(summaries)

    summary = truncate_at_sentence(summaries[0], MAX_SUMMARY_CHARS) if summaries else ''
    name = (
        await generate_summary_description(llm_client, summary) if summary else 'community'
    )
    now = utc_now()
    community_node = CommunityNode(
        name=name,
        group_id=community_cluster[0].group_id,
        labels=['Community'],
        created_at=now,
        summary=summary,
    )
    community_edges = build_community_edges(community_cluster, community_node, now)

    logger.debug(
        'Built community %s with %d member edges (summary from %d/%d members)',
        community_node.uuid,
        len(community_edges),
        len(summary_members),
        len(community_cluster),
    )

    return community_node, community_edges


async def build_communities(
    driver: GraphDriver,
    llm_client: LLMClient,
    group_ids: list[str] | None,
    *,
    sample_size: int | None = None,
) -> tuple[list[CommunityNode], list[CommunityEdge]]:
    """Cluster entities into communities and build a summary node for each.

    Args:
        driver: Graph driver.
        llm_client: LLM client for community summarization.
        group_ids: Scope clustering to these group ids (or all if None).
        sample_size: If set, each community's summary is built from only
            the top-K most representative members (by in-community weighted
            degree, then summary length). Reduces LLM cost from O(total nodes)
            to O(num_communities * sample_size). Recommended for graphs
            >10k nodes.
    """
    clusters_result = await get_community_clusters(driver, group_ids, return_projection=True)
    assert isinstance(clusters_result, tuple)
    community_clusters, projection = clusters_result

    semaphore = asyncio.Semaphore(MAX_COMMUNITY_BUILD_CONCURRENCY)

    async def limited_build_community(cluster):
        async with semaphore:
            return await build_community(
                llm_client,
                cluster,
                projection=projection,
                sample_size=sample_size,
            )

    communities: list[tuple[CommunityNode, list[CommunityEdge]]] = list(
        await semaphore_gather(
            *[limited_build_community(cluster) for cluster in community_clusters]
        )
    )

    community_nodes: list[CommunityNode] = []
    community_edges: list[CommunityEdge] = []
    for community in communities:
        community_nodes.append(community[0])
        community_edges.extend(community[1])

    return community_nodes, community_edges


async def remove_communities(driver: GraphDriver):
    if driver.graph_operations_interface:
        try:
            return await driver.graph_operations_interface.remove_communities(driver)
        except NotImplementedError:
            pass

    await driver.execute_query(
        """
        MATCH (c:Community)
        DETACH DELETE c
        """
    )


async def determine_entity_community(
    driver: GraphDriver, entity: EntityNode
) -> tuple[CommunityNode | None, bool]:
    if driver.graph_operations_interface:
        try:
            return await driver.graph_operations_interface.determine_entity_community(
                driver, entity
            )
        except NotImplementedError:
            pass

    # Check if the node is already part of a community
    records, _, _ = await driver.execute_query(
        """
        MATCH (c:Community)-[:HAS_MEMBER]->(n:Entity {uuid: $entity_uuid})
        RETURN
        """
        + COMMUNITY_NODE_RETURN,
        entity_uuid=entity.uuid,
    )

    if len(records) > 0:
        return get_community_node_from_record(records[0]), False

    # If the node has no community, add it to the mode community of surrounding entities
    match_query = """
        MATCH (c:Community)-[:HAS_MEMBER]->(m:Entity)-[:RELATES_TO]-(n:Entity {uuid: $entity_uuid})
    """
    if driver.provider == GraphProvider.KUZU:
        match_query = """
            MATCH (c:Community)-[:HAS_MEMBER]->(m:Entity)-[:RELATES_TO]-(e:RelatesToNode_)-[:RELATES_TO]-(n:Entity {uuid: $entity_uuid})
        """
    records, _, _ = await driver.execute_query(
        match_query
        + """
        RETURN
        """
        + COMMUNITY_NODE_RETURN,
        entity_uuid=entity.uuid,
    )

    communities: list[CommunityNode] = [
        get_community_node_from_record(record) for record in records
    ]

    community_map: dict[str, int] = defaultdict(int)
    for community in communities:
        community_map[community.uuid] += 1

    community_uuid = None
    max_count = 0
    for uuid, count in community_map.items():
        if count > max_count:
            community_uuid = uuid
            max_count = count

    if max_count == 0:
        return None, False

    for community in communities:
        if community.uuid == community_uuid:
            return community, True

    return None, False


async def update_community(
    driver: GraphDriver,
    llm_client: LLMClient,
    embedder: EmbedderClient,
    entity: EntityNode,
) -> tuple[list[CommunityNode], list[CommunityEdge]]:
    community, is_new = await determine_entity_community(driver, entity)

    if community is None:
        return [], []

    new_summary = await summarize_pair(llm_client, (entity.summary, community.summary))
    new_name = await generate_summary_description(llm_client, new_summary)

    community.summary = new_summary
    community.name = new_name

    community_edges = []
    if is_new:
        community_edge = (build_community_edges([entity], community, utc_now()))[0]
        await community_edge.save(driver)
        community_edges.append(community_edge)

    await community.generate_name_embedding(embedder)

    await community.save(driver)

    return [community], community_edges
