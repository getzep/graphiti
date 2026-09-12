"""Route MCP operations to the FalkorDB graph that owns their data."""

from __future__ import annotations

from copy import copy
from typing import Any

ENTITY_EDGE_UUID_QUERY = 'MATCH ()-[e:RELATES_TO {uuid: $uuid}]->() RETURN e.uuid AS uuid LIMIT 1'
EPISODE_UUID_QUERY = 'MATCH (e:Episodic {uuid: $uuid}) RETURN e.uuid AS uuid LIMIT 1'
EPISODE_UUIDS_QUERY = 'MATCH (e:Episodic) WHERE e.uuid IN $uuids RETURN e.uuid AS uuid'


class AmbiguousUuidError(ValueError):
    """Raised when one UUID exists in multiple tenant graphs."""


def is_falkor(driver: Any) -> bool:
    """Return whether a driver exposes FalkorDB's per-graph API."""
    client = getattr(driver, 'client', None)
    return hasattr(driver, 'clone') and hasattr(client, 'list_graphs')


def driver_for_group(driver: Any, group_id: str) -> Any:
    """Return a non-initializing driver view bound to one FalkorDB graph."""
    if getattr(driver, '_database', None) == group_id:
        return driver
    scoped_driver = copy(driver)
    scoped_driver._database = group_id
    return scoped_driver


async def driver_for_uuid(driver: Any, match_cypher: str, uuid: str) -> tuple[Any, str | None]:
    """Find the sole FalkorDB graph containing a UUID."""
    match: tuple[Any, str] | None = None
    for group_id in await driver.client.list_graphs():
        scoped_driver = driver_for_group(driver, group_id)
        records, _, _ = await scoped_driver.execute_query(match_cypher, uuid=uuid, routing_='r')
        if not records:
            continue
        if match is not None:
            raise AmbiguousUuidError(
                f'UUID {uuid} exists in multiple FalkorDB graphs: {match[1]}, {group_id}'
            )
        match = scoped_driver, group_id
    return match if match is not None else (driver, None)


async def drivers_for_uuids(
    driver: Any, match_cypher: str, uuids: list[str]
) -> dict[str, tuple[Any, list[str]]]:
    """Resolve UUIDs to tenant graphs with one query per graph."""
    requested_uuids = list(dict.fromkeys(uuids))
    owners: dict[str, str] = {}
    buckets: dict[str, tuple[Any, list[str]]] = {}

    for group_id in await driver.client.list_graphs():
        scoped_driver = driver_for_group(driver, group_id)
        records, _, _ = await scoped_driver.execute_query(
            match_cypher, uuids=requested_uuids, routing_='r'
        )
        matched_uuids = {record['uuid'] for record in records}
        for uuid in requested_uuids:
            if uuid not in matched_uuids:
                continue
            if uuid in owners:
                raise AmbiguousUuidError(
                    f'UUID {uuid} exists in multiple FalkorDB graphs: {owners[uuid]}, {group_id}'
                )
            owners[uuid] = group_id
            buckets.setdefault(group_id, (scoped_driver, []))[1].append(uuid)

    return buckets


def client_bound_to(client: Any, driver: Any) -> Any:
    """Return a shallow Graphiti client view bound to a scoped driver."""
    scoped_client = copy(client)
    scoped_client.driver = driver
    return scoped_client
