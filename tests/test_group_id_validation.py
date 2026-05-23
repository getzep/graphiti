from graphiti_core.driver.driver import GraphProvider
from graphiti_core.driver.falkordb_driver import FalkorDriver
from graphiti_core.helpers import get_default_group_id, validate_group_id


def test_falkor_default_group_id_is_valid():
    group_id = get_default_group_id(GraphProvider.FALKORDB)

    assert group_id == FalkorDriver.default_group_id
    assert validate_group_id(group_id)
