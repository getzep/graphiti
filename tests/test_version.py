from importlib.metadata import PackageNotFoundError, version

import graphiti_core


def test_graphiti_core_exposes_distribution_version():
    try:
        expected_version = version('graphiti-core')
    except PackageNotFoundError:
        expected_version = 'unknown'

    assert graphiti_core.__version__ == expected_version
