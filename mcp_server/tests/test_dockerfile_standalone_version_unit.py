import re
import tomllib
from pathlib import Path


def _version_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split('.'))


def test_standalone_dockerfile_default_graphiti_core_version_is_not_stale():
    mcp_server_dir = Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads((mcp_server_dir / 'pyproject.toml').read_text(encoding='utf-8'))
    dockerfile = (mcp_server_dir / 'docker' / 'Dockerfile.standalone').read_text(encoding='utf-8')

    dependency = next(
        item
        for item in pyproject['project']['dependencies']
        if item.startswith('graphiti-core[')
    )
    required_match = re.search(r'>=\s*([0-9]+\.[0-9]+\.[0-9]+)', dependency)
    dockerfile_match = re.search(
        r'^ARG GRAPHITI_CORE_VERSION=([0-9]+\.[0-9]+\.[0-9]+)$',
        dockerfile,
        re.MULTILINE,
    )

    assert required_match is not None
    assert dockerfile_match is not None
    assert _version_tuple(dockerfile_match.group(1)) >= _version_tuple(required_match.group(1))
