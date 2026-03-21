from pathlib import Path


def test_standalone_builds_use_repo_root_context_and_bundle_local_graphiti_core():
    mcp_server_dir = Path(__file__).resolve().parents[1]
    docker_dir = mcp_server_dir / 'docker'

    external_compose = (docker_dir / 'docker-compose-neo4j-external.yml').read_text(encoding='utf-8')
    neo4j_compose = (docker_dir / 'docker-compose-neo4j.yml').read_text(encoding='utf-8')
    falkordb_compose = (docker_dir / 'docker-compose-falkordb.yml').read_text(encoding='utf-8')
    build_script = (docker_dir / 'build-standalone.sh').read_text(encoding='utf-8')
    dockerfile = (docker_dir / 'Dockerfile.standalone').read_text(encoding='utf-8')

    assert 'context: ${MCP_BUILD_CONTEXT:-../..}' in external_compose
    assert 'dockerfile: mcp_server/docker/Dockerfile.standalone' in external_compose
    assert 'context: ../..' in neo4j_compose
    assert 'dockerfile: mcp_server/docker/Dockerfile.standalone' in neo4j_compose
    assert 'context: ../..' in falkordb_compose
    assert 'dockerfile: mcp_server/docker/Dockerfile.standalone' in falkordb_compose
    assert '-f mcp_server/docker/Dockerfile.standalone' in build_script
    assert '../..' in build_script
    assert 'COPY mcp_server/main.py ./' in dockerfile
    assert 'COPY mcp_server/src/ ./src/' in dockerfile
    assert 'COPY mcp_server/config/ ./config/' in dockerfile
    assert 'COPY graphiti_core/ /app/graphiti_core/' in dockerfile
