from config.schema import DatabaseConfig, DatabaseProvidersConfig, FalkorDBProviderConfig
from services.factories import DatabaseDriverFactory


def test_falkordb_config_preserves_uri_username(monkeypatch):
    monkeypatch.delenv('FALKORDB_URI', raising=False)
    monkeypatch.delenv('FALKORDB_USERNAME', raising=False)
    monkeypatch.delenv('FALKORDB_PASSWORD', raising=False)

    config = DatabaseConfig(
        provider='falkordb',
        providers=DatabaseProvidersConfig(
            falkordb=FalkorDBProviderConfig(
                uri='redis://falkordb:secret@example.cloud:6379',
                database='cloud-db',
            )
        ),
    )

    db_config = DatabaseDriverFactory.create_config(config)

    assert db_config['host'] == 'example.cloud'
    assert db_config['port'] == 6379
    assert db_config['username'] == 'falkordb'
    assert db_config['password'] is None
    assert db_config['database'] == 'cloud-db'


def test_falkordb_group_routing_defaults_to_database(monkeypatch):
    """Default config keeps graph-per-group routing (#1684)."""
    monkeypatch.delenv('FALKORDB_URI', raising=False)

    config = DatabaseConfig(
        provider='falkordb',
        providers=DatabaseProvidersConfig(falkordb=FalkorDBProviderConfig()),
    )

    db_config = DatabaseDriverFactory.create_config(config)

    assert db_config['group_routing'] == 'database'


def test_falkordb_record_group_routing_is_passed_through(monkeypatch):
    """Opting into record routing reaches the driver config unchanged."""
    monkeypatch.delenv('FALKORDB_URI', raising=False)

    config = DatabaseConfig(
        provider='falkordb',
        providers=DatabaseProvidersConfig(
            falkordb=FalkorDBProviderConfig(
                database='graphiti_personal',
                group_routing='record',
            )
        ),
    )

    db_config = DatabaseDriverFactory.create_config(config)

    assert db_config['database'] == 'graphiti_personal'
    assert db_config['group_routing'] == 'record'


def test_falkordb_username_env_overrides_uri(monkeypatch):
    monkeypatch.setenv('FALKORDB_URI', 'redis://uri-user:secret@example.cloud:6380')
    monkeypatch.setenv('FALKORDB_USERNAME', 'env-user')
    monkeypatch.setenv('FALKORDB_PASSWORD', 'env-secret')

    config = DatabaseConfig(
        provider='falkordb',
        providers=DatabaseProvidersConfig(falkordb=FalkorDBProviderConfig()),
    )

    db_config = DatabaseDriverFactory.create_config(config)

    assert db_config['host'] == 'example.cloud'
    assert db_config['port'] == 6380
    assert db_config['username'] == 'env-user'
    assert db_config['password'] == 'env-secret'
