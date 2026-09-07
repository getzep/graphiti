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
