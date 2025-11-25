#!/usr/bin/env python3
"""
Tests for Neptune database provider configuration and validation.

These tests validate Neptune-specific configuration requirements including:
- Endpoint format validation (neptune-db:// and neptune-graph://)
- AOSS host requirement validation
- AWS credential validation (mocked)
- Environment variable overrides
- Factory creation with Neptune provider
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

from config.schema import DatabaseProvidersConfig, GraphitiConfig, NeptuneProviderConfig
from services.factories import DatabaseDriverFactory


def test_neptune_provider_config_validation():
    """Test NeptuneProviderConfig validation rules."""
    print('\nTesting Neptune provider configuration validation...')

    # Test valid Database endpoint
    try:
        config = NeptuneProviderConfig(
            host='neptune-db://my-cluster.us-east-1.neptune.amazonaws.com',
            aoss_host='my-aoss.us-east-1.aoss.amazonaws.com',
            port=8182,
            aoss_port=443,
        )
        print('✓ Valid Database endpoint accepted')
        assert config.host.startswith('neptune-db://')
    except Exception as e:
        print(f'✗ Failed to accept valid Database endpoint: {e}')
        raise

    # Test valid Analytics endpoint
    try:
        config = NeptuneProviderConfig(
            host='neptune-graph://g-abc123xyz',
            aoss_host='my-aoss.us-east-1.aoss.amazonaws.com',
        )
        print('✓ Valid Analytics endpoint accepted')
        assert config.host.startswith('neptune-graph://')
    except Exception as e:
        print(f'✗ Failed to accept valid Analytics endpoint: {e}')
        raise

    # Test invalid endpoint format
    try:
        config = NeptuneProviderConfig(
            host='https://my-neptune.com',
            aoss_host='my-aoss.us-east-1.aoss.amazonaws.com',
        )
        print('✗ Invalid endpoint format should have been rejected')
        raise AssertionError('Expected ValueError for invalid endpoint format')
    except ValueError as e:
        print(f'✓ Invalid endpoint format rejected: {str(e)[:60]}...')
        assert 'must start with neptune-db:// or neptune-graph://' in str(e)

    # Test missing AOSS host
    try:
        config = NeptuneProviderConfig(
            host='neptune-db://my-cluster.us-east-1.neptune.amazonaws.com',
            aoss_host=None,
        )
        print('✗ Missing AOSS host should have been rejected')
        raise AssertionError('Expected ValueError for missing AOSS host')
    except ValueError as e:
        print(f'✓ Missing AOSS host rejected: {str(e)[:60]}...')
        assert 'requires aoss_host' in str(e)

    # Test port range validation
    try:
        config = NeptuneProviderConfig(
            host='neptune-db://my-cluster.us-east-1.neptune.amazonaws.com',
            aoss_host='my-aoss.us-east-1.aoss.amazonaws.com',
            port=70000,  # Invalid port
        )
        print('✗ Invalid port should have been rejected')
        raise AssertionError('Expected ValidationError for invalid port')
    except Exception as e:
        print(f'✓ Invalid port rejected: {str(e)[:60]}...')

    print('✓ Neptune provider configuration validation complete')


def test_neptune_environment_overrides():
    """Test Neptune configuration with environment variable overrides."""
    print('\nTesting Neptune environment variable overrides...')

    # Set Neptune environment variables
    test_env = {
        'NEPTUNE_HOST': 'neptune-db://test-cluster.us-west-2.neptune.amazonaws.com',
        'AOSS_HOST': 'test-aoss.us-west-2.aoss.amazonaws.com',
        'NEPTUNE_PORT': '9999',
        'AOSS_PORT': '9443',
        'AWS_REGION': 'us-west-2',
    }

    with patch.dict(os.environ, test_env, clear=False):
        try:
            # Load config with environment overrides
            config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
            config = GraphitiConfig(_env_file=None, config_path=str(config_path))

            print('✓ Loaded configuration with environment overrides')

            # Verify environment variables were applied
            if config.database.providers and config.database.providers.neptune:
                neptune_config = config.database.providers.neptune
                print(f'  Neptune host: {neptune_config.host}')
                print(f'  AOSS host: {neptune_config.aoss_host}')
                print(f'  Neptune port: {neptune_config.port}')
                print(f'  AOSS port: {neptune_config.aoss_port}')
                print(f'  Region: {neptune_config.region}')

                # Verify the overrides were applied correctly
                assert neptune_config.host == test_env['NEPTUNE_HOST']
                assert neptune_config.aoss_host == test_env['AOSS_HOST']
                assert neptune_config.port == 9999
                assert neptune_config.aoss_port == 9443
                assert neptune_config.region == test_env['AWS_REGION']
                print('✓ All environment overrides applied correctly')
            else:
                print('⚠ Neptune provider not configured, skipping validation')

        except Exception as e:
            print(f'✗ Failed to load configuration with environment overrides: {e}')
            raise

    print('✓ Neptune environment override tests complete')


def test_neptune_factory_creation_with_mock_credentials():
    """Test Neptune factory creation with mocked AWS credentials."""
    print('\nTesting Neptune factory creation with mocked credentials...')

    # Mock AWS credentials
    mock_credentials = MagicMock()
    mock_credentials.access_key = 'mock_access_key'
    mock_credentials.secret_key = 'mock_secret_key'
    mock_credentials.token = None

    mock_session = MagicMock()
    mock_session.get_credentials.return_value = mock_credentials
    mock_session.region_name = 'us-east-1'

    # Create test configuration
    test_config = {
        'database': {
            'provider': 'neptune',
            'providers': {
                'neptune': {
                    'host': 'neptune-db://test-cluster.us-east-1.neptune.amazonaws.com',
                    'aoss_host': 'test-aoss.us-east-1.aoss.amazonaws.com',
                    'port': 8182,
                    'aoss_port': 443,
                    'region': 'us-east-1',
                }
            },
        }
    }

    with patch('boto3.Session', return_value=mock_session):
        try:
            # Create database config using factory
            config = DatabaseProvidersConfig(**test_config['database']['providers'])
            print('✓ Created DatabaseProvidersConfig with Neptune')

            # Verify Neptune config was created
            assert config.neptune is not None
            print('✓ Neptune provider config exists')

            # Create driver config from factory
            db_config = DatabaseDriverFactory.create_config(test_config)
            print('✓ Created driver config from factory')

            # Verify config contains expected Neptune parameters
            assert db_config['driver'] == 'neptune'
            assert db_config['host'] == 'neptune-db://test-cluster.us-east-1.neptune.amazonaws.com'
            assert db_config['aoss_host'] == 'test-aoss.us-east-1.aoss.amazonaws.com'
            assert db_config['port'] == 8182
            assert db_config['aoss_port'] == 443
            assert db_config['region'] == 'us-east-1'
            print('✓ All Neptune parameters validated correctly')

        except Exception as e:
            print(f'✗ Factory creation failed: {e}')
            raise

    print('✓ Neptune factory creation tests complete')


def test_neptune_factory_missing_credentials():
    """Test Neptune factory creation fails gracefully without AWS credentials."""
    print('\nTesting Neptune factory behavior with missing AWS credentials...')

    # Mock AWS session with no credentials
    mock_session = MagicMock()
    mock_session.get_credentials.return_value = None
    mock_session.region_name = 'us-east-1'

    test_config = {
        'database': {
            'provider': 'neptune',
            'providers': {
                'neptune': {
                    'host': 'neptune-db://test-cluster.us-east-1.neptune.amazonaws.com',
                    'aoss_host': 'test-aoss.us-east-1.aoss.amazonaws.com',
                    'port': 8182,
                    'aoss_port': 443,
                }
            },
        }
    }

    with patch('boto3.Session', return_value=mock_session):
        try:
            DatabaseDriverFactory.create_config(test_config)
            print('✗ Factory should have failed with missing credentials')
            raise AssertionError('Expected ValueError for missing AWS credentials')
        except ValueError as e:
            print(f'✓ Missing credentials rejected: {str(e)[:60]}...')
            assert 'AWS credentials not configured' in str(e)
            assert 'aws configure' in str(e).lower()

    print('✓ Missing credentials handling validated')


def test_neptune_factory_import_check():
    """Test Neptune factory import checking and error messages."""
    print('\nTesting Neptune driver import availability...')

    try:
        from graphiti_core.driver.neptune_driver import NeptuneDriver

        print('✓ NeptuneDriver successfully imported')
        print(f'  NeptuneDriver class: {NeptuneDriver.__name__}')

    except ImportError as e:
        print(f'⚠ NeptuneDriver not available: {e}')
        print('  This is expected if graphiti-core[neptune] is not installed')
        print('  Install with: uv add graphiti-core[neptune]')

    print('✓ Import check complete')


def test_database_providers_config_with_neptune():
    """Test DatabaseProvidersConfig accepts Neptune configuration."""
    print('\nTesting DatabaseProvidersConfig with Neptune...')

    try:
        config = DatabaseProvidersConfig(
            neptune=NeptuneProviderConfig(
                host='neptune-db://my-cluster.us-east-1.neptune.amazonaws.com',
                aoss_host='my-aoss.us-east-1.aoss.amazonaws.com',
                port=8182,
                aoss_port=443,
                region='us-east-1',
            )
        )
        print('✓ DatabaseProvidersConfig created with Neptune')

        assert config.neptune is not None
        assert config.neptune.host == 'neptune-db://my-cluster.us-east-1.neptune.amazonaws.com'
        assert config.neptune.aoss_host == 'my-aoss.us-east-1.aoss.amazonaws.com'
        assert config.neptune.port == 8182
        assert config.neptune.aoss_port == 443
        assert config.neptune.region == 'us-east-1'
        print('✓ All Neptune fields validated correctly')

    except Exception as e:
        print(f'✗ Failed to create DatabaseProvidersConfig with Neptune: {e}')
        raise

    print('✓ DatabaseProvidersConfig Neptune integration complete')


if __name__ == '__main__':
    print('=' * 70)
    print('Neptune Configuration Tests')
    print('=' * 70)

    try:
        test_neptune_provider_config_validation()
        test_neptune_environment_overrides()
        test_database_providers_config_with_neptune()
        test_neptune_factory_import_check()
        test_neptune_factory_creation_with_mock_credentials()
        test_neptune_factory_missing_credentials()

        print('\n' + '=' * 70)
        print('All Neptune tests passed!')
        print('=' * 70)

    except Exception as e:
        print('\n' + '=' * 70)
        print(f'Tests failed: {e}')
        print('=' * 70)
        raise
