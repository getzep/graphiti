"""Utility functions for Graphiti MCP Server."""

from collections.abc import Callable


def create_azure_credential_token_provider() -> Callable[[], str]:
    """
    Create Azure credential token provider for managed identity authentication.

    Requires azure-identity package. Install with: pip install mcp-server[azure]

    Raises:
        ImportError: If azure-identity package is not installed
    """
    try:
        from azure.identity import DefaultAzureCredential, get_bearer_token_provider
    except ImportError:
        raise ImportError(
            'azure-identity is required for Azure AD authentication. '
            'Install it with: pip install mcp-server[azure]'
        ) from None

    credential = DefaultAzureCredential()
    token_provider = get_bearer_token_provider(
        credential, 'https://cognitiveservices.azure.com/.default'
    )
    return token_provider
