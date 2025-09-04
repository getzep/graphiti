"""Utility functions for Graphiti MCP Server."""

from collections.abc import Callable

from azure.identity import DefaultAzureCredential, get_bearer_token_provider


def create_azure_credential_token_provider() -> Callable[[], str]:
    """Create Azure credential token provider for managed identity authentication."""
    credential = DefaultAzureCredential()
    token_provider = get_bearer_token_provider(
        credential, 'https://cognitiveservices.azure.com/.default'
    )
    return token_provider
