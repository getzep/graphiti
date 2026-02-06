"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from google import genai
else:
    try:
        from google import genai
    except ImportError:
        raise ImportError(
            'google-genai is required for Gemini clients. '
            'Install it with: pip install graphiti-core[google-genai]'
        ) from None

logger = logging.getLogger(__name__)


def create_gemini_client(
    api_key: str | None,
    client_type: str = 'client',
) -> 'genai.Client':
    """
    Create a Google Gemini client with support for both API key and ADC authentication.

    This helper consolidates the client initialization logic used across GeminiClient,
    GeminiEmbedder, and GeminiRerankerClient.

    Authentication precedence:
    1. If api_key is provided, use Google AI API with the API key
    2. If api_key is None, use Vertex AI with Application Default Credentials (ADC)

    Args:
        api_key: The Google API key. If None, will use ADC with Vertex AI.
        client_type: A descriptive name for logging (e.g., "client", "embedder", "reranker").

    Returns:
        genai.Client: An initialized Google Gemini client.

    Raises:
        ValueError: If ADC is used but GOOGLE_CLOUD_PROJECT is not set.
        Exception: If authentication fails with helpful error messages.

    Environment Variables (for ADC):
        GOOGLE_CLOUD_PROJECT: Required when using ADC. Your GCP project ID.
        GOOGLE_CLOUD_LOCATION: Optional. Defaults to 'us-central1'.
        GOOGLE_APPLICATION_CREDENTIALS: Optional. Path to service account JSON.
    """
    # If no API key is provided, use Vertex AI with Application Default Credentials
    if api_key is None:
        # Get project and location from environment variables
        project = os.getenv('GOOGLE_CLOUD_PROJECT')
        location = os.getenv('GOOGLE_CLOUD_LOCATION', 'us-central1')

        if not project:
            raise ValueError(
                'GOOGLE_CLOUD_PROJECT environment variable is required when using '
                'Application Default Credentials (ADC) with Gemini.\n\n'
                'To fix this:\n'
                '1. Set GOOGLE_CLOUD_PROJECT=your-project-id, or\n'
                '2. Set GOOGLE_API_KEY if you want to use the Google AI API instead of Vertex AI'
            )

        logger.info(
            f'Creating Gemini {client_type} with Vertex AI using project={project}, location={location}'
        )
        try:
            return genai.Client(vertexai=True, project=project, location=location)
        except Exception as e:
            error_msg = str(e).lower()
            if any(
                keyword in error_msg
                for keyword in [
                    'credentials',
                    'authentication',
                    'unauthorized',
                    'permission denied',
                ]
            ):
                raise Exception(
                    f'Google authentication failed: {e}\n\n'
                    'To fix this, either:\n'
                    '1. Set GOOGLE_API_KEY environment variable with your API key, or\n'
                    '2. Set up Application Default Credentials for Vertex AI:\n'
                    '   - Run: gcloud auth application-default login\n'
                    '   - Or set GOOGLE_APPLICATION_CREDENTIALS to your service account JSON path\n'
                    '   - And set GOOGLE_CLOUD_PROJECT to your GCP project ID'
                ) from e
            raise
    else:
        # Use Google AI API with the provided API key
        logger.info(f'Creating Gemini {client_type} with API key')
        try:
            return genai.Client(api_key=api_key)
        except Exception as e:
            error_msg = str(e).lower()
            if any(
                keyword in error_msg
                for keyword in [
                    'credentials',
                    'authentication',
                    'unauthorized',
                    'permission denied',
                ]
            ):
                raise Exception(
                    f'Google authentication failed: {e}\n\n'
                    'To fix this, either:\n'
                    '1. Set GOOGLE_API_KEY environment variable with your API key, or\n'
                    '2. Set up Application Default Credentials for Vertex AI:\n'
                    '   - Run: gcloud auth application-default login\n'
                    '   - Or set GOOGLE_APPLICATION_CREDENTIALS to your service account JSON path\n'
                    '   - And set GOOGLE_CLOUD_PROJECT to your GCP project ID'
                ) from e
            raise
