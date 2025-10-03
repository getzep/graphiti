"""Utilities for summarizing document diffs using LLM."""

from graphiti_core.llm_client import LLMClient
from graphiti_core.prompts import prompt_library
from graphiti_core.prompts.summarize_diff import DiffSummary


async def summarize_diff(
    llm_client: LLMClient,
    diff_content: str,
    uri: str,
) -> str:
    """Generate semantic summary of document changes from unified diff.

    Args:
        llm_client: LLM client for generating summary
        diff_content: Unified diff content
        uri: Document URI for context

    Returns:
        Semantic summary of changes
    """
    context = {
        'diff_content': diff_content,
        'document_uri': uri,
    }

    # Use big model (no model_size parameter = ModelSize.medium = MODEL_NAME from .env)
    llm_response = await llm_client.generate_response(
        prompt_library.summarize_diff.summarize_diff(context),
        response_model=DiffSummary,
    )

    return llm_response.get('summary', '')
