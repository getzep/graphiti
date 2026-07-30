"""Maps OpenAI's failures onto HTTP statuses, so they don't reach the caller as a bare 500.

Every retrieval endpoint embeds its query before it can search, so an unusable OPENAI_API_KEY
surfaces on `/search` rather than at ingestion time. Unmapped, it reached the client as a bare 500
`Internal Server Error`, which reads as a bug in the graph query — the actual cause was only in
the service logs, and only as a traceback.

main.py registers the handler against openai.APIError, the base class, so a subclass the SDK adds
later is still mapped. openai.OpenAIError siblings that aren't APIError (LengthFinishReasonError,
ContentFilterFinishReasonError) stay 500s deliberately: they describe a response that arrived, not
an upstream that refused us.
"""

import openai


def describe_failure(exc: openai.APIError) -> tuple[int, str]:
    """Map an OpenAI failure to a status and a message that names what to go and fix."""
    if isinstance(exc, openai.RateLimitError):
        # 429 either way, because OpenAI returns 429 for a real rate limit and for
        # insufficient_quota alike, and only the caller can tell whether retrying is worth it.
        # The distinction that matters to the operator is in the message, so pass it through.
        return 429, (
            'OpenAI rejected the request: either a rate limit, or the quota on the account '
            f'behind OPENAI_API_KEY is exhausted. Upstream said: {exc}'
        )

    if isinstance(exc, openai.AuthenticationError | openai.PermissionDeniedError):
        # 502, not 401/403: the caller's GRAPHITI_API_KEY was accepted, and echoing OpenAI's
        # status would tell them to go fix a credential they do not hold.
        return 502, (
            "OpenAI rejected this service's credentials. Check that OPENAI_API_KEY is set to a "
            f'valid key with access to the configured model. Upstream said: {exc}'
        )

    if isinstance(exc, openai.APIStatusError):
        return 502, f'OpenAI returned an error. Upstream said: {exc}'

    # Everything else is a connection failure or a timeout: no response ever arrived.
    return 504, f'Could not reach OpenAI. Upstream said: {exc}'
