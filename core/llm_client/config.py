class LLMConfig:
    """
    Configuration class for the Language Learning Model (LLM).

    This class encapsulates the necessary parameters to interact with an LLM API,
    such as OpenAI's GPT models. It stores the API key, model name, and base URL
    for making requests to the LLM service.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        base_url: str = "https://api.openai.com",
    ):
        """
        Initialize the LLMConfig with the provided parameters.

        Args:
            api_key (str): The authentication key for accessing the LLM API.
                           This is required for making authorized requests.

            model (str, optional): The specific LLM model to use for generating responses.
                                   Defaults to "gpt-4o", which appears to be a custom model name.
                                   Common values might include "gpt-3.5-turbo" or "gpt-4".

            base_url (str, optional): The base URL of the LLM API service.
                                      Defaults to "https://api.openai.com", which is OpenAI's standard API endpoint.
                                      This can be changed if using a different provider or a custom endpoint.
        """
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
