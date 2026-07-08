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

from abc import ABC, abstractmethod


class CrossEncoderClient(ABC):
    """
    CrossEncoderClient is an abstract base class that defines the interface
    for cross-encoder models used for ranking passages based on their relevance to a query.
    It allows for different implementations of cross-encoder models to be used interchangeably.
    """

    @abstractmethod
    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        """
        Rank the given passages based on their relevance to the query.

        Args:
            query (str): The query string.
            passages (list[str]): A list of passages to rank.

        Returns:
            list[tuple[str, float]]: A list of tuples containing the passage and its score,
                                     sorted in descending order of relevance.
        """
        pass
