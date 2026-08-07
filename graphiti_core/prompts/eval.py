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
from typing import Any

from pydantic import BaseModel, Field

from .models import ChatPrompt, SystemMessage, UserMessage
from .prompt_helpers import to_prompt_json


class QueryExpansion(BaseModel):
    query: str = Field(..., description='query optimized for database search')


class QAResponse(BaseModel):
    ANSWER: str = Field(..., description='how Alice would answer the question')


class EvalResponse(BaseModel):
    is_correct: bool = Field(..., description='boolean if the answer is correct or incorrect')
    reasoning: str = Field(
        ..., description='why you determined the response was correct or incorrect'
    )


class EvalAddEpisodeResults(BaseModel):
    candidate_is_worse: bool = Field(
        ...,
        description='boolean if the baseline extraction is higher quality than the candidate extraction.',
    )
    reasoning: str = Field(
        ..., description='why you determined the response was correct or incorrect'
    )


class EvalPrompts(ABC):
    @abstractmethod
    def query_expansion(self, context: dict[str, Any]) -> ChatPrompt:
        """
        Purpose
                Rephrase a question into a retrieval query.

                Called from
                Unused in core runtime (eval tooling).

                Context inputs
                ``query``.

                Output contract
                ``ChatPrompt`` for ``eval.query_expansion``; ``QueryExpansion``.

                Gotchas
                Eval-only; not used in ingestion.

                Modification guidance
                Safe to customize for eval experiments.
        """

    @abstractmethod
    def qa_prompt(self, context: dict[str, Any]) -> ChatPrompt:
        """
        Purpose
                Answer a question as Alice using summaries and facts.

                Called from
                Unused in core runtime (eval tooling).

                Context inputs
                ``entity_summaries``, ``facts``, ``query``.

                Output contract
                ``ChatPrompt`` for ``eval.qa_prompt``; ``QAResponse``.

                Gotchas
                Eval-only.

                Modification guidance
                Safe to customize for eval experiments.
        """

    @abstractmethod
    def eval_prompt(self, context: dict[str, Any]) -> ChatPrompt:
        """
        Purpose
                Grade a response against a gold-standard answer.

                Called from
                Unused in core runtime (eval tooling).

                Context inputs
                ``query``, ``answer``, ``response``.

                Output contract
                ``ChatPrompt`` for ``eval.eval_prompt``; ``EvalResponse``.

                Gotchas
                Eval-only.

                Modification guidance
                Safe to customize for eval experiments.
        """

    @abstractmethod
    def eval_add_episode_results(self, context: dict[str, Any]) -> ChatPrompt:
        """
        Purpose
                Compare baseline vs candidate episode extraction quality.

                Called from
                Unused in core runtime (eval tooling).

                Context inputs
                ``previous_messages``, ``message``, ``baseline``, ``candidate``.

                Output contract
                ``ChatPrompt`` for ``eval.eval_add_episode_results``; ``EvalAddEpisodeResults``.

                Gotchas
                Eval-only.

                Modification guidance
                Safe to customize for eval experiments.
        """


class DefaultEvalPrompts(EvalPrompts):
    def query_expansion(self, context: dict[str, Any]) -> ChatPrompt:
        sys_prompt = """You are an expert at rephrasing questions into queries used in a database retrieval system"""

        user_prompt = f"""
        Bob is asking Alice a question, are you able to rephrase the question into a simpler one about Alice in the third person
        that maintains the relevant context?
        <QUESTION>
        {to_prompt_json(context['query'])}
        </QUESTION>
        """
        return ChatPrompt(
            system=SystemMessage(content=sys_prompt),
            user=UserMessage(content=user_prompt),
        )

    def qa_prompt(self, context: dict[str, Any]) -> ChatPrompt:
        sys_prompt = """You are Alice and should respond to all questions from the first person perspective of Alice"""

        user_prompt = f"""
        Your task is to briefly answer the question in the way that you think Alice would answer the question.
        You are given the following entity summaries and facts to help you determine the answer to your question.
        <ENTITY_SUMMARIES>
        {to_prompt_json(context['entity_summaries'])}
        </ENTITY_SUMMARIES>
        <FACTS>
        {to_prompt_json(context['facts'])}
        </FACTS>
        <QUESTION>
        {context['query']}
        </QUESTION>
        """
        return ChatPrompt(
            system=SystemMessage(content=sys_prompt),
            user=UserMessage(content=user_prompt),
        )

    def eval_prompt(self, context: dict[str, Any]) -> ChatPrompt:
        sys_prompt = """You are a judge that determines if answers to questions match a gold standard answer"""

        user_prompt = f"""
        Given the QUESTION and the gold standard ANSWER determine if the RESPONSE to the question is correct or incorrect.
        Although the RESPONSE may be more verbose, mark it as correct as long as it references the same topic 
        as the gold standard ANSWER. Also include your reasoning for the grade.
        <QUESTION>
        {context['query']}
        </QUESTION>
        <ANSWER>
        {context['answer']}
        </ANSWER>
        <RESPONSE>
        {context['response']}
        </RESPONSE>
        """
        return ChatPrompt(
            system=SystemMessage(content=sys_prompt),
            user=UserMessage(content=user_prompt),
        )

    def eval_add_episode_results(self, context: dict[str, Any]) -> ChatPrompt:
        sys_prompt = """You are a judge that determines whether a baseline graph building result from a list of messages is better
            than a candidate graph building result based on the same messages."""

        user_prompt = f"""
        Given the following PREVIOUS MESSAGES and MESSAGE, determine if the BASELINE graph data extracted from the 
        conversation is higher quality than the CANDIDATE graph data extracted from the conversation.
        
        Return False if the BASELINE extraction is better, and True otherwise. If the CANDIDATE extraction and
        BASELINE extraction are nearly identical in quality, return True. Add your reasoning for your decision to the reasoning field
        
        <PREVIOUS MESSAGES>
        {context['previous_messages']}
        </PREVIOUS MESSAGES>
        <MESSAGE>
        {context['message']}
        </MESSAGE>
        
        <BASELINE>
        {context['baseline']}
        </BASELINE>
        
        <CANDIDATE>
        {context['candidate']}
        </CANDIDATE>
        """
        return ChatPrompt(
            system=SystemMessage(content=sys_prompt),
            user=UserMessage(content=user_prompt),
        )
