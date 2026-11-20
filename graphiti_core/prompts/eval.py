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
from typing import Any, Protocol, TypedDict

from pydantic import BaseModel, Field

from .models import ChatPrompt, Message, PromptFunction, PromptVersion
from .prompt_helpers import to_prompt_json
from .versions import PromptTemplate, render_template

QUERY_EXPANSION_SYSTEM_TEMPLATE = (
    'You are an expert at rephrasing questions into queries used in a database retrieval system'
)
QUERY_EXPANSION_USER_TEMPLATE = """
    Bob is asking Alice a question, are you able to rephrase the question into a simpler one about Alice in the third person
    that maintains the relevant context?
    <QUESTION>
    {query}
    </QUESTION>
    """
QA_SYSTEM_TEMPLATE = (
    'You are Alice and should respond to all questions from the first person perspective of Alice'
)
QA_USER_TEMPLATE = """
    Your task is to briefly answer the question in the way that you think Alice would answer the question.
    You are given the following entity summaries and facts to help you determine the answer to your question.
    <ENTITY_SUMMARIES>
    {entity_summaries}
    </ENTITY_SUMMARIES>
    <FACTS>
    {facts}
    </FACTS>
    <QUESTION>
    {query}
    </QUESTION>
    """
EVAL_SYSTEM_TEMPLATE = (
    'You are a judge that determines if answers to questions match a gold standard answer'
)
EVAL_USER_TEMPLATE = """
    Given the QUESTION and the gold standard ANSWER determine if the RESPONSE to the question is correct or incorrect.
    Although the RESPONSE may be more verbose, mark it as correct as long as it references the same topic\x20
    as the gold standard ANSWER. Also include your reasoning for the grade.
    <QUESTION>
    {query}
    </QUESTION>
    <ANSWER>
    {answer}
    </ANSWER>
    <RESPONSE>
    {response}
    </RESPONSE>
    """
EVAL_ADD_EPISODE_RESULTS_SYSTEM_TEMPLATE = """You are a judge that determines whether a baseline graph building result from a list of messages is better
        than a candidate graph building result based on the same messages."""
EVAL_ADD_EPISODE_RESULTS_USER_TEMPLATE = """
    Given the following PREVIOUS MESSAGES and MESSAGE, determine if the BASELINE graph data extracted from the\x20
    conversation is higher quality than the CANDIDATE graph data extracted from the conversation.
\x20\x20\x20\x20
    Return False if the BASELINE extraction is better, and True otherwise. If the CANDIDATE extraction and
    BASELINE extraction are nearly identical in quality, return True. Add your reasoning for your decision to the reasoning field
\x20\x20\x20\x20
    <PREVIOUS MESSAGES>
    {previous_messages}
    </PREVIOUS MESSAGES>
    <MESSAGE>
    {message}
    </MESSAGE>
\x20\x20\x20\x20
    <BASELINE>
    {baseline}
    </BASELINE>
\x20\x20\x20\x20
    <CANDIDATE>
    {candidate}
    </CANDIDATE>
    """


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


class Prompt(Protocol):
    qa_prompt: PromptVersion
    eval_prompt: PromptVersion
    query_expansion: PromptVersion
    eval_add_episode_results: PromptVersion


class Versions(TypedDict):
    qa_prompt: PromptFunction
    eval_prompt: PromptFunction
    query_expansion: PromptFunction
    eval_add_episode_results: PromptFunction


def query_expansion(context: dict[str, Any]) -> list[Message]:
    user_prompt = render_template(
        QUERY_EXPANSION_USER_TEMPLATE, query=to_prompt_json(context['query'])
    )
    return [
        Message(role='system', content=QUERY_EXPANSION_SYSTEM_TEMPLATE),
        Message(role='user', content=user_prompt),
    ]


def qa_prompt(context: dict[str, Any]) -> list[Message]:
    user_prompt = render_template(
        QA_USER_TEMPLATE,
        entity_summaries=to_prompt_json(context['entity_summaries']),
        facts=to_prompt_json(context['facts']),
        query=context['query'],
    )
    return [
        Message(role='system', content=QA_SYSTEM_TEMPLATE),
        Message(role='user', content=user_prompt),
    ]


def eval_prompt(context: dict[str, Any]) -> list[Message]:
    user_prompt = render_template(
        EVAL_USER_TEMPLATE,
        query=context['query'],
        answer=context['answer'],
        response=context['response'],
    )
    return [
        Message(role='system', content=EVAL_SYSTEM_TEMPLATE),
        Message(role='user', content=user_prompt),
    ]


def eval_add_episode_results(context: dict[str, Any]) -> list[Message]:
    user_prompt = render_template(
        EVAL_ADD_EPISODE_RESULTS_USER_TEMPLATE,
        previous_messages=context['previous_messages'],
        message=context['message'],
        baseline=context['baseline'],
        candidate=context['candidate'],
    )
    return [
        Message(role='system', content=EVAL_ADD_EPISODE_RESULTS_SYSTEM_TEMPLATE),
        Message(role='user', content=user_prompt),
    ]


versions: Versions = {
    'qa_prompt': qa_prompt,
    'eval_prompt': eval_prompt,
    'query_expansion': query_expansion,
    'eval_add_episode_results': eval_add_episode_results,
}

query_expansion.__prompt_template__ = PromptTemplate(
    (QUERY_EXPANSION_SYSTEM_TEMPLATE, QUERY_EXPANSION_USER_TEMPLATE)
)
qa_prompt.__prompt_template__ = PromptTemplate((QA_SYSTEM_TEMPLATE, QA_USER_TEMPLATE))
eval_prompt.__prompt_template__ = PromptTemplate((EVAL_SYSTEM_TEMPLATE, EVAL_USER_TEMPLATE))
eval_add_episode_results.__prompt_template__ = PromptTemplate(
    (EVAL_ADD_EPISODE_RESULTS_SYSTEM_TEMPLATE, EVAL_ADD_EPISODE_RESULTS_USER_TEMPLATE)
)


class EvalPrompts(ABC):
    @abstractmethod
    def query_expansion(self, context: dict[str, Any]) -> ChatPrompt: ...

    @abstractmethod
    def qa_prompt(self, context: dict[str, Any]) -> ChatPrompt: ...

    @abstractmethod
    def eval_prompt(self, context: dict[str, Any]) -> ChatPrompt: ...

    @abstractmethod
    def eval_add_episode_results(self, context: dict[str, Any]) -> ChatPrompt: ...


class DefaultEvalPrompts(EvalPrompts):
    def query_expansion(self, context: dict[str, Any]) -> ChatPrompt:
        return ChatPrompt.from_messages(query_expansion(context))

    def qa_prompt(self, context: dict[str, Any]) -> ChatPrompt:
        return ChatPrompt.from_messages(qa_prompt(context))

    def eval_prompt(self, context: dict[str, Any]) -> ChatPrompt:
        return ChatPrompt.from_messages(eval_prompt(context))

    def eval_add_episode_results(self, context: dict[str, Any]) -> ChatPrompt:
        return ChatPrompt.from_messages(eval_add_episode_results(context))
