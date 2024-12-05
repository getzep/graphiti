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

import json
from typing import Any, Protocol, TypedDict

from pydantic import BaseModel, Field

from .models import Message, PromptFunction, PromptVersion


class QueryExpansion(BaseModel):
    query: str = Field(..., description='query optimized for database search')


class QAResponse(BaseModel):
    ANSWER: str = Field(..., description='how Alice would answer the question')


class EvalResponse(BaseModel):
    is_correct: bool = Field(..., description='boolean if the answer is correct or incorrect')
    reasoning: str = Field(
        ..., description='why you determined the response was correct or incorrect'
    )


class Prompt(Protocol):
    qa_prompt: PromptVersion
    eval_prompt: PromptVersion
    query_expansion: PromptVersion


class Versions(TypedDict):
    qa_prompt: PromptFunction
    eval_prompt: PromptFunction
    query_expansion: PromptFunction


def query_expansion(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an expert at rephrasing questions into queries used in a database retrieval system"""

    user_prompt = f"""
    Bob is asking Alice a question, are you able to rephrase the question into a simpler one about Alice in the third person
    that maintains the relevant context?
    <QUESTION>
    {json.dumps(context['query'])}
    </QUESTION>
    """
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def qa_prompt(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are Alice and should respond to all questions from the first person perspective of Alice"""

    user_prompt = f"""
    Your task is to briefly answer the question in the way that you think Alice would answer the question.
    You are given the following entity summaries and facts to help you determine the answer to your question.
    <ENTITY_SUMMARIES>
    {json.dumps(context['entity_summaries'])}
    </ENTITY_SUMMARIES>
    <FACTS>
    {json.dumps(context['facts'])}
    </FACTS>
    <QUESTION>
    {context['query']}
    </QUESTION>
    """
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def eval_prompt(context: dict[str, Any]) -> list[Message]:
    sys_prompt = (
        """You are a judge that determines if answers to questions match a gold standard answer"""
    )

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
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


versions: Versions = {
    'qa_prompt': qa_prompt,
    'eval_prompt': eval_prompt,
    'query_expansion': query_expansion,
}
