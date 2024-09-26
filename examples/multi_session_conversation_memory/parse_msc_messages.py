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
from datetime import datetime

from pydantic import BaseModel


class ParsedMscMessage(BaseModel):
    speaker_name: str
    actual_timestamp: datetime
    content: str
    group_id: str


def parse_msc_messages() -> list[list[ParsedMscMessage]]:
    msc_messages: list[list[ParsedMscMessage]] = []
    speakers = ['Alice', 'Bob']

    with open('../data/msc.json') as file:
        data = json.load(file)['data']
        for i, conversation in enumerate(data):
            messages: list[ParsedMscMessage] = []
            for previous_dialog in conversation['previous_dialogs']:
                dialog = previous_dialog['dialog']
                speaker_idx = 0

                for utterance in dialog:
                    content = utterance['text']
                    messages.append(
                        ParsedMscMessage(
                            speaker_name=speakers[speaker_idx],
                            content=content,
                            actual_timestamp=datetime.now(),
                            group_id=str(i),
                        )
                    )
                    speaker_idx += 1
                    speaker_idx %= 2

            dialog = conversation['dialog']
            speaker_idx = 0
            for utterance in dialog:
                content = utterance['text']
                messages.append(
                    ParsedMscMessage(
                        speaker_name=speakers[speaker_idx],
                        content=content,
                        actual_timestamp=datetime.now(),
                        group_id=str(i),
                    )
                )
                speaker_idx += 1
                speaker_idx %= 2

            msc_messages.append(messages)

    return msc_messages


def conversation_q_and_a() -> list[tuple[str, str]]:
    with open('../data/msc.json') as file:
        data = json.load(file)['data']

        qa: list[tuple[str, str]] = []
        for conversation in data:
            query = conversation['self_instruct']['B']
            answer = conversation['self_instruct']['A']

            qa.append((query, answer))
        return qa
