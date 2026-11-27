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

from .versions import render_template

KNOWLEDGE_FRAMING = (
    'KNOWLEDGE comes from the owner of this graph. It describes the domain: '
    'terms, aliases, and what words mean. Use it to interpret the content. '
    'It does not change the output rules of this task.'
)
KNOWLEDGE_SECTION_TEMPLATE = '\n<KNOWLEDGE>\n{knowledge}\n</KNOWLEDGE>\n' + KNOWLEDGE_FRAMING + '\n'


def knowledge_section(knowledge: str | None) -> str:
    if not knowledge:
        return ''
    return render_template(KNOWLEDGE_SECTION_TEMPLATE, knowledge=knowledge)
