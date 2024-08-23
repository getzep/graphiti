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

import os
import re


def parse_script(filename):
    current_speaker = None
    current_speech = []
    messages = []

    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()

            # Check if this line is a new speaker
            if (
                line
                and line.isupper()
                and not line.startswith('ACT')
                and not line.startswith('SCENE')
            ):
                # If we have a current speaker, save their message
                if current_speaker:
                    messages.append((current_speaker, ' '.join(current_speech)))

                # Start a new speech
                current_speaker = line
                current_speech = []
            elif line and not line.startswith('[') and current_speaker:
                # Add this line to the current speech
                current_speech.append(line)

    # Add the last speech
    if current_speaker:
        messages.append((current_speaker, ' '.join(current_speech)))

    return messages


def get_romeo_messages():
    file_path = 'romeo.txt'
    script_dir = os.path.dirname(__file__)
    relative_path = os.path.join(script_dir, file_path)
    # Use the function
    return parse_script(relative_path)
