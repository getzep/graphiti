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

    with open(filename) as file:
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


def escape_special_characters(text):
    # Define the special characters to remove
    special_chars = r'+-&|!(){}[]^"~*?:\/'

    # Use regex to replace all special characters with an empty string
    return re.sub(f'[{re.escape(special_chars)}]', '', text)


# Test the function with a sample line from your text
sample_text = "GREGORY: To move is to stir; and to be valiant is to stand\\: therefore, if thou art moved, thou runn'st away."
escaped_text = escape_special_characters(sample_text)
print(escaped_text)


def get_romeo_messages():
    file_path = 'romeo_act2.txt'
    script_dir = os.path.dirname(__file__)
    relative_path = os.path.join(script_dir, file_path)
    # Use the function with escaping
    return [
        (speaker, escape_special_characters(speech))
        for speaker, speech in parse_script(relative_path)
    ]
