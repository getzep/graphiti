import os
import re


def sanitize_text(text):
    # Remove special characters and extra whitespace
    sanitized = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    sanitized = ' '.join(sanitized.split())
    return sanitized


def parse_script(filename):
    current_speaker = None
    current_speech = []
    messages = []

    with open(filename, encoding='utf-8') as file:
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
                    sanitized_speech = sanitize_text(' '.join(current_speech))
                    messages.append((sanitize_text(current_speaker), sanitized_speech))

                # Start a new speech
                current_speaker = line
                current_speech = []
            elif line and not line.startswith('[') and current_speaker:
                # Add this line to the current speech
                current_speech.append(line)

    # Add the last speech
    if current_speaker:
        sanitized_speech = sanitize_text(' '.join(current_speech))
        messages.append((sanitize_text(current_speaker), sanitized_speech))

    return messages


def get_hamilton_messages():
    file_path = 'hamilton.txt'
    script_dir = os.path.dirname(__file__)
    relative_path = os.path.join(script_dir, file_path)
    # Use the function
    return parse_script(relative_path)
