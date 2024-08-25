import os
import re
import unicodedata


def sanitize_string(text):
    # Normalize Unicode characters
    text = unicodedata.normalize('NFKD', text)
    # Replace specific problematic characters
    text = text.replace('â€™', "'")
    text = text.replace('â€œ', '"')
    text = text.replace('â€', '"')
    # Remove any remaining non-ASCII characters
    text = text.encode('ascii', 'ignore').decode('ascii')
    return text.strip()


def parse_debate(file_path):
    with open(file_path, encoding='utf-8') as file:
        content = file.read()

    # Split the content into episodes
    episodes = re.split(r'\n(?=(?:[A-Z]+):)', content)

    parsed_episodes = []
    for episode in episodes:
        # Extract the role and statement
        match = re.match(r'([A-Z]+):\s*(.*)', episode.strip(), re.DOTALL)
        if match:
            role, statement = match.groups()
            parsed_episodes.append(
                {'role': sanitize_string(role), 'statement': sanitize_string(statement)}
            )

    return parsed_episodes


def get_debate_messages():
    file_path = 'bush_gore_debate.txt'
    script_dir = os.path.dirname(__file__)
    relative_path = os.path.join(script_dir, file_path)
    return parse_debate(relative_path)


def get_kennedy_debate_messages():
    file_path = 'kennedy_nixon_debate.txt'
    script_dir = os.path.dirname(__file__)
    relative_path = os.path.join(script_dir, file_path)
    return parse_debate(relative_path)
