import os
import re


def parse_wizard_of_oz(file_path):
    with open(file_path, encoding='utf-8') as file:
        content = file.read()

    # Split the content into chapters
    chapters = re.split(r'\n\n+Chapter [IVX]+\n', content)[
        1:
    ]  # Skip the first split which is before Chapter I

    episodes = []
    for i, chapter in enumerate(chapters, start=1):
        # Extract chapter title
        title_match = re.match(r'(.*?)\n\n', chapter)
        title = title_match.group(1) if title_match else f'Chapter {i}'

        # Remove the title from the chapter content
        chapter_content = chapter[len(title) :].strip() if title_match else chapter.strip()

        # Create episode dictionary
        episode = {'episode_number': i, 'title': title, 'content': chapter_content}
        episodes.append(episode)

    return episodes


def get_wizard_of_oz_messages():
    file_path = 'woo.txt'
    script_dir = os.path.dirname(__file__)
    relative_path = os.path.join(script_dir, file_path)
    # Use the function
    parsed_episodes = parse_wizard_of_oz(relative_path)
    return parsed_episodes
