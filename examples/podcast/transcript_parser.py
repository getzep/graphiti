import os
import re
from datetime import datetime, timedelta, timezone

from pydantic import BaseModel


class Speaker(BaseModel):
    index: int
    name: str
    role: str


class ParsedMessage(BaseModel):
    speaker_index: int
    speaker_name: str
    role: str
    relative_timestamp: str
    actual_timestamp: datetime
    content: str


def parse_timestamp(timestamp: str) -> timedelta:
    if 'm' in timestamp:
        match = re.match(r'(\d+)m(?:\s*(\d+)s)?', timestamp)
        if match:
            minutes = int(match.group(1))
            seconds = int(match.group(2)) if match.group(2) else 0
            return timedelta(minutes=minutes, seconds=seconds)
    elif 's' in timestamp:
        match = re.match(r'(\d+)s', timestamp)
        if match:
            seconds = int(match.group(1))
            return timedelta(seconds=seconds)
    return timedelta()  # Return 0 duration if parsing fails


def parse_conversation_file(file_path: str, speakers: list[Speaker]) -> list[ParsedMessage]:
    with open(file_path) as file:
        content = file.read()

    messages = content.split('\n\n')
    speaker_dict = {speaker.index: speaker for speaker in speakers}

    parsed_messages: list[ParsedMessage] = []

    # Find the last timestamp to determine podcast duration
    last_timestamp = timedelta()
    for message in reversed(messages):
        lines = message.strip().split('\n')
        if lines:
            first_line = lines[0]
            parts = first_line.split(':', 1)
            if len(parts) == 2:
                header = parts[0]
                header_parts = header.split()
                if len(header_parts) >= 2:
                    timestamp = header_parts[1].strip('()')
                    last_timestamp = parse_timestamp(timestamp)
                    break

    # Calculate the start time
    now = datetime.now(timezone.utc)
    podcast_start_time = now - last_timestamp

    for message in messages:
        lines = message.strip().split('\n')
        if lines:
            first_line = lines[0]
            parts = first_line.split(':', 1)
            if len(parts) == 2:
                header, content = parts
                header_parts = header.split()
                if len(header_parts) >= 2:
                    speaker_index = int(header_parts[0])
                    timestamp = header_parts[1].strip('()')

                    if len(lines) > 1:
                        content += '\n' + '\n'.join(lines[1:])

                    delta = parse_timestamp(timestamp)
                    actual_time = podcast_start_time + delta

                    speaker = speaker_dict.get(speaker_index)
                    if speaker:
                        speaker_name = speaker.name
                        role = speaker.role
                    else:
                        speaker_name = f'Unknown Speaker {speaker_index}'
                        role = 'Unknown'

                    parsed_messages.append(
                        ParsedMessage(
                            speaker_index=speaker_index,
                            speaker_name=speaker_name,
                            role=role,
                            relative_timestamp=timestamp,
                            actual_timestamp=actual_time,
                            content=content.strip(),
                        )
                    )

    return parsed_messages


def parse_podcast_messages():
    file_path = 'podcast_transcript.txt'
    script_dir = os.path.dirname(__file__)
    relative_path = os.path.join(script_dir, file_path)

    speakers = [
        Speaker(index=0, name='Stephen DUBNER', role='Host'),
        Speaker(index=1, name='Tania Tetlow', role='Guest'),
        Speaker(index=4, name='Narrator', role='Narrator'),
        Speaker(index=5, name='Kamala Harris', role='Quoted'),
        Speaker(index=6, name='Unknown Speaker', role='Unknown'),
        Speaker(index=7, name='Unknown Speaker', role='Unknown'),
        Speaker(index=8, name='Unknown Speaker', role='Unknown'),
        Speaker(index=10, name='Unknown Speaker', role='Unknown'),
    ]

    parsed_conversation = parse_conversation_file(relative_path, speakers)
    print(f'Number of messages: {len(parsed_conversation)}')
    return parsed_conversation
