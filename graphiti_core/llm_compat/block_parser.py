def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith('```'):
        lines = stripped.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == '```':
            lines = lines[:-1]
        stripped = '\n'.join(lines)
    return stripped


def extract_item_blocks(text: str) -> list[str]:
    content = _strip_code_fences(text)
    begin_items = content.find('BEGIN ITEMS')
    if begin_items != -1:
        content = content[begin_items + len('BEGIN ITEMS') :]

    end_items = content.find('END ITEMS')
    if end_items != -1:
        content = content[:end_items]

    blocks: list[str] = []
    current: list[str] | None = None

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if line == 'BEGIN ITEM':
            if current:
                blocks.append('\n'.join(current))
            current = []
            continue
        if line == 'END ITEM':
            if current is not None:
                blocks.append('\n'.join(current))
                current = None
            continue
        if current is not None:
            current.append(raw_line)

    if current:
        blocks.append('\n'.join(current))

    return [block for block in blocks if parse_fields(block)]


def parse_fields(block: str) -> dict[str, str]:
    fields: dict[str, str] = {}

    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line or ':' not in line:
            continue

        key, value = line.split(':', 1)
        normalized_key = key.strip().upper()
        if not normalized_key:
            continue
        fields[normalized_key] = value.strip()

    return fields
