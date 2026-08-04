import re
from pathlib import Path

CANONICAL_SIGNATURE = 'I have read the CLA Document and I hereby sign the CLA'
WORKFLOW_PATH = Path(__file__).parents[1] / '.github' / 'workflows' / 'cla.yml'


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _cla_assistant_condition() -> str:
    lines = WORKFLOW_PATH.read_text().splitlines()
    step_index = next(
        index for index, line in enumerate(lines) if line.strip() == '- name: "CLA Assistant"'
    )
    step_indent = _indent(lines[step_index])

    for index in range(step_index + 1, len(lines)):
        line = lines[index]
        stripped = line.strip()
        if stripped and _indent(line) <= step_indent:
            break
        if not stripped.startswith('if:'):
            continue

        if_indent = _indent(line)
        value = stripped.removeprefix('if:').strip()
        if value not in {'>', '>-', '|', '|-'}:
            return value

        folded_lines: list[str] = []
        for continuation in lines[index + 1 :]:
            if continuation.strip() and _indent(continuation) <= if_indent:
                break
            if continuation.strip():
                folded_lines.append(continuation.strip())
        return ' '.join(folded_lines)

    raise AssertionError('CLA Assistant step has no if condition')


def _condition_runs(condition: str, event_name: str, comment_body: str | None) -> bool:
    clauses = [clause.strip().lstrip('(').strip() for clause in condition.split('||')]
    results: list[bool] = []

    for clause in clauses:
        event_match = re.fullmatch(
            r"github\.event_name\s*==\s*(['\"])pull_request_target\1\)*",
            clause,
        )
        if event_match:
            results.append(event_name == 'pull_request_target')
            continue

        exact_match = re.fullmatch(
            r"github\.event\.comment\.body\s*==\s*(['\"])(.*?)\1\)*",
            clause,
        )
        if exact_match:
            expected_body = exact_match.group(2)
            assert expected_body in {'recheck', CANONICAL_SIGNATURE}, (
                f'unsupported exact comment clause: {clause}'
            )
            results.append(comment_body == expected_body)
            continue

        starts_with_match = re.fullmatch(
            r"startsWith\(github\.event\.comment\.body,\s*(['\"])(.*?)\1\)\)*",
            clause,
        )
        if starts_with_match:
            prefix = starts_with_match.group(2)
            assert prefix == CANONICAL_SIGNATURE, f'unsupported signature prefix: {prefix}'
            results.append(comment_body is not None and comment_body.startswith(prefix))
            continue

        raise AssertionError(f'unsupported CLA condition clause: {clause}')

    return any(results)


def test_cla_assistant_trigger_contract() -> None:
    condition = _cla_assistant_condition()
    cases = (
        ('issue_comment', CANONICAL_SIGNATURE, True),
        ('issue_comment', f'{CANONICAL_SIGNATURE}, e-mail: user@example.com', True),
        (
            'issue_comment',
            f'{CANONICAL_SIGNATURE} behalf of my company, e-mail: user@example.com',
            True,
        ),
        ('issue_comment', 'recheck', True),
        ('issue_comment', 'unrelated body', False),
        ('pull_request_target', None, True),
        ('pull_request_target', 'arbitrary body', True),
    )

    for event_name, comment_body, expected in cases:
        observed = _condition_runs(condition, event_name, comment_body)
        assert observed is expected, f'{event_name=}, {comment_body=}, {expected=}'
