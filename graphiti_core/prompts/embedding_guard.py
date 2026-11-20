import ast
import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

KNOWN_EMBEDDING_FIELDS: frozenset[str] = frozenset({'name_embedding', 'fact_embedding'})
CANARY_VALUE: float = 0.0123456789
FLOAT_RUN_MIN: int = 64

_DECIMAL = r'[+-]?(?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?'
_EMBEDDING_FIELD_PATTERN = '|'.join(
    re.escape(name) for name in sorted(KNOWN_EMBEDDING_FIELDS, key=len, reverse=True)
)
_KEYED_VECTOR_RE = re.compile(rf"""["']?\b({_EMBEDDING_FIELD_PATTERN})\b["']?\s*[:=]\s*\[\s*-?\d""")
_FLOAT_RUN_RE = re.compile(rf'(?<![\w.]){_DECIMAL}(?:\s*,\s*{_DECIMAL}){{{FLOAT_RUN_MIN - 1},}}')


@dataclass(frozen=True)
class FloatListField:
    path: str
    cls: str
    field: str
    line: int


def _annotation_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _is_none_annotation(node: ast.expr) -> bool:
    return (isinstance(node, ast.Constant) and node.value is None) or (
        _annotation_name(node) == 'None'
    )


def _is_float_annotation(node: ast.expr) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        try:
            return _is_float_annotation(ast.parse(node.value, mode='eval').body)
        except SyntaxError:
            return False
    return _annotation_name(node) == 'float'


def _is_float_list_annotation(node: ast.expr) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        try:
            return _is_float_list_annotation(ast.parse(node.value, mode='eval').body)
        except SyntaxError:
            return False
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _is_float_list_annotation(node.left) or _is_float_list_annotation(node.right)
    if not isinstance(node, ast.Subscript):
        return False

    name = _annotation_name(node.value)
    arguments = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
    if name in ('Annotated', 'Optional'):
        return bool(arguments) and _is_float_list_annotation(arguments[0])
    if name == 'Union':
        return any(
            _is_float_list_annotation(argument)
            for argument in arguments
            if not _is_none_annotation(argument)
        )
    if name in ('list', 'List', 'Sequence'):
        return len(arguments) == 1 and _is_float_annotation(arguments[0])
    if name in ('tuple', 'Tuple'):
        return (
            len(arguments) == 2
            and _is_float_annotation(arguments[0])
            and isinstance(arguments[1], ast.Constant)
            and arguments[1].value is Ellipsis
        )
    return False


def find_float_list_fields(package_dir: Path) -> list[FloatListField]:
    fields: list[FloatListField] = []
    for source_path in sorted(package_dir.rglob('*.py')):
        module = ast.parse(source_path.read_text(encoding='utf-8'), filename=str(source_path))
        relative_path = source_path.relative_to(package_dir).as_posix()

        def visit_class(
            class_node: ast.ClassDef,
            parents: tuple[str, ...] = (),
            path: str = relative_path,
        ) -> None:
            class_path = (*parents, class_node.name)
            class_name = '.'.join(class_path)
            for statement in class_node.body:
                if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
                    if _is_float_list_annotation(statement.annotation):
                        fields.append(
                            FloatListField(
                                path=path,
                                cls=class_name,
                                field=statement.target.id,
                                line=statement.lineno,
                            )
                        )
                elif isinstance(statement, ast.ClassDef):
                    visit_class(statement, class_path)

        for statement in module.body:
            if isinstance(statement, ast.ClassDef):
                visit_class(statement)

    return fields


def canary_vector(dim: int = 768) -> list[float]:
    return [CANARY_VALUE] * dim


def with_canary_embeddings(model: BaseModel) -> BaseModel:
    updates = {
        name: canary_vector()
        for name in KNOWN_EMBEDDING_FIELDS
        if name in model.__class__.model_fields
    }
    return model.model_copy(update=updates)


@dataclass(frozen=True)
class EmbeddingLeak:
    rule: Literal['canary', 'keyed_vector', 'float_run']
    field: str | None
    offset: int


def find_embedding_leaks(text: str) -> list[EmbeddingLeak]:
    leaks: list[EmbeddingLeak] = []
    canary_match = text.find(str(CANARY_VALUE))
    if canary_match >= 0:
        leaks.append(EmbeddingLeak('canary', None, canary_match))

    leaks.extend(
        EmbeddingLeak('keyed_vector', match.group(1), match.start(1))
        for match in _KEYED_VECTOR_RE.finditer(text)
    )
    leaks.extend(
        EmbeddingLeak('float_run', None, match.start()) for match in _FLOAT_RUN_RE.finditer(text)
    )
    return sorted(leaks, key=lambda leak: (leak.offset, leak.rule))


@dataclass(frozen=True)
class PromptLeakRecord:
    boundary: Literal['message_create', 'message_update', 'gateway_generate', 'gateway_decide']
    role: str | None
    prompt_name: str | None
    leak: EmbeddingLeak
    excerpt: str


@dataclass
class PromptScan:
    records: list[PromptLeakRecord] = field(default_factory=list)

    def check(
        self,
        text: str,
        *,
        boundary: Literal['message_create', 'message_update', 'gateway_generate', 'gateway_decide'],
        role: str | None = None,
    ) -> None:
        try:
            leaks = find_embedding_leaks(text)
        except Exception:
            return

        try:
            from graphiti_core.llm_client.client import current_prompt_name

            prompt_name = current_prompt_name.get()
        except Exception:
            prompt_name = None
        for leak in leaks:
            start = max(0, leak.offset - 60)
            end = min(len(text), leak.offset + 60)
            self.records.append(
                PromptLeakRecord(
                    boundary=boundary,
                    role=role,
                    prompt_name=prompt_name,
                    leak=leak,
                    excerpt=text[start:end],
                )
            )

    def format_failure(self) -> str:
        lines = ['Embedding vectors were found in LLM prompts:']
        for record in self.records:
            lines.append(
                f'- boundary={record.boundary} role={record.role!r} '
                f'prompt={record.prompt_name!r} rule={record.leak.rule} '
                f'field={record.leak.field!r} offset={record.leak.offset} '
                f'excerpt={record.excerpt!r}'
            )
        return '\n'.join(lines)


@contextmanager
def scan_message_prompts() -> Iterator[PromptScan]:
    from graphiti_core.prompts.models import Message

    scan = PromptScan()
    original_init = Message.__init__
    original_setattr = Message.__setattr__

    def scanned_init(self: Message, *args: object, **kwargs: object) -> None:
        original_init(self, *args, **kwargs)
        scan.check(self.content, boundary='message_create', role=self.role)

    def scanned_setattr(self: Message, name: str, value: object) -> None:
        original_setattr(self, name, value)
        if name == 'content' and isinstance(value, str):
            scan.check(value, boundary='message_update', role=self.role)

    Message.__init__ = scanned_init
    Message.__setattr__ = scanned_setattr
    try:
        yield scan
    finally:
        Message.__init__ = original_init
        Message.__setattr__ = original_setattr
