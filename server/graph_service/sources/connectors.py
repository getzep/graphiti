from __future__ import annotations

import asyncio
import io
import json
import zipfile
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import httpx
from pypdf import PdfReader
from pypdf.errors import PyPdfError

from graph_service.config import Settings

from .models import SourceDocument

TEXT_SUFFIXES = {
    '.txt',
    '.md',
    '.markdown',
    '.csv',
    '.tsv',
    '.json',
    '.jsonl',
    '.ndjson',
    '.yaml',
    '.yml',
    '.xml',
    '.html',
    '.htm',
    '.log',
}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | {'.docx', '.pdf'}
MAX_DOCX_XML_BYTES = 32 * 1024 * 1024
MAX_PDF_INPUT_BYTES = 25 * 1024 * 1024
MAX_PDF_PAGES = 500
MAX_PDF_PAGE_TEXT_CHARS = 200_000
MAX_PDF_TEXT_CHARS = 2_000_000
DEFAULT_MAX_LOCAL_FILE_BYTES = 25 * 1024 * 1024


def _byte_limit_label(size: int) -> str:
    if size >= 1024 * 1024 and size % (1024 * 1024) == 0:
        return f'{size // (1024 * 1024)} MiB'
    return f'{size} 字节'


class ConnectorError(RuntimeError):
    pass


class UnsupportedFileError(ConnectorError):
    pass


class _PDFPageTextLimitReached(RuntimeError):
    pass


class _PDFTotalTextLimitReached(RuntimeError):
    pass


class _HTMLTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag.casefold() in {'script', 'style'}:
            self._ignored_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() in {'script', 'style'} and self._ignored_depth:
            self._ignored_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth and data.strip():
            self.parts.append(data.strip())


def _decode_text(data: bytes) -> str:
    for encoding in ('utf-8-sig', 'utf-16', 'gb18030'):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode('utf-8', errors='replace')


def _extract_docx(data: bytes) -> str:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            info = archive.getinfo('word/document.xml')
            if info.file_size > MAX_DOCX_XML_BYTES:
                raise UnsupportedFileError('DOCX 正文解压后过大')
            document = archive.read('word/document.xml')
    except (KeyError, zipfile.BadZipFile) as exc:
        raise UnsupportedFileError('无效的 DOCX 文件') from exc

    try:
        root = ElementTree.fromstring(document)
    except ElementTree.ParseError as exc:
        raise UnsupportedFileError('DOCX 正文 XML 无效') from exc
    namespace = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
    paragraphs: list[str] = []
    for paragraph in root.iter(f'{namespace}p'):
        text = ''.join(node.text or '' for node in paragraph.iter(f'{namespace}t')).strip()
        if text:
            paragraphs.append(text)
    return '\n'.join(paragraphs)


def _extract_pdf(data: bytes, *, max_input_bytes: int | None = None) -> str:
    input_limit = max_input_bytes if max_input_bytes is not None else MAX_PDF_INPUT_BYTES
    if len(data) > input_limit:
        raise UnsupportedFileError(f'PDF 文件过大；解析器最多接受 {_byte_limit_label(input_limit)}')
    # The PDF header must occur within the first 1024 bytes. Rejecting other input before
    # invoking the parser keeps renamed binaries away from its more expensive recovery path.
    if b'%PDF-' not in data[:1024]:
        raise UnsupportedFileError('无效的 PDF 文件：缺少 PDF 文件头')

    try:
        # strict=False tolerates common producer quirks (for example a slightly broken xref)
        # while the explicit input/page/text limits still bound parser work.
        reader = PdfReader(io.BytesIO(data), strict=False)
        if reader.is_encrypted:
            raise UnsupportedFileError('PDF 已加密或受密码保护；请先解密后再上传')
        page_count = len(reader.pages)
    except UnsupportedFileError:
        raise
    except (PyPdfError, OSError, ValueError, TypeError, KeyError, RecursionError) as exc:
        raise UnsupportedFileError('无效或已损坏的 PDF 文件') from exc

    if page_count == 0:
        raise UnsupportedFileError('PDF 不包含任何页面')
    if page_count > MAX_PDF_PAGES:
        raise UnsupportedFileError(f'PDF 页数过多；最多支持 {MAX_PDF_PAGES} 页')

    pages: list[str] = []
    total_chars = 0
    for page_number, page in enumerate(reader.pages, start=1):
        visited_chars = 0

        def enforce_text_limits(
            text: str,
            *_args: Any,
            _total_before_page: int = total_chars,
            _separator_chars: int = 2 if pages else 0,
        ) -> None:
            nonlocal visited_chars
            visited_chars += len(text)
            if visited_chars > MAX_PDF_PAGE_TEXT_CHARS:
                raise _PDFPageTextLimitReached
            if _total_before_page + _separator_chars + visited_chars > MAX_PDF_TEXT_CHARS:
                raise _PDFTotalTextLimitReached

        try:
            text = (page.extract_text(visitor_text=enforce_text_limits) or '').replace('\x00', '')
        except _PDFPageTextLimitReached as exc:
            raise UnsupportedFileError(
                f'PDF 第 {page_number} 页文本过长；单页最多支持 {MAX_PDF_PAGE_TEXT_CHARS} 个字符'
            ) from exc
        except _PDFTotalTextLimitReached as exc:
            raise UnsupportedFileError(
                f'PDF 提取文本过长；最多支持 {MAX_PDF_TEXT_CHARS} 个字符'
            ) from exc
        except (PyPdfError, OSError, ValueError, TypeError, KeyError, RecursionError) as exc:
            raise UnsupportedFileError(f'PDF 第 {page_number} 页文本提取失败') from exc

        text = text.strip()
        if len(text) > MAX_PDF_PAGE_TEXT_CHARS:
            raise UnsupportedFileError(
                f'PDF 第 {page_number} 页文本过长；单页最多支持 {MAX_PDF_PAGE_TEXT_CHARS} 个字符'
            )
        if not text:
            continue
        separator_chars = 2 if pages else 0
        if total_chars + separator_chars + len(text) > MAX_PDF_TEXT_CHARS:
            raise UnsupportedFileError(f'PDF 提取文本过长；最多支持 {MAX_PDF_TEXT_CHARS} 个字符')
        pages.append(text)
        total_chars += separator_chars + len(text)

    if not pages:
        raise UnsupportedFileError('PDF 未检测到可提取文本；可能是扫描件，请先进行 OCR 后上传')
    return '\n\n'.join(pages)


def extract_file_content(filename: str, data: bytes, *, max_input_bytes: int | None = None) -> str:
    suffix = Path(filename).suffix.casefold()
    if suffix not in SUPPORTED_SUFFIXES:
        supported = ', '.join(sorted(SUPPORTED_SUFFIXES))
        raise UnsupportedFileError(f'不支持 {suffix or "无扩展名"} 文件；当前支持：{supported}')
    if suffix == '.docx':
        return _extract_docx(data)
    if suffix == '.pdf':
        return _extract_pdf(data, max_input_bytes=max_input_bytes)

    text = _decode_text(data).replace('\x00', '')
    if suffix == '.json':
        try:
            return json.dumps(json.loads(text), ensure_ascii=False, indent=2, sort_keys=True)
        except json.JSONDecodeError:
            return text
    if suffix in {'.html', '.htm'}:
        parser = _HTMLTextExtractor()
        parser.feed(text)
        return '\n'.join(parser.parts)
    if suffix == '.xml':
        try:
            root = ElementTree.fromstring(text)
            return '\n'.join(part.strip() for part in root.itertext() if part.strip())
        except ElementTree.ParseError:
            return text
    return text


def _as_datetime(value: Any, *, milliseconds: bool = False) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, int | float) or (isinstance(value, str) and value.isdigit()):
        timestamp = int(value)
        if milliseconds or timestamp > 10_000_000_000:
            timestamp /= 1000
        return datetime.fromtimestamp(timestamp, tz=timezone.utc)
    if isinstance(value, str) and value:
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc)


class SourceConnector(ABC):
    def __init__(self):
        self.warnings: list[str] = []
        self.errors: list[str] = []
        self.seen_external_ids: set[str] = set()
        self.inventory_complete = True

    @abstractmethod
    async def fetch(
        self, *, watermark_ms: int | None = None, full_sync: bool = False
    ) -> list[SourceDocument]: ...


class LocalConnector(SourceConnector):
    def __init__(self, root: Path, *, max_file_bytes: int = DEFAULT_MAX_LOCAL_FILE_BYTES):
        super().__init__()
        self.root = root
        self.max_file_bytes = max_file_bytes

    async def fetch(
        self, *, watermark_ms: int | None = None, full_sync: bool = False
    ) -> list[SourceDocument]:
        del watermark_ms, full_sync
        documents: list[SourceDocument] = []
        if not self.root.exists():
            return documents
        try:
            paths = await asyncio.to_thread(lambda: sorted(self.root.rglob('*')))
        except OSError as exc:
            self.inventory_complete = False
            self.errors.append(f'无法列举本地目录：{exc}')
            return documents

        for path in paths:
            try:
                # A local source directory may be writable by another process. Never follow a
                # symlink out of the configured upload root and ingest arbitrary server files.
                if path.is_symlink() or not path.is_file():
                    continue
                relative_name = path.relative_to(self.root).as_posix()
                if path.suffix.casefold() not in SUPPORTED_SUFFIXES:
                    self.warnings.append(f'{relative_name}: 暂不支持该文件类型')
                    continue
                self.seen_external_ids.add(relative_name)
                stat = await asyncio.to_thread(path.stat)
                if stat.st_size > self.max_file_bytes:
                    raise UnsupportedFileError(
                        f'文件大小超过限制（最多 {_byte_limit_label(self.max_file_bytes)}）'
                    )
                data = await asyncio.to_thread(path.read_bytes)
                # Check again after reading because another process may replace or grow the file
                # between stat() and read_bytes().
                if len(data) > self.max_file_bytes:
                    raise UnsupportedFileError(
                        f'文件大小超过限制（最多 {_byte_limit_label(self.max_file_bytes)}）'
                    )
                content = await asyncio.to_thread(
                    extract_file_content,
                    path.name,
                    data,
                    max_input_bytes=self.max_file_bytes,
                )
            except (OSError, UnsupportedFileError) as exc:
                label = path.relative_to(self.root).as_posix()
                self.errors.append(f'{label}: {exc}')
                continue
            documents.append(
                SourceDocument(
                    external_id=relative_name,
                    title=path.name,
                    content=content,
                    updated_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
                    remote_version=str(stat.st_mtime_ns),
                    metadata={'path': relative_name, 'size': stat.st_size},
                )
            )
        return documents


class FeishuConnector(SourceConnector):
    def __init__(
        self,
        settings: Settings,
        config: dict[str, Any],
        *,
        token_supplier: Callable[[], Awaitable[str]] | None = None,
    ):
        super().__init__()
        if token_supplier is None and (
            not settings.feishu_app_id or not settings.feishu_app_secret
        ):
            raise ConnectorError('请先设置 FEISHU_APP_ID 和 FEISHU_APP_SECRET')
        self.app_id = settings.feishu_app_id
        self.app_secret = settings.feishu_app_secret
        self.token_supplier = token_supplier
        self.base_url = settings.feishu_base_url.rstrip('/')
        self.folder_token = str(config.get('folder_token') or '').strip()
        self.root_folder = bool(config.get('root_folder', False))
        raw_document_tokens = config.get('document_tokens') or []
        if not isinstance(raw_document_tokens, list):
            raise ConnectorError('document_tokens 必须是列表')
        self.document_tokens = list(
            dict.fromkeys(str(token).strip() for token in raw_document_tokens if str(token).strip())
        )
        raw_document_metadata = config.get('document_metadata') or {}
        self.document_metadata = (
            raw_document_metadata if isinstance(raw_document_metadata, dict) else {}
        )
        self.recursive = bool(config.get('recursive', True))
        self.max_file_bytes = settings.max_upload_bytes
        if not self.root_folder and not self.folder_token and not self.document_tokens:
            raise ConnectorError('飞书数据源需要选择根目录、文件夹或文档')

    @staticmethod
    def _check(payload: dict[str, Any], operation: str) -> dict[str, Any]:
        if payload.get('code', 0) != 0:
            raise ConnectorError(f'{operation}失败：{payload.get("msg") or payload.get("message")}')
        return payload.get('data') or {}

    async def _token(self, client: httpx.AsyncClient) -> str:
        if self.token_supplier is not None:
            token = await self.token_supplier()
            if not token:
                raise ConnectorError('飞书 OAuth 连接未返回访问凭证')
            return token
        response = await client.post(
            '/auth/v3/tenant_access_token/internal',
            json={'app_id': self.app_id, 'app_secret': self.app_secret},
        )
        response.raise_for_status()
        payload = response.json()
        self._check(payload, '获取飞书 tenant_access_token')
        token = payload.get('tenant_access_token')
        if not token:
            raise ConnectorError('飞书认证响应中没有 tenant_access_token')
        return token

    async def _list_files(
        self, client: httpx.AsyncClient, headers: dict[str, str]
    ) -> list[dict[str, Any]]:
        if not self.root_folder and not self.folder_token:
            return []
        files: list[dict[str, Any]] = []
        folders: list[str | None] = [self.folder_token or None]
        visited_folders: set[str] = set()
        seen_files: set[str] = set()
        while folders:
            folder_token = folders.pop(0)
            folder_key = folder_token or '__root__'
            if folder_key in visited_folders:
                continue
            visited_folders.add(folder_key)
            page_token: str | None = None
            while True:
                params: dict[str, Any] = {'order_by': 'EditedTime', 'direction': 'DESC'}
                if folder_token:
                    params['folder_token'] = folder_token
                    params['page_size'] = 200
                if page_token and folder_token:
                    params['page_token'] = page_token
                response = await client.get('/drive/v1/files', headers=headers, params=params)
                response.raise_for_status()
                data = self._check(response.json(), '列举飞书目录')
                page_files = data.get('files') or []
                for item in page_files:
                    token = str(item.get('token') or '')
                    if token and token not in seen_files:
                        seen_files.add(token)
                        files.append(item)
                if self.recursive:
                    folders.extend(
                        str(item['token'])
                        for item in page_files
                        if item.get('type') == 'folder' and item.get('token')
                    )
                if not folder_token or not data.get('has_more'):
                    break
                page_token = data.get('next_page_token')
                if not page_token:
                    break
        return files

    async def _docx(
        self,
        client: httpx.AsyncClient,
        headers: dict[str, str],
        token: str,
        item: dict[str, Any],
    ) -> SourceDocument:
        metadata_response = await client.get(f'/docx/v1/documents/{token}', headers=headers)
        metadata_response.raise_for_status()
        metadata = self._check(metadata_response.json(), '读取飞书文档元数据').get('document') or {}

        content_response = await client.get(
            f'/docx/v1/documents/{token}/raw_content', headers=headers
        )
        content_response.raise_for_status()
        content_data = self._check(content_response.json(), '读取飞书文档正文')
        updated = item.get('modified_time') or metadata.get('updated_at')
        return SourceDocument(
            external_id=token,
            title=str(metadata.get('title') or item.get('name') or token),
            content=str(content_data.get('content') or ''),
            updated_at=_as_datetime(updated),
            remote_version=str(metadata.get('revision_id') or updated or ''),
            url=item.get('url'),
            metadata={'feishu_type': 'docx', 'revision_id': metadata.get('revision_id')},
        )

    async def _file(
        self,
        client: httpx.AsyncClient,
        headers: dict[str, str],
        token: str,
        item: dict[str, Any],
    ) -> SourceDocument | None:
        name = str(item.get('name') or token)
        if Path(name).suffix.casefold() not in SUPPORTED_SUFFIXES:
            self.warnings.append(f'{name}: 暂不支持该飞书文件类型')
            return None
        content = bytearray()
        async with client.stream(
            'GET', f'/drive/v1/files/{token}/download', headers=headers
        ) as response:
            response.raise_for_status()
            content_length = response.headers.get('content-length')
            if content_length and int(content_length) > self.max_file_bytes:
                raise ConnectorError(f'{name} 超过远端文件大小限制')
            async for chunk in response.aiter_bytes():
                content.extend(chunk)
                if len(content) > self.max_file_bytes:
                    raise ConnectorError(f'{name} 超过远端文件大小限制')
        updated = item.get('modified_time')
        return SourceDocument(
            external_id=token,
            title=name,
            content=await asyncio.to_thread(
                extract_file_content,
                name,
                bytes(content),
                max_input_bytes=self.max_file_bytes,
            ),
            updated_at=_as_datetime(updated),
            remote_version=str(updated or ''),
            url=item.get('url'),
            metadata={'feishu_type': 'file', 'size': len(content)},
        )

    async def fetch(
        self, *, watermark_ms: int | None = None, full_sync: bool = False
    ) -> list[SourceDocument]:
        del watermark_ms, full_sync
        documents: list[SourceDocument] = []
        async with httpx.AsyncClient(base_url=self.base_url, timeout=60) as client:
            token = await self._token(client)
            headers = {'Authorization': f'Bearer {token}'}
            files = await self._list_files(client, headers)
            files.extend(
                {
                    'token': token,
                    'type': str((self.document_metadata.get(token) or {}).get('type') or 'docx'),
                    'name': str((self.document_metadata.get(token) or {}).get('name') or token),
                }
                for token in self.document_tokens
                if not any(item.get('token') == token for item in files)
            )
            for item in files:
                token = str(item.get('token') or '')
                item_type = item.get('type')
                if not token or item_type == 'folder':
                    continue
                # Seeing an item in a complete directory listing is distinct from being able to
                # parse its latest contents. A transient content error must not tombstone it.
                self.seen_external_ids.add(token)
                try:
                    if item_type == 'docx':
                        documents.append(await self._docx(client, headers, token, item))
                    elif item_type == 'file':
                        document = await self._file(client, headers, token, item)
                        if document:
                            documents.append(document)
                    else:
                        self.warnings.append(
                            f'{item.get("name") or token}: 暂不支持飞书 {item_type} 类型'
                        )
                except (httpx.HTTPError, ConnectorError, UnsupportedFileError, ValueError) as exc:
                    self.errors.append(f'{item.get("name") or token}: {exc}')
        return documents


class MeegoConnector(SourceConnector):
    OVERLAP_MS = 5 * 60 * 1000

    def __init__(self, settings: Settings, config: dict[str, Any]):
        super().__init__()
        if not settings.meego_plugin_id or not settings.meego_plugin_secret:
            raise ConnectorError('请先设置 MEEGO_PLUGIN_ID 和 MEEGO_PLUGIN_SECRET')
        self.plugin_id = settings.meego_plugin_id
        self.plugin_secret = settings.meego_plugin_secret
        self.user_key = str(settings.meego_user_key or '').strip()
        if not self.user_key:
            raise ConnectorError('MeeGo 读取需要后端环境变量 MEEGO_USER_KEY')
        self.base_url = settings.meego_base_url.rstrip('/')
        self.project_key = str(config.get('project_key') or '').strip()
        if not self.project_key:
            raise ConnectorError('MeeGo 数据源需要 project_key')
        raw_work_item_types = config.get('work_item_type_keys') or []
        if not isinstance(raw_work_item_types, list):
            raise ConnectorError('work_item_type_keys 必须是列表')
        self.work_item_types = list(
            dict.fromkeys(str(value).strip() for value in raw_work_item_types if str(value).strip())
        )
        try:
            self.page_size = min(100, max(1, int(config.get('page_size', 100))))
        except (TypeError, ValueError) as exc:
            raise ConnectorError('page_size 必须是 1 到 100 的整数') from exc

    @staticmethod
    def _check(payload: dict[str, Any], operation: str) -> Any:
        error = payload.get('error') or {}
        code = error.get('code', payload.get('err_code', payload.get('code', 0)))
        if code not in (0, '0', None):
            message = error.get('msg') or payload.get('err_msg') or payload.get('msg')
            raise ConnectorError(f'{operation}失败：{message or code}')
        return payload.get('data')

    async def _token(self, client: httpx.AsyncClient) -> str:
        response = await client.post(
            '/authen/plugin_token',
            json={'plugin_id': self.plugin_id, 'plugin_secret': self.plugin_secret, 'type': 0},
        )
        response.raise_for_status()
        data = self._check(response.json(), '获取 MeeGo plugin token') or {}
        token = data.get('token')
        if not token:
            raise ConnectorError('MeeGo 认证响应中没有 token')
        return token

    async def _types(self, client: httpx.AsyncClient, headers: dict[str, str]) -> list[str]:
        if self.work_item_types:
            return self.work_item_types
        response = await client.get(f'/{self.project_key}/work_item/all-types', headers=headers)
        response.raise_for_status()
        data = self._check(response.json(), '读取 MeeGo 工作项类型') or []
        types = [
            str(item.get('type_key'))
            for item in data
            if item.get('type_key') and not item.get('is_disabled')
        ]
        if not types:
            raise ConnectorError('MeeGo 项目没有可读取的工作项类型')
        return list(dict.fromkeys(types))

    @staticmethod
    def _render(item: dict[str, Any]) -> str:
        payload = {
            'id': item.get('id'),
            'name': item.get('name'),
            'project_key': item.get('project_key'),
            'work_item_type_key': item.get('work_item_type_key'),
            'status': item.get('work_item_status'),
            'current_nodes': item.get('current_nodes'),
            'fields': item.get('fields'),
            'multi_texts': item.get('multi_texts'),
            'created_by': item.get('created_by'),
            'updated_by': item.get('updated_by'),
            'created_at': item.get('created_at'),
            'updated_at': item.get('updated_at'),
            'deleted_at': item.get('deleted_at'),
        }
        return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)

    async def fetch(
        self, *, watermark_ms: int | None = None, full_sync: bool = False
    ) -> list[SourceDocument]:
        documents_by_id: dict[str, SourceDocument] = {}
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        async with httpx.AsyncClient(base_url=self.base_url, timeout=60) as client:
            token = await self._token(client)
            headers = {'X-Plugin-Token': token, 'X-User-Key': self.user_key}
            work_item_types = await self._types(client, headers)
            page = 1
            while True:
                body: dict[str, Any] = {
                    'work_item_type_keys': work_item_types,
                    'page_num': page,
                    'page_size': self.page_size,
                    'expand': {
                        'need_multi_text': True,
                        'need_rich_text_mark_down': True,
                    },
                }
                if watermark_ms and not full_sync:
                    body['updated_at'] = {
                        'start': max(0, watermark_ms - self.OVERLAP_MS),
                        'end': now_ms,
                    }
                response = await client.post(
                    f'/{self.project_key}/work_item/filter', headers=headers, json=body
                )
                response.raise_for_status()
                payload = response.json()
                items = self._check(payload, '拉取 MeeGo 工作项') or []
                for item in items:
                    work_item_id = item.get('id')
                    if work_item_id in (None, ''):
                        self.errors.append('MeeGo 返回了缺少 id 的工作项，已跳过')
                        self.inventory_complete = False
                        continue
                    work_item_type = item.get('work_item_type_key') or 'unknown'
                    try:
                        updated_at = int(item.get('updated_at') or item.get('created_at') or now_ms)
                    except (TypeError, ValueError):
                        self.errors.append(f'MeeGo 工作项 {work_item_id} 的更新时间无效，已跳过')
                        self.inventory_complete = False
                        continue
                    external_id = f'{self.project_key}:{work_item_type}:{work_item_id}'
                    self.seen_external_ids.add(external_id)
                    document = SourceDocument(
                        external_id=external_id,
                        title=str(item.get('name') or f'MeeGo #{work_item_id}'),
                        content=self._render(item),
                        updated_at=_as_datetime(updated_at, milliseconds=True),
                        remote_version=str(updated_at),
                        metadata={
                            'project_key': self.project_key,
                            'work_item_type_key': work_item_type,
                            'work_item_id': work_item_id,
                        },
                    )
                    previous = documents_by_id.get(external_id)
                    if previous is None or document.updated_at >= previous.updated_at:
                        documents_by_id[external_id] = document
                pagination = payload.get('pagination') or {}
                total = int(pagination.get('total') or 0)
                if not items or page * self.page_size >= total:
                    break
                page += 1
        return list(documents_by_id.values())


def _find_first_key(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        if key in value:
            return value[key]
        for child in value.values():
            found = _find_first_key(child, key)
            if found not in (None, ''):
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_first_key(child, key)
            if found not in (None, ''):
                return found
    return None


def _meego_type_records(value: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()

    def visit(node: Any) -> None:
        if isinstance(node, list):
            for child in node:
                visit(child)
            return
        if not isinstance(node, dict):
            return
        type_key = node.get('type_key') or node.get('work_item_type_key')
        if type_key and str(type_key) not in seen:
            seen.add(str(type_key))
            records.append(node)
        for child in node.values():
            if isinstance(child, dict | list):
                visit(child)

    visit(value)
    return records


def _meego_work_item_records(value: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()

    def visit(node: Any) -> None:
        if isinstance(node, list):
            for child in node:
                visit(child)
            return
        if not isinstance(node, dict):
            return
        work_item_id = node.get('work_item_id') or node.get('id')
        if work_item_id not in (None, '') and any(
            key in node for key in ('name', 'title', 'fields', 'updated_at', 'created_at')
        ):
            marker = str(work_item_id)
            if marker not in seen:
                seen.add(marker)
                records.append(node)
        for child in node.values():
            if isinstance(child, dict | list):
                visit(child)

    visit(value)
    return records


def _meego_view_records(value: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()

    def visit(node: Any) -> None:
        if isinstance(node, list):
            for child in node:
                visit(child)
            return
        if not isinstance(node, dict):
            return
        attribute = node.get('work_item_attribute')
        if isinstance(attribute, dict):
            work_item_id = attribute.get('work_item_id') or attribute.get('id')
            if work_item_id not in (None, '') and str(work_item_id) not in seen:
                seen.add(str(work_item_id))
                records.append(node)
                return
        for child in node.values():
            if isinstance(child, dict | list):
                visit(child)

    visit(value)
    return records


class MeegoOAuthConnector(SourceConnector):
    """Read MeeGo through its user OAuth MCP data plane."""

    def __init__(self, config: dict[str, Any], connection_id: str, connection_manager: Any):
        super().__init__()
        self.connection_id = connection_id
        self.connection_manager = connection_manager
        self.view_url = str(config.get('view_url') or '').strip()
        self.view_id = str(config.get('view_id') or '').strip()
        self.view_work_item_type = str(config.get('work_item_type_key') or '').strip()
        self.project_key = str(config.get('project_key') or '').strip()
        self.project_name = str(config.get('project_name') or self.project_key).strip()
        self.view_name = str(config.get('view_name') or self.view_id).strip()
        if not self.project_key or '`' in self.project_key:
            raise ConnectorError('MeeGo 项目标识无效')
        if self.view_url and (not self.view_id or not self.view_work_item_type):
            raise ConnectorError('MeeGo 视图配置不完整')
        if any('`' in value for value in (self.view_id, self.view_work_item_type)):
            raise ConnectorError('MeeGo 视图标识无效')
        raw_types = config.get('work_item_type_keys') or []
        if not isinstance(raw_types, list):
            raise ConnectorError('work_item_type_keys 必须是列表')
        self.work_item_types = list(
            dict.fromkeys(str(value).strip() for value in raw_types if str(value).strip())
        )
        if any('`' in value for value in self.work_item_types):
            raise ConnectorError('MeeGo 工作项类型标识无效')

    async def _call(
        self,
        resource: str,
        method: str,
        fallback: str,
        arguments: dict[str, Any],
    ) -> Any:
        return await self.connection_manager.meego_call(
            self.connection_id,
            resource,
            method,
            fallback,
            arguments,
        )

    async def _types(self) -> list[str]:
        if self.work_item_types:
            return self.work_item_types
        value = await self._call(
            'workitem',
            'meta-types',
            'list_workitem_types',
            {'project_key': self.project_key},
        )
        types = [
            str(item.get('type_key') or item.get('work_item_type_key'))
            for item in _meego_type_records(value)
            if not item.get('is_disabled')
        ]
        types = [value for value in types if value and value != 'None']
        if not types:
            raise ConnectorError('MeeGo 项目没有可读取的工作项类型')
        return list(dict.fromkeys(types))

    async def _query_type(self, type_key: str) -> list[dict[str, Any]]:
        mql = (
            'SELECT `id`, `name`, `updated_at`, `created_at` '
            f'FROM `{self.project_key}`.`{type_key}` ORDER BY `updated_at` DESC'
        )
        first = await self._call(
            'workitem',
            'query',
            'search_by_mql',
            {'project_key': self.project_key, 'mql': mql},
        )
        records = _meego_work_item_records(first)
        session_id = str(_find_first_key(first, 'session_id') or '')
        group_infos = _find_first_key(first, 'group_infos')
        groups = (
            [item for item in group_infos if isinstance(item, dict)]
            if isinstance(group_infos, list)
            else []
        )
        if session_id and not groups and len(records) >= 50:
            groups = [{'group_id': '1', 'count': len(records) + 50}]

        for group in groups:
            group_id = str(group.get('group_id') or '1')
            total = int(group.get('count') or group.get('total') or 0)
            page = 2
            max_page = min(200, max(2, (total + 49) // 50 if total else 200))
            while page <= max_page:
                value = await self._call(
                    'workitem',
                    'query',
                    'search_by_mql',
                    {
                        'project_key': self.project_key,
                        'session_id': session_id,
                        'group_pagination_list': [{'group_id': group_id, 'page_num': page}],
                    },
                )
                page_records = _meego_work_item_records(value)
                if not page_records:
                    break
                records.extend(page_records)
                if len(page_records) < 50:
                    break
                page += 1

        deduplicated: dict[str, dict[str, Any]] = {}
        for record in records:
            work_item_id = record.get('work_item_id') or record.get('id')
            if work_item_id not in (None, ''):
                deduplicated[str(work_item_id)] = record
        return list(deduplicated.values())

    async def _query_view(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        page = 1
        while page <= 200:
            arguments: dict[str, Any] = {
                'project_key': self.project_key,
                'view_id': self.view_id,
                'page_num': page,
            }
            if self.view_url:
                arguments['url'] = self.view_url
            value = await self._call(
                'view',
                'detail',
                'get_view_detail',
                arguments,
            )
            page_records = _meego_view_records(value)
            records.extend(page_records)
            pagination = _find_first_key(value, 'pagination')
            has_more = pagination.get('has_more') if isinstance(pagination, dict) else None
            if has_more is False or (has_more is None and len(page_records) < 50):
                break
            if not page_records:
                break
            page += 1

        deduplicated: dict[str, dict[str, Any]] = {}
        for record in records:
            attribute = record.get('work_item_attribute') or {}
            work_item_id = attribute.get('work_item_id') or attribute.get('id')
            if work_item_id not in (None, ''):
                deduplicated[str(work_item_id)] = record
        return list(deduplicated.values())

    async def fetch(
        self, *, watermark_ms: int | None = None, full_sync: bool = False
    ) -> list[SourceDocument]:
        del watermark_ms, full_sync
        documents: list[SourceDocument] = []
        view_items: list[dict[str, Any]] = []
        view_updated_at: list[datetime] = []
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        rows: list[tuple[str, dict[str, Any]]] = []
        if self.view_id:
            for row in await self._query_view():
                attribute = row.get('work_item_attribute') or {}
                work_item_type = attribute.get('work_item_type') or {}
                type_key = str(work_item_type.get('key') or self.view_work_item_type)
                rows.append((type_key, row))
        else:
            for type_key in await self._types():
                rows.extend((type_key, row) for row in await self._query_type(type_key))

        for type_key, row in rows:
            attribute = row.get('work_item_attribute') or {}
            work_item_id = (
                attribute.get('work_item_id')
                or attribute.get('id')
                or row.get('work_item_id')
                or row.get('id')
            )
            if work_item_id in (None, ''):
                continue
            external_id = f'{self.project_key}:{type_key}:{work_item_id}'
            try:
                brief = await self._call(
                    'workitem',
                    'get',
                    'get_workitem_brief',
                    {
                        'project_key': self.project_key,
                        'work_item_id': str(work_item_id),
                        'fields': ['_all'],
                        'page_size': 200,
                    },
                )
            except Exception as exc:
                self.errors.append(f'MeeGo 工作项 {work_item_id}: {exc}')
                self.inventory_complete = False
                continue
            brief_payload = brief if isinstance(brief, dict) else row
            payload = (
                {'work_item': brief_payload, 'view_context': row}
                if self.view_id
                else brief_payload
            )
            updated_value = (
                attribute.get('update_time')
                or attribute.get('updated_at')
                or _find_first_key(brief_payload, 'updated_at')
                or _find_first_key(brief_payload, 'update_time')
                or attribute.get('create_time')
                or _find_first_key(brief_payload, 'created_at')
                or now_ms
            )
            updated_at = _as_datetime(updated_value, milliseconds=True)
            title = str(
                attribute.get('work_item_name')
                or attribute.get('name')
                or brief_payload.get('name')
                or brief_payload.get('title')
                or row.get('name')
                or work_item_id
            )
            metadata = {
                'project_key': self.project_key,
                'work_item_type_key': type_key,
                'work_item_id': work_item_id,
            }
            if self.view_id:
                metadata.update({'view_id': self.view_id, 'view_url': self.view_url})
                view_items.append(
                    {
                        'title': title,
                        'work_item_type_key': type_key,
                        'work_item_id': work_item_id,
                        'work_item': brief_payload,
                        'view_context': row,
                    }
                )
                view_updated_at.append(updated_at)
                continue
            self.seen_external_ids.add(external_id)
            documents.append(
                SourceDocument(
                    external_id=external_id,
                    title=title,
                    content=json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
                    updated_at=updated_at,
                    remote_version=str(updated_value),
                    metadata=metadata,
                )
            )
        if self.view_id and view_items:
            # A selected view is the user's import boundary. Keep it as one versioned
            # source document so Graphiti can understand relationships across all rows
            # without paying one complete extraction pipeline per work item.
            external_id = (
                f'{self.project_key}:{self.view_work_item_type or "work_item"}:view:{self.view_id}'
            )
            updated_at = max(view_updated_at)
            self.seen_external_ids.add(external_id)
            documents.append(
                SourceDocument(
                    external_id=external_id,
                    title=f'{self.project_name} · {self.view_name}',
                    content=json.dumps(
                        {
                            'project_key': self.project_key,
                            'project_name': self.project_name,
                            'view_id': self.view_id,
                            'view_name': self.view_name,
                            'work_items': view_items,
                        },
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                    ),
                    updated_at=updated_at,
                    remote_version=str(int(updated_at.timestamp() * 1000)),
                    metadata={
                        'project_key': self.project_key,
                        'work_item_type_key': self.view_work_item_type,
                        'view_id': self.view_id,
                        'view_url': self.view_url,
                        'item_count': len(view_items),
                    },
                )
            )
        return documents


def build_connector(
    source: dict[str, Any],
    settings: Settings,
    upload_root: Path,
    connection_manager: Any | None = None,
) -> SourceConnector:
    kind = source['kind']
    if kind == 'local':
        return LocalConnector(upload_root / source['id'], max_file_bytes=settings.max_upload_bytes)
    if kind == 'feishu':
        if source.get('connection_id'):
            if connection_manager is None:
                raise ConnectorError('OAuth 连接管理器尚未就绪')
            connection_id = str(source['connection_id'])
            return FeishuConnector(
                settings,
                source['config'],
                token_supplier=lambda: connection_manager.get_access_token(connection_id, 'feishu'),
            )
        return FeishuConnector(settings, source['config'])
    if kind == 'meego':
        if source.get('connection_id'):
            if connection_manager is None:
                raise ConnectorError('OAuth 连接管理器尚未就绪')
            return MeegoOAuthConnector(
                source['config'],
                str(source['connection_id']),
                connection_manager,
            )
        return MeegoConnector(settings, source['config'])
    raise ConnectorError(f'不支持的数据源类型：{kind}')
