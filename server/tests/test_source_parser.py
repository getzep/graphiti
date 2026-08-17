import io
import zipfile

import pytest
from pydantic import ValidationError
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, StreamObject

from graph_service.sources import connectors
from graph_service.sources.connectors import (
    LocalConnector,
    UnsupportedFileError,
    extract_file_content,
)
from graph_service.sources.models import FileUploadRequest, SourceCreateRequest


def _docx(document_xml: bytes) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('word/document.xml', document_xml)
    return output.getvalue()


def _pdf(*page_texts: str, password: str | None = None) -> bytes:
    output = io.BytesIO()
    writer = PdfWriter()
    font = DictionaryObject(
        {
            NameObject('/Type'): NameObject('/Font'),
            NameObject('/Subtype'): NameObject('/Type1'),
            NameObject('/BaseFont'): NameObject('/Helvetica'),
        }
    )
    font_ref = writer._add_object(font)
    for text in page_texts:
        page = writer.add_blank_page(width=612, height=792)
        resources = DictionaryObject(
            {NameObject('/Font'): DictionaryObject({NameObject('/F1'): font_ref})}
        )
        page[NameObject('/Resources')] = resources
        if text:
            escaped = text.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')
            content = StreamObject()
            content.set_data(f'BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET'.encode('ascii'))
            page[NameObject('/Contents')] = writer._add_object(content)
    if password:
        writer.encrypt(password)
    writer.write(output)
    return output.getvalue()


def test_parser_normalizes_structured_text_and_ignores_active_html_content():
    assert extract_file_content('data.json', b'{"b": 2, "a": 1}') == (  # noqa: RUF001
        '{\n  "a": 1,\n  "b": 2\n}'
    )
    html = b'<style>secret-style</style><h1>Title</h1><script>secret-script</script><p>Body</p>'
    assert extract_file_content('page.html', html) == 'Title\nBody'


def test_docx_parser_extracts_paragraphs_and_rejects_bad_or_oversized_xml(monkeypatch):
    namespace = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
    valid = _docx(
        f'<w:document xmlns:w="{namespace}"><w:body>'
        '<w:p><w:r><w:t>第一段</w:t></w:r></w:p>'
        '<w:p><w:r><w:t>第二段</w:t></w:r></w:p>'
        '</w:body></w:document>'.encode()
    )
    assert extract_file_content('guide.docx', valid) == '第一段\n第二段'

    with pytest.raises(UnsupportedFileError, match='XML 无效'):
        extract_file_content('bad.docx', _docx(b'<broken'))

    monkeypatch.setattr(connectors, 'MAX_DOCX_XML_BYTES', 8)
    with pytest.raises(UnsupportedFileError, match='解压后过大'):
        extract_file_content('large.docx', _docx(b'123456789'))


def test_pdf_parser_extracts_text_across_pages():
    assert extract_file_content('guide.PDF', _pdf('first page', 'second page')) == (
        'first page\n\nsecond page'
    )


def test_pdf_parser_rejects_encrypted_invalid_and_scanned_files():
    with pytest.raises(UnsupportedFileError, match='加密|密码'):
        extract_file_content('secret.pdf', _pdf('secret', password='password'))
    with pytest.raises(UnsupportedFileError, match='文件头'):
        extract_file_content('renamed.pdf', b'not really a pdf')
    with pytest.raises(UnsupportedFileError, match='OCR'):
        extract_file_content('scan.pdf', _pdf(''))


def test_pdf_parser_enforces_input_page_and_text_limits(monkeypatch):
    monkeypatch.setattr(connectors, 'MAX_PDF_INPUT_BYTES', 4)
    with pytest.raises(UnsupportedFileError, match='文件过大'):
        extract_file_content('large.pdf', _pdf('text'))

    monkeypatch.setattr(connectors, 'MAX_PDF_INPUT_BYTES', 25 * 1024 * 1024)
    monkeypatch.setattr(connectors, 'MAX_PDF_PAGES', 1)
    with pytest.raises(UnsupportedFileError, match='页数过多'):
        extract_file_content('many.pdf', _pdf('one', 'two'))

    monkeypatch.setattr(connectors, 'MAX_PDF_PAGES', 500)
    monkeypatch.setattr(connectors, 'MAX_PDF_PAGE_TEXT_CHARS', 4)
    with pytest.raises(UnsupportedFileError, match='单页最多支持'):
        extract_file_content('long-page.pdf', _pdf('too long'))

    monkeypatch.setattr(connectors, 'MAX_PDF_PAGE_TEXT_CHARS', 200_000)
    monkeypatch.setattr(connectors, 'MAX_PDF_TEXT_CHARS', 7)
    with pytest.raises(UnsupportedFileError, match='提取文本过长'):
        extract_file_content('long-total.pdf', _pdf('four', 'five'))


@pytest.mark.asyncio
async def test_local_connector_does_not_follow_symlinks_and_tracks_parse_failures(tmp_path):
    source_root = tmp_path / 'source'
    source_root.mkdir()
    (source_root / 'good.md').write_text('safe', encoding='utf-8')
    (source_root / 'broken.docx').write_bytes(b'not-a-docx')
    outside = tmp_path / 'outside.md'
    outside.write_text('must not be ingested', encoding='utf-8')
    try:
        (source_root / 'leak.md').symlink_to(outside)
    except OSError:
        pytest.skip('filesystem does not support symlinks')

    connector = LocalConnector(source_root)
    documents = await connector.fetch()

    assert [document.external_id for document in documents] == ['good.md']
    assert connector.seen_external_ids == {'broken.docx', 'good.md'}
    assert connector.errors and 'broken.docx' in connector.errors[0]
    assert all(document.content != 'must not be ingested' for document in documents)


@pytest.mark.asyncio
async def test_local_connector_rejects_files_written_past_upload_limit(tmp_path):
    source_root = tmp_path / 'source'
    source_root.mkdir()
    (source_root / 'oversized.md').write_bytes(b'12345')

    connector = LocalConnector(source_root, max_file_bytes=4)
    assert await connector.fetch() == []
    assert connector.seen_external_ids == {'oversized.md'}
    assert connector.errors == ['oversized.md: 文件大小超过限制（最多 4 字节）']


def test_source_models_use_neo4j_default_and_reject_path_filenames():
    assert SourceCreateRequest(kind='local', name='local').group_id == 'neo4j'
    with pytest.raises(ValidationError, match='不含路径'):
        FileUploadRequest(filename='../secret.md', content_base64='YQ==')
