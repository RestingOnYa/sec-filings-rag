"""PDF -> list of pages. Page numbers are 0-indexed everywhere in this project."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Page:
    doc_name: str
    page: int  # 0-indexed
    text: str


def parse_pdf(path: str | Path, backend: str = "pymupdf") -> list[Page]:
    path = Path(path)
    if backend == "pymupdf":
        return _parse_pymupdf(path)
    if backend == "docling":
        return _parse_docling(path)
    raise ValueError(f"unknown parse backend: {backend}")


def _parse_pymupdf(path: Path) -> list[Page]:
    import fitz  # PyMuPDF

    with fitz.open(path) as doc:
        return [Page(path.stem, i, p.get_text("text")) for i, p in enumerate(doc)]


def _parse_docling(path: Path) -> list[Page]:
    """Layout-aware parsing that renders tables as markdown. Slower; try it in week 3."""
    from docling.document_converter import DocumentConverter

    result = DocumentConverter().convert(str(path))
    doc = result.document
    pages = []
    for page_no in sorted(doc.pages):  # docling pages are 1-indexed
        text = doc.export_to_markdown(page_no=page_no)
        pages.append(Page(path.stem, page_no - 1, text))
    return pages
