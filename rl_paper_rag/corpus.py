"""Turn PDFs into text, and text into overlapping chunks. No models needed."""
import json
import logging
import os
import re
from typing import Iterable

from pypdf import PdfReader

log = logging.getLogger(__name__)

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
MIN_CHUNKABLE_CHARS = 200  # papers with less text than this are skipped
USABLE_TEXT_CHARS = 500    # "extracted successfully" means more than this


def safe_id(short_id: str) -> str:
    """Old-style arXiv ids contain a slash (cs/0112017); make them safe as file names."""
    return short_id.replace("/", "_")


def clean_text(text: str) -> str:
    """Collapse every run of whitespace (newlines, tabs, double spaces) into one space."""
    return re.sub(r"\s+", " ", text).strip()


def extract_text_from_pdf(pdf_path: str) -> str:
    """All text of a PDF, page by page. Returns "" (and logs) if the file cannot be read."""
    try:
        reader = PdfReader(pdf_path)
        parts = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                parts.append(page_text)
        return "\n".join(parts) + ("\n" if parts else "")
    except Exception as e:  # pypdf raises many different error types on damaged files
        log.warning("Failed to read %s: %s", pdf_path, e)
        return ""


def load_manifest(path: str) -> list[dict]:
    """The fixed list of papers in the corpus: [{"id", "title", ...}]."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_papers(manifest: Iterable[dict], pdf_dir: str) -> list[dict]:
    """Attach cleaned full text to every manifest entry whose PDF is on disk."""
    papers = []
    for entry in manifest:
        pdf_path = os.path.join(pdf_dir, f"{entry['id']}.pdf")
        if not os.path.exists(pdf_path):
            log.warning("Missing PDF for %s (run scripts/download_papers.py)", entry["id"])
            continue
        papers.append({**entry, "pdf_path": pdf_path,
                       "full_text": clean_text(extract_text_from_pdf(pdf_path))})
    return papers


def chunk_papers(papers: Iterable[dict], chunk_size: int = CHUNK_SIZE,
                 chunk_overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """Split each paper into overlapping character chunks, keeping which paper each came from."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len)
    chunks = []
    for paper in papers:
        if len(paper["full_text"]) < MIN_CHUNKABLE_CHARS:
            continue
        for text in splitter.split_text(paper["full_text"]):
            chunks.append({"text": text, "paper_id": paper["id"], "paper_title": paper["title"]})
    return chunks
