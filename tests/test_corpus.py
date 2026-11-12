import json
import os

from rl_paper_rag.corpus import (
    build_papers, chunk_papers, clean_text, extract_text_from_pdf, load_manifest, safe_id,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_safe_id_removes_slashes():
    assert safe_id("cs/0112017v1") == "cs_0112017v1"
    assert safe_id("2608.03391v1") == "2608.03391v1"


def test_clean_text_collapses_all_whitespace():
    assert clean_text("  a \n\n b\t\tc  ") == "a b c"


def test_extract_text_from_a_real_pdf(make_pdf):
    path = make_pdf("p.pdf", "Policy gradient methods learn directly")
    assert "Policy gradient methods learn directly" in extract_text_from_pdf(path)


def test_extract_text_returns_empty_for_a_damaged_file(tmp_path):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"this is not a pdf")
    assert extract_text_from_pdf(str(bad)) == ""


def test_extract_text_returns_empty_for_a_missing_file(tmp_path):
    assert extract_text_from_pdf(str(tmp_path / "nope.pdf")) == ""


def test_build_papers_skips_missing_pdfs_and_cleans_text(tmp_path, make_pdf):
    make_pdf("A1.pdf", "alpha   beta")
    manifest = [{"id": "A1", "title": "Paper A"}, {"id": "B2", "title": "Missing"}]
    papers = build_papers(manifest, str(tmp_path))
    assert [p["id"] for p in papers] == ["A1"]
    assert papers[0]["full_text"] == "alpha beta"


def _paper(pid, text):
    return {"id": pid, "title": f"Title {pid}", "full_text": text}


def test_chunks_respect_size_and_keep_their_paper():
    text = " ".join(f"word{i}" for i in range(600))
    chunks = chunk_papers([_paper("P1", text)])
    assert len(chunks) > 1
    assert all(len(c["text"]) <= 500 for c in chunks)
    assert {c["paper_id"] for c in chunks} == {"P1"}
    assert {c["paper_title"] for c in chunks} == {"Title P1"}


def test_consecutive_chunks_overlap():
    text = " ".join(f"word{i}" for i in range(600))
    chunks = chunk_papers([_paper("P1", text)])
    first_words, second_words = chunks[0]["text"].split(), chunks[1]["text"].split()
    assert set(first_words[-3:]) & set(second_words[:12])


def test_chunking_all_text_is_covered():
    text = " ".join(f"word{i}" for i in range(600))
    chunks = chunk_papers([_paper("P1", text)])
    joined = " ".join(c["text"] for c in chunks)
    assert all(f"word{i}" in joined for i in range(600))


def test_papers_with_almost_no_text_are_skipped():
    assert chunk_papers([_paper("P1", "too short")]) == []


def test_chunking_is_deterministic():
    text = " ".join(f"w{i}" for i in range(800))
    assert chunk_papers([_paper("P", text)]) == chunk_papers([_paper("P", text)])


def test_committed_manifest_is_the_50_paper_corpus():
    manifest = load_manifest(os.path.join(ROOT, "data", "corpus_manifest.json"))
    ids = [e["id"] for e in manifest]
    assert len(ids) == 50 and len(set(ids)) == 50
    assert all(e["title"] for e in manifest)
    assert all(e["chunks_in_committed_index"] > 0 for e in manifest)
    assert sum(e["chunks_in_committed_index"] for e in manifest) == 7779


def test_load_manifest_reads_json(tmp_path):
    p = tmp_path / "m.json"
    p.write_text(json.dumps([{"id": "X", "title": "T"}]))
    assert load_manifest(str(p)) == [{"id": "X", "title": "T"}]
