from types import SimpleNamespace

from rl_paper_rag.download import download_papers


def _result(short_id, title="A paper"):
    return SimpleNamespace(
        get_short_id=lambda: short_id, pdf_url=f"https://arxiv.org/pdf/{short_id}", title=title,
        authors=[SimpleNamespace(name="Ada"), SimpleNamespace(name="Bo")],
        published="2026-08-01", summary="An abstract.")


def test_downloads_each_pdf_and_returns_metadata(tmp_path):
    fetched = []
    papers = download_papers(
        str(tmp_path), results=[_result("2608.00001v1"), _result("2608.00002v1", "Second")],
        fetch=lambda url, path: (fetched.append((url, path)), open(path, "wb").close()),
        sleep=lambda s: None)
    assert [p["id"] for p in papers] == ["2608.00001v1", "2608.00002v1"]
    assert papers[0]["authors"] == ["Ada", "Bo"] and papers[1]["title"] == "Second"
    assert fetched[0][0] == "https://arxiv.org/pdf/2608.00001v1"
    assert (tmp_path / "2608.00001v1.pdf").exists()


def test_existing_files_are_not_downloaded_again(tmp_path):
    (tmp_path / "2608.00001v1.pdf").write_bytes(b"already here")
    calls = []
    download_papers(str(tmp_path), results=[_result("2608.00001v1")],
                    fetch=lambda url, path: calls.append(url), sleep=lambda s: None)
    assert calls == []
    assert (tmp_path / "2608.00001v1.pdf").read_bytes() == b"already here"


def test_slash_in_old_style_ids_is_made_safe(tmp_path):
    papers = download_papers(str(tmp_path), results=[_result("cs/0112017v1")],
                             fetch=lambda url, path: open(path, "wb").close(), sleep=lambda s: None)
    assert papers[0]["id"] == "cs_0112017v1"
    assert (tmp_path / "cs_0112017v1.pdf").exists()


def test_it_waits_between_downloads_only_when_it_downloaded(tmp_path):
    waits = []
    download_papers(str(tmp_path), results=[_result("a"), _result("b")],
                    fetch=lambda url, path: open(path, "wb").close(), sleep=waits.append)
    assert waits == [1, 1]


def test_creates_the_output_folder(tmp_path):
    target = tmp_path / "nested" / "pdfs"
    download_papers(str(target), results=[])
    assert target.is_dir()
