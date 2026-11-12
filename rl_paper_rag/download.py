"""Download the corpus PDFs from arXiv."""
import os
import time
from typing import Callable, Optional
from urllib.request import urlretrieve

from .corpus import safe_id

DEFAULT_QUERY = 'cat:cs.LG AND ("reinforcement learning")'


def _client_and_search(ids=None, query=DEFAULT_QUERY, max_results=50):
    import arxiv

    client = arxiv.Client()
    if ids:
        return client, arxiv.Search(id_list=list(ids), max_results=len(ids))
    return client, arxiv.Search(query=query, max_results=max_results,
                                sort_by=arxiv.SortCriterion.SubmittedDate)


def download_papers(
    pdf_dir: str,
    ids: Optional[list[str]] = None,
    query: str = DEFAULT_QUERY,
    max_results: int = 50,
    results=None,
    fetch: Callable[[str, str], object] = urlretrieve,
    sleep: Callable[[float], None] = time.sleep,
) -> list[dict]:
    """Download PDFs into pdf_dir and return one metadata dict per paper.

    Pass `ids` to fetch an exact, reproducible set (see data/corpus_manifest.json).
    Without `ids`, the `max_results` newest papers matching `query` are fetched, which
    changes every day. Files already on disk are not downloaded again.
    `results`, `fetch` and `sleep` can be replaced in tests so nothing touches the network.
    """
    os.makedirs(pdf_dir, exist_ok=True)
    if results is None:
        client, search = _client_and_search(ids, query, max_results)
        results = client.results(search)

    papers = []
    for result in results:
        pid = safe_id(result.get_short_id())
        pdf_path = os.path.join(pdf_dir, f"{pid}.pdf")
        if not os.path.exists(pdf_path):
            fetch(result.pdf_url, pdf_path)
            sleep(1)  # be polite to arXiv
        papers.append({
            "id": pid,
            "title": result.title,
            "authors": [a.name for a in result.authors],
            "published": str(result.published),
            "pdf_path": pdf_path,
            "abstract": result.summary,
        })
    return papers
