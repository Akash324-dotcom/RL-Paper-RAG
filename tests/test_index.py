import pytest

from rl_paper_rag.index import build_index, chunks_per_paper, load_index, search_with_attribution

CHUNKS = [
    {"text": "PPO clips the policy update to stay close to the old policy.", "paper_id": "P1", "paper_title": "About PPO"},
    {"text": "Reward shaping adds extra signal to guide exploration.", "paper_id": "P1", "paper_title": "About PPO"},
    {"text": "Offline RL learns from a fixed dataset of transitions.", "paper_id": "P2", "paper_title": "About offline RL"},
]


@pytest.fixture
def store(fake_embeddings):
    return build_index(CHUNKS, fake_embeddings)


def test_index_holds_every_chunk(store):
    assert store.index.ntotal == 3


def test_exact_text_query_finds_its_own_chunk_first(store):
    hits = search_with_attribution(store, CHUNKS[2]["text"], k=3)
    assert hits[0]["paper_id"] == "P2"
    assert hits[0]["paper_title"] == "About offline RL"
    assert hits[0]["score"] == pytest.approx(0.0, abs=1e-4)
    assert [h["rank"] for h in hits] == [1, 2, 3]


def test_results_are_ordered_closest_first(store):
    scores = [h["score"] for h in search_with_attribution(store, CHUNKS[0]["text"], k=3)]
    assert scores == sorted(scores)


def test_attribution_reports_how_many_chunks_each_paper_contributed(store):
    hits = search_with_attribution(store, CHUNKS[0]["text"], k=3)
    by_paper = {h["paper_id"]: h["paper_chunks"] for h in hits}
    assert by_paper == {"P1": 2, "P2": 1}
    assert dict(chunks_per_paper(store)) == {"P1": 2, "P2": 1}


def test_snippet_is_truncated(store):
    hit = search_with_attribution(store, CHUNKS[0]["text"], k=1, snippet_chars=10)[0]
    assert hit["snippet"] == CHUNKS[0]["text"][:10]


def test_k_limits_the_number_of_results(store):
    assert len(search_with_attribution(store, "anything", k=2)) == 2


def test_save_and_load_round_trip(store, fake_embeddings, tmp_path):
    store.save_local(str(tmp_path / "idx"))
    loaded = load_index(str(tmp_path / "idx"), fake_embeddings)
    assert loaded.index.ntotal == 3
    assert search_with_attribution(loaded, CHUNKS[1]["text"], k=1)[0]["paper_id"] == "P1"


def test_empty_input_is_rejected(fake_embeddings):
    with pytest.raises(ValueError):
        build_index([], fake_embeddings)


def test_the_committed_index_is_consistent_with_the_manifest(fake_embeddings):
    import json, os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    committed = load_index(os.path.join(root, "faiss_rl_papers_index"), fake_embeddings) \
        if os.path.exists(os.path.join(root, "faiss_rl_papers_index", "index.faiss")) else None
    if committed is None:
        pytest.skip("committed index not present")
    manifest = json.load(open(os.path.join(root, "data", "corpus_manifest.json")))
    expected = {e["id"]: e["chunks_in_committed_index"] for e in manifest}
    assert dict(chunks_per_paper(committed)) == expected
    assert committed.index.ntotal == 7779
