"""Build, save, load and search the FAISS index. The embedding model is passed in, so tests
can use a tiny fake one instead of downloading MiniLM."""
from collections import Counter
from typing import Iterable

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_INDEX_DIR = "faiss_rl_papers_index"


def get_embeddings(model_name: str = EMBED_MODEL_NAME):
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=model_name)


def build_index(chunks: Iterable[dict], embeddings):
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document

    documents = [
        Document(page_content=c["text"],
                 metadata={"paper_id": c["paper_id"], "paper_title": c["paper_title"]})
        for c in chunks
    ]
    if not documents:
        raise ValueError("no chunks to index")
    return FAISS.from_documents(documents, embeddings)


def load_index(index_dir: str, embeddings):
    """Loading unpickles the document store, so only load an index you built yourself."""
    from langchain_community.vectorstores import FAISS

    return FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)


def chunks_per_paper(vector_store) -> Counter:
    return Counter(d.metadata["paper_id"] for d in vector_store.docstore._dict.values())


def search_with_attribution(vector_store, query: str, k: int = 5, snippet_chars: int = 200) -> list[dict]:
    """Top-k chunks with their FAISS distance (lower = closer) and the paper each came from."""
    counts = chunks_per_paper(vector_store)
    out = []
    for rank, (doc, score) in enumerate(vector_store.similarity_search_with_score(query, k=k), 1):
        pid = doc.metadata["paper_id"]
        out.append({
            "rank": rank,
            "score": float(score),
            "paper_id": pid,
            "paper_title": doc.metadata["paper_title"],
            "paper_chunks": counts[pid],
            "snippet": doc.page_content[:snippet_chars],
        })
    return out
