"""PDFs -> text -> chunks -> embeddings -> FAISS index."""
import argparse
import os
from collections import Counter

from rl_paper_rag.corpus import build_papers, chunk_papers, load_manifest
from rl_paper_rag.index import build_index, get_embeddings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--pdf-dir", default=os.path.join(ROOT, "arxiv_pdfs"))
ap.add_argument("--manifest", default=os.path.join(ROOT, "data", "corpus_manifest.json"))
ap.add_argument("--out", default=os.path.join(ROOT, "faiss_rl_papers_index"))
args = ap.parse_args()

papers = build_papers(load_manifest(args.manifest), args.pdf_dir)
usable = sum(len(p["full_text"]) > 500 for p in papers)
print(f"papers with usable text: {usable}/{len(papers)}")
chunks = chunk_papers(papers)
counts = Counter(c["paper_id"] for c in chunks)
print(f"{len(chunks)} chunks from {len(counts)} papers "
      f"(min {min(counts.values())}, max {max(counts.values())} per paper)")
store = build_index(chunks, get_embeddings())
store.save_local(args.out)
print(f"saved index to {args.out}")
