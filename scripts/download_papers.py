"""Download the PDFs. By default fetches the exact 50 papers in data/corpus_manifest.json."""
import argparse
import json
import os

from rl_paper_rag.corpus import load_manifest
from rl_paper_rag.download import download_papers

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--pdf-dir", default=os.path.join(ROOT, "arxiv_pdfs"))
ap.add_argument("--manifest", default=os.path.join(ROOT, "data", "corpus_manifest.json"))
ap.add_argument("--latest", type=int, metavar="N",
                help="instead, fetch the N newest cs.LG reinforcement-learning papers (the set changes daily)")
args = ap.parse_args()

if args.latest:
    papers = download_papers(args.pdf_dir, max_results=args.latest)
else:
    papers = download_papers(args.pdf_dir, ids=[e["id"] for e in load_manifest(args.manifest)])
print(f"{len(papers)} papers in {args.pdf_dir}")
