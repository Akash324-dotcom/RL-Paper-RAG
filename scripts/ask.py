"""Ask a question about the corpus. Optionally check each answer sentence against the context."""
import argparse
import os

from rl_paper_rag.faithfulness import BartMNLIScorer, check_faithfulness
from rl_paper_rag.index import get_embeddings, load_index
from rl_paper_rag.rag import BASELINE_PROMPT, STRICT_PROMPT, build_chain, format_docs, load_tinyllama

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("question")
ap.add_argument("-k", type=int, default=3, help="chunks to retrieve")
ap.add_argument("--index", default=os.path.join(ROOT, "faiss_rl_papers_index"))
ap.add_argument("--baseline-prompt", action="store_true", help="use the first, less strict prompt")
ap.add_argument("--check", action="store_true", help="flag answer sentences the context does not support")
args = ap.parse_args()

store = load_index(args.index, get_embeddings())
retriever = store.as_retriever(search_kwargs={"k": args.k})
chain = build_chain(retriever, load_tinyllama(),
                    BASELINE_PROMPT if args.baseline_prompt else STRICT_PROMPT)
answer = chain.invoke(args.question)
print("\nAnswer:", answer)

if args.check:
    context = format_docs(retriever.invoke(args.question))
    print("\nFaithfulness check (context = premise, answer sentence = hypothesis):")
    for v in check_faithfulness(context, answer, BartMNLIScorer()):
        mark = "supported " if v.supported else "FLAGGED   "
        print(f"  {mark}[{v.label} {v.confidence:.2f}] {v.sentence}")
