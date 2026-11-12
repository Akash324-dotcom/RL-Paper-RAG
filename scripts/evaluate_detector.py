"""Run the NLI hallucination detector on data/detector_eval.jsonl and report precision and recall.
Downloads facebook/bart-large-mnli (about 1.6 GB) the first time."""
import argparse
import csv
import os

from rl_paper_rag.evaluate import format_summary, load_eval_set, predict, summarize
from rl_paper_rag.faithfulness import BartMNLIScorer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--data", default=os.path.join(ROOT, "data", "detector_eval.jsonl"))
ap.add_argument("--out", default=os.path.join(ROOT, "results", "detector_eval_predictions.csv"))
args = ap.parse_args()

preds = predict(load_eval_set(args.data), BartMNLIScorer())
print(format_summary(summarize(preds)))

os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["id", "kind", "label", "nli_label", "confidence", "flagged", "hypothesis"],
                       extrasaction="ignore")
    w.writeheader()
    w.writerows(preds)
print(f"per-example predictions: {args.out}")
