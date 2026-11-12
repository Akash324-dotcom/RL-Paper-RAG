"""Score the hallucination detector against a small labeled set.

The detector's decision is: flag a sentence when the NLI label is not 'entailment'.
A flagged sentence counts as a positive prediction of "unsupported".
"""
import json
from collections import Counter, defaultdict
from typing import Callable, Iterable

SUPPORTED, UNSUPPORTED = "supported", "unsupported"


def load_eval_set(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            missing = {"id", "premise", "hypothesis", "label"} - row.keys()
            if missing:
                raise ValueError(f"{path}:{n} is missing {sorted(missing)}")
            if row["label"] not in (SUPPORTED, UNSUPPORTED):
                raise ValueError(f"{path}:{n} has label {row['label']!r}")
            rows.append(row)
    return rows


def predict(rows: Iterable[dict], scorer: Callable[[str, str], tuple[str, float]]) -> list[dict]:
    """Run the scorer on every row. `flagged` is the detector's prediction of 'unsupported'."""
    out = []
    for r in rows:
        label, conf = scorer(r["premise"], r["hypothesis"])
        out.append({**r, "nli_label": label, "confidence": conf, "flagged": label != "entailment"})
    return out


def _ratio(num: int, den: int):
    return num / den if den else None


def summarize(predictions: list[dict]) -> dict:
    """Precision, recall and F1 for catching unsupported sentences, plus accuracy and the
    confusion counts. A metric is None when its denominator is zero (nothing to measure)."""
    tp = sum(p["flagged"] and p["label"] == UNSUPPORTED for p in predictions)
    fp = sum(p["flagged"] and p["label"] == SUPPORTED for p in predictions)
    fn = sum(not p["flagged"] and p["label"] == UNSUPPORTED for p in predictions)
    tn = sum(not p["flagged"] and p["label"] == SUPPORTED for p in predictions)
    precision, recall = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    f1 = None if not precision or not recall else 2 * precision * recall / (precision + recall)

    by_kind = defaultdict(lambda: [0, 0])  # kind -> [flagged, total] among unsupported rows
    for p in predictions:
        if p["label"] == UNSUPPORTED:
            by_kind[p.get("kind", "unspecified")][1] += 1
            by_kind[p.get("kind", "unspecified")][0] += p["flagged"]
    return {
        "n": len(predictions),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "accuracy": _ratio(tp + tn, len(predictions)),
        "recall_by_kind": {k: (v[0], v[1]) for k, v in sorted(by_kind.items())},
        "nli_labels": dict(Counter(p["nli_label"] for p in predictions)),
    }


def format_summary(s: dict) -> str:
    def pct(x):
        return "n/a" if x is None else f"{x:.1%}"

    lines = [
        f"examples: {s['n']}   (flagged correctly {s['tp']}, missed {s['fn']}, "
        f"false alarms {s['fp']}, passed correctly {s['tn']})",
        f"precision {pct(s['precision'])}   recall {pct(s['recall'])}   "
        f"F1 {pct(s['f1'])}   accuracy {pct(s['accuracy'])}",
        "recall by kind of unsupported sentence:",
    ]
    for kind, (hit, total) in s["recall_by_kind"].items():
        lines.append(f"  {kind:<13} {hit}/{total}")
    return "\n".join(lines)
