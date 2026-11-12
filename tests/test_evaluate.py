import json
import os

import pytest

from rl_paper_rag.evaluate import format_summary, load_eval_set, predict, summarize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_PATH = os.path.join(ROOT, "data", "detector_eval.jsonl")


def P(label, flagged, kind="contradicted"):
    return {"label": label, "flagged": flagged, "kind": kind, "nli_label": "neutral" if flagged else "entailment"}


def test_summary_counts_and_metrics_on_a_known_confusion_matrix():
    preds = ([P("unsupported", True)] * 6 + [P("unsupported", False)] * 2 +   # tp=6 fn=2
             [P("supported", True)] * 1 + [P("supported", False)] * 9)         # fp=1 tn=9
    s = summarize(preds)
    assert (s["tp"], s["fn"], s["fp"], s["tn"]) == (6, 2, 1, 9)
    assert s["precision"] == pytest.approx(6 / 7)
    assert s["recall"] == pytest.approx(6 / 8)
    assert s["f1"] == pytest.approx(2 * (6 / 7) * (6 / 8) / ((6 / 7) + (6 / 8)))
    assert s["accuracy"] == pytest.approx(15 / 18)


def test_metrics_are_none_when_there_is_nothing_to_measure():
    s = summarize([P("supported", False)] * 3)       # no unsupported rows, nothing flagged
    assert s["precision"] is None and s["recall"] is None and s["f1"] is None
    assert s["accuracy"] == 1.0


def test_a_detector_that_flags_nothing_has_zero_recall():
    s = summarize([P("unsupported", False)] * 4 + [P("supported", False)] * 4)
    assert s["recall"] == 0.0 and s["precision"] is None and s["f1"] is None


def test_recall_is_broken_down_by_kind_of_unsupported_sentence():
    preds = [P("unsupported", True, "contradicted"), P("unsupported", False, "contradicted"),
             P("unsupported", True, "unstated"), P("supported", False, "entailed")]
    assert summarize(preds)["recall_by_kind"] == {"contradicted": (1, 2), "unstated": (1, 1)}


def test_predict_flags_everything_that_is_not_entailment():
    rows = [{"id": str(i), "premise": "p", "hypothesis": "h", "label": "supported"} for i in range(3)]
    labels = iter([("entailment", 0.9), ("neutral", 0.8), ("contradiction", 0.7)])
    out = predict(rows, lambda p, h: next(labels))
    assert [r["flagged"] for r in out] == [False, True, True]
    assert out[1]["nli_label"] == "neutral" and out[1]["confidence"] == 0.8


def test_format_summary_is_readable_and_handles_missing_metrics():
    text = format_summary(summarize([P("supported", False)] * 2))
    assert "precision n/a" in text and "accuracy 100.0%" in text


def test_load_eval_set_rejects_a_bad_label(tmp_path):
    f = tmp_path / "e.jsonl"
    f.write_text(json.dumps({"id": "1", "premise": "p", "hypothesis": "h", "label": "maybe"}) + "\n")
    with pytest.raises(ValueError, match="label"):
        load_eval_set(str(f))


def test_load_eval_set_rejects_missing_fields(tmp_path):
    f = tmp_path / "e.jsonl"
    f.write_text(json.dumps({"id": "1", "premise": "p"}) + "\n")
    with pytest.raises(ValueError, match="missing"):
        load_eval_set(str(f))


def test_load_eval_set_skips_blank_lines(tmp_path):
    f = tmp_path / "e.jsonl"
    f.write_text("\n" + json.dumps({"id": "1", "premise": "p", "hypothesis": "h", "label": "supported"}) + "\n\n")
    assert len(load_eval_set(str(f))) == 1


# ---- checks on the committed labeled set itself ----

@pytest.fixture(scope="module")
def committed():
    return load_eval_set(EVAL_PATH)


def test_committed_set_is_balanced_and_has_unique_ids(committed):
    labels = [r["label"] for r in committed]
    assert len(committed) == 34 and labels.count("supported") == labels.count("unsupported") == 17
    assert len({r["id"] for r in committed}) == len(committed)


def test_committed_set_has_no_repeated_claims_and_real_premises(committed):
    assert len({(r["premise"], r["hypothesis"]) for r in committed}) == len(committed)
    assert all(len(r["premise"]) > 100 and r["hypothesis"].endswith(".") for r in committed)


def test_every_premise_belongs_to_a_paper_in_the_corpus(committed):
    manifest = json.load(open(os.path.join(ROOT, "data", "corpus_manifest.json")))
    assert {r["paper_id"] for r in committed} <= {e["id"] for e in manifest}


def test_kinds_line_up_with_labels(committed):
    for r in committed:
        assert (r["kind"] == "entailed") == (r["label"] == "supported")
        assert r["kind"] in {"entailed", "contradicted", "unstated"}


def test_each_premise_has_both_supported_and_unsupported_claims_where_possible(committed):
    by_prem = {}
    for r in committed:
        by_prem.setdefault(r["premise"], set()).add(r["label"])
    assert all(v == {"supported", "unsupported"} for v in by_prem.values())
