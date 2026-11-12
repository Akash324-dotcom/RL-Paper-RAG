import pytest

from rl_paper_rag.faithfulness import (
    SentenceVerdict, check_faithfulness, label_from_probs, split_into_sentences, unsupported,
)

ID2LABEL = {0: "contradiction", 1: "neutral", 2: "entailment"}


def test_split_into_sentences_on_end_punctuation():
    text = "PPO is a policy gradient method. It clips updates! Does it work well?"
    assert split_into_sentences(text) == [
        "PPO is a policy gradient method.", "It clips updates!", "Does it work well?"]


def test_short_fragments_are_dropped():
    assert split_into_sentences("Yes. This one is long enough to keep.") == ["This one is long enough to keep."]


def test_a_sentence_of_exactly_16_characters_is_kept():
    s = "a" * 15 + "."
    assert len(s) == 16 and split_into_sentences(s) == [s]
    assert split_into_sentences("a" * 14 + ".") == []


def test_empty_answer_has_no_sentences():
    assert split_into_sentences("   ") == []


@pytest.mark.parametrize("probs, expected", [
    ([0.1, 0.2, 0.7], ("entailment", 0.7)),
    ([0.05, 0.9, 0.05], ("neutral", 0.9)),
    ([0.6, 0.3, 0.1], ("contradiction", 0.6)),
])
def test_label_from_probs_picks_the_largest(probs, expected):
    label, conf = label_from_probs(probs, ID2LABEL)
    assert label == expected[0] and conf == pytest.approx(expected[1])


def test_label_names_are_lowercased():
    assert label_from_probs([0, 0, 1], {0: "A", 1: "B", 2: "ENTAILMENT"})[0] == "entailment"


def fake_scorer(context, sentence):
    return ("entailment", 0.9) if "clip" in sentence else ("neutral", 0.97)


def test_each_sentence_gets_its_own_verdict():
    answer = "PPO uses a clipped objective function. It was invented by Sam Altman and others."
    verdicts = check_faithfulness("context", answer, fake_scorer)
    assert [v.supported for v in verdicts] == [True, False]
    assert verdicts[1].label == "neutral" and verdicts[1].confidence == 0.97


def test_only_non_entailed_sentences_are_flagged():
    verdicts = [SentenceVerdict("a" * 20, "entailment", 0.9),
                SentenceVerdict("b" * 20, "contradiction", 0.8),
                SentenceVerdict("c" * 20, "neutral", 0.7)]
    assert [v.sentence[0] for v in unsupported(verdicts)] == ["b", "c"]


def test_the_scorer_gets_the_context_as_premise_and_the_sentence_as_hypothesis():
    calls = []
    check_faithfulness("THE CONTEXT", "This is one full sentence here.",
                       lambda c, s: calls.append((c, s)) or ("entailment", 1.0))
    assert calls == [("THE CONTEXT", "This is one full sentence here.")]
