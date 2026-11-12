"""Sentence-level faithfulness check with an NLI model: the retrieved context is the premise and
each answer sentence is the hypothesis. Anything not 'entailment' is flagged."""
import re
from dataclasses import dataclass
from typing import Callable, Sequence

NLI_MODEL_ID = "facebook/bart-large-mnli"
MIN_SENTENCE_CHARS = 16  # shorter fragments are skipped (">15" in the original notebook)


@dataclass(frozen=True)
class SentenceVerdict:
    sentence: str
    label: str          # "entailment", "neutral" or "contradiction"
    confidence: float

    @property
    def supported(self) -> bool:
        return self.label == "entailment"


def split_into_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in parts if len(s) >= MIN_SENTENCE_CHARS]


def label_from_probs(probs: Sequence[float], id2label: dict) -> tuple[str, float]:
    """The most likely label and its probability."""
    best = max(range(len(probs)), key=lambda i: probs[i])
    return id2label[best].lower(), float(probs[best])


class BartMNLIScorer:
    """Callable (context, sentence) -> (label, confidence). Loads the model on first use."""

    def __init__(self, model_id: str = NLI_MODEL_ID):
        self.model_id = model_id
        self._tok = self._model = None

    def _load(self):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
        self._model.eval()

    def __call__(self, context: str, sentence: str) -> tuple[str, float]:
        import torch

        if self._model is None:
            self._load()
        # truncation="only_first" cuts the context, never the sentence being judged
        inputs = self._tok(context, sentence, return_tensors="pt",
                           truncation="only_first", max_length=1024)
        with torch.no_grad():
            probs = torch.softmax(self._model(**inputs).logits, dim=1)[0].tolist()
        return label_from_probs(probs, self._model.config.id2label)


def check_faithfulness(context: str, answer: str,
                       scorer: Callable[[str, str], tuple[str, float]]) -> list[SentenceVerdict]:
    return [SentenceVerdict(s, *scorer(context, s)) for s in split_into_sentences(answer)]


def unsupported(verdicts: Sequence[SentenceVerdict]) -> list[SentenceVerdict]:
    return [v for v in verdicts if not v.supported]
