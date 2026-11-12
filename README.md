# RL-Paper-RAG

[![tests](https://github.com/Akash324-dotcom/RL-Paper-RAG/actions/workflows/tests.yml/badge.svg)](https://github.com/Akash324-dotcom/RL-Paper-RAG/actions/workflows/tests.yml)

A retrieval-augmented generation (RAG) system that answers questions about recent reinforcement learning research, plus an automated **hallucination detector** that checks each sentence of the answer against the retrieved sources.

Runs fully on CPU with open-source models. No paid APIs.

## Why

RAG is supposed to ground LLM answers in real documents, but small models still make things up. This project builds the pipeline end to end, then measures where it fails and adds an NLI-based faithfulness check to catch those failures automatically.

## Pipeline

```
arXiv API ──► PDF text extraction ──► chunking ──► embeddings ──► FAISS index
                                                                     │
question ──► (optional query expansion) ──► top-k retrieval ◄────────┘
                                                 │
                                   prompt + context ──► TinyLlama ──► answer
                                                                        │
                                          NLI faithfulness check ◄──────┘
                                  (flags unsupported sentences)
```

| Stage | Details |
|---|---|
| Corpus | 50 recent `cs.LG` papers matching "reinforcement learning", pulled with the `arxiv` API. The exact papers are listed in `data/corpus_manifest.json`, so the corpus can be rebuilt (a fresh "most recent" query returns different papers every day) |
| Extraction | `pypdf`, whitespace cleanup (50/50 papers extracted successfully) |
| Chunking | LangChain `RecursiveCharacterTextSplitter`, 500 chars with 50 overlap. The notebook run produced **7,780 chunks**; the committed index holds **7,779** (see the note below) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | FAISS (saved in `faiss_rl_papers_index/`, 384-dimensional vectors) |
| Generator | `TinyLlama-1.1B-Chat-v1.0`, chosen so it runs on CPU |
| Chain | LangChain LCEL: retriever → prompt → LLM → parser, with source-paper attribution |
| Faithfulness | `facebook/bart-large-mnli`: context = premise, each answer sentence = hypothesis |

## Experiments and findings

- **Grounding prompt.** When the context didn't contain the answer, the model said so instead of guessing. The "say you don't know" instruction worked.
- **Repetition penalty (1.3).** Removed the looping and repeated phrases seen in the baseline in every test run.
- **Stricter "don't guess" prompt.** *Did not* stop hallucinations. Across runs, the model still invented who created PPO (e.g., attributing it to the wrong people and organizations). Prompting alone isn't enough.
- **Query expansion.** TinyLlama struggled to rephrase queries reliably. It added labels, drifted into answering, or returned nothing. It improved with few-shot examples and output filtering, but stayed unreliable at 1.1B parameters.
- **Retrieval depth (k = 1, 3, 6).** Showed no clear quality difference on this corpus. Sampling noise (temperature > 0) dominated.

### Hallucination detector results

| Answer sentence | NLI label | Verdict |
|---|---|---|
| "PPO … developed by OpenAI's Sam Altman and others." *(fabricated)* | Neutral (0.97) | ⚠️ flagged |
| "It works by limiting the rate at which policies change…" | Entailment (0.91) | ✅ supported |
| Clean answer, sentence 1 | Entailment (0.99) | ✅ supported |
| Clean answer, sentence 2 | Entailment (0.62) | ✅ supported |

The detector flagged the fabricated claim and passed the grounded sentences. It separates hallucinated content from supported content instead of flagging everything.

## Run it

```bash
pip install -r requirements.txt
```

**Ask a question** with the prebuilt index (downloads TinyLlama, about 2 GB, the first time):

```bash
python scripts/ask.py "What is Proximal Policy Optimization and how does it work?" --check
```

`--check` then runs the NLI detector over each sentence of the answer and marks the ones the retrieved context does not support.

**Rebuild everything from scratch:**

```bash
python scripts/download_papers.py     # the 50 PDFs listed in data/corpus_manifest.json
python scripts/build_index.py         # extract, chunk, embed, save faiss_rl_papers_index/
```

**Explore the experiments** in `RAG_Pipeline_Hallucinations_and_NLI.ipynb` (prompt comparison, repetition penalty, query expansion, retrieval depth, NLI detector). It is the original exploratory notebook; the code that matters has been moved into the `rl_paper_rag/` package so it can be tested.

Loading a saved index unpickles its document store, so only load an index you built yourself.

## Project layout

```
rl_paper_rag/
  corpus.py        PDF text extraction, cleaning, chunking
  download.py      fetch PDFs from arXiv (exact ids, or the newest N)
  index.py         build / save / load the FAISS index, search with paper attribution
  rag.py           prompts, context formatting, the retriever -> prompt -> LLM chain
  expansion.py     query expansion and the filter for the junk a 1.1B model produces
  faithfulness.py  sentence splitting and the NLI check (bart-large-mnli)
  evaluate.py      precision / recall of the detector on a labeled set
scripts/           download_papers, build_index, ask, evaluate_detector
data/              corpus_manifest.json (the 50 papers), detector_eval.jsonl (labeled set)
tests/             66 tests
```

## Tests

```bash
pip install -r requirements-dev.txt     # no torch, no model downloads
python -m pytest -q
```

They run on every push through GitHub Actions in about a second. To keep that fast and key-free, they replace the models with stand-ins: a hash-based fake embedding model for the FAISS tests, and a fake LLM and fake NLI scorer for the chain and detector logic. They check what the code around the models is responsible for:

- text extraction (including damaged and missing PDFs), whitespace cleaning, chunk size, overlap and full coverage of the text
- download logic with a fake arXiv: no repeat downloads, safe file names, polite delay
- the index: every chunk is stored, nearest-first ordering, per-paper attribution, save and load round trip, and a check that the committed index matches `data/corpus_manifest.json`
- the chain: retrieved chunks and the question land in the prompt with their paper titles, and the strict prompt differs from the baseline
- query expansion filtering, sentence splitting, and how NLI probabilities become a verdict
- the evaluation metrics, checked against a hand-computed confusion matrix

What they do **not** test is the quality of the models themselves. Whether TinyLlama hallucinates, or how well BART-MNLI catches it, is measured by the evaluation below.

## Evaluating the detector

`data/detector_eval.jsonl` holds 34 labeled claims (17 supported, 17 unsupported). Each claim is checked against a passage copied word for word from one of the 50 papers:

| Kind of claim | Count | Example idea |
|---|---|---|
| `entailed` (supported) | 17 | restates a fact the passage states |
| `contradicted` (unsupported) | 12 | changes a number, a direction or a model size |
| `unstated` (unsupported) | 5 | adds a plausible detail the passage never mentions |

The detector flags a sentence when the NLI label is anything other than entailment. To score it:

```bash
python scripts/evaluate_detector.py     # downloads bart-large-mnli (~1.6 GB) the first time
```

It prints precision, recall, F1 and accuracy for catching unsupported claims, plus recall split by kind, and writes every prediction to `results/detector_eval_predictions.csv`.

**No results are reported here yet.** The set was written by hand from the passages, so read the labels before you quote a number from it. It is small, so a single claim moves recall by about 6 points.

## Notes on the numbers

- **Chunk counts are not exactly reproducible.** The notebook run printed 7,780 chunks, the committed index holds 7,779, and re-extracting the same PDFs in a different environment gave 7,768 (44 of the 50 papers matched the index exactly, six differed by 1 to 14 chunks). The splitter is deterministic, so the likely cause is small differences in the PDF text extraction between `pypdf` versions; this was not isolated.
- The results table above comes from the notebook run. The prompt and retrieval comparisons used sampling (temperature above 0), so repeating them gives different wording.
- The NLI check truncates the context only, never the sentence being judged (`truncation="only_first"` in the package; the notebook used the default, which can cut either).

## Limitations and next steps

- The 1.1B generator limits answer quality. Swapping in a larger model (e.g., Llama 3 8B, Mistral 7B) is the obvious upgrade.
- MNLI truncates long contexts at 1,024 tokens. Checking each sentence against each retrieved chunk would be more precise.
- The detector currently flags errors but doesn't fix them. Next step: regenerate or drop flagged sentences automatically.
- Run `scripts/evaluate_detector.py`, check the labels, and report precision and recall for the detector (the labeled set and the script exist; the numbers do not yet). The set is small and hand-written, so the next step beyond it is real model-generated answers labeled by hand.

## Stack

Python · LangChain · FAISS · Hugging Face Transformers · sentence-transformers · PyTorch · pypdf · arXiv API
