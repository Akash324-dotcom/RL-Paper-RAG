# RL-Paper-RAG

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
| Corpus | 50 most recent `cs.LG` papers matching "reinforcement learning", pulled with the `arxiv` API |
| Extraction | `pypdf`, whitespace cleanup (50/50 papers extracted successfully) |
| Chunking | LangChain `RecursiveCharacterTextSplitter`, 500 chars with 50 overlap, **7,780 chunks** |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | FAISS (saved in `faiss_rl_papers_index/`) |
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
pip install arxiv pypdf langchain langchain-community langchain-huggingface \
            sentence-transformers faiss-cpu transformers torch accelerate
jupyter notebook "RAG_Pipeline_Hallucinations_and_NLI (1).ipynb"
```

The prebuilt FAISS index is included, so you can skip re-embedding by loading it with `FAISS.load_local("faiss_rl_papers_index", embeddings, allow_dangerous_deserialization=True)`.

## Limitations and next steps

- The 1.1B generator limits answer quality. Swapping in a larger model (e.g., Llama 3 8B, Mistral 7B) is the obvious upgrade.
- MNLI truncates long contexts at 1,024 tokens. Checking each sentence against each retrieved chunk would be more precise.
- The detector currently flags errors but doesn't fix them. Next step: regenerate or drop flagged sentences automatically.
- Add a small labeled evaluation set to report precision and recall for the detector.

## Stack

Python · LangChain · FAISS · Hugging Face Transformers · sentence-transformers · PyTorch · pypdf · arXiv API
