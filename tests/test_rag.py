import pytest
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from rl_paper_rag.index import build_index
from rl_paper_rag.rag import BASELINE_PROMPT, STRICT_PROMPT, build_chain, format_docs

CHUNKS = [
    {"text": "PPO clips the policy update to stay close to the old policy.", "paper_id": "P1", "paper_title": "About PPO"},
    {"text": "Offline RL learns from a fixed dataset of transitions.", "paper_id": "P2", "paper_title": "About offline RL"},
]


def test_format_docs_labels_each_chunk_with_its_paper():
    docs = [Document(page_content="alpha", metadata={"paper_title": "T1"}),
            Document(page_content="beta", metadata={"paper_title": "T2"})]
    assert format_docs(docs) == "[From: T1]\nalpha\n\n[From: T2]\nbeta"


@pytest.mark.parametrize("prompt", [BASELINE_PROMPT, STRICT_PROMPT])
def test_prompts_have_the_two_slots_and_the_chat_markers(prompt):
    assert "{context}" in prompt and "{question}" in prompt
    assert prompt.count("<|assistant|>") == 1 and prompt.rstrip().endswith("<|assistant|>")


def test_strict_prompt_forbids_guessing_names():
    assert "Do NOT guess" in STRICT_PROMPT and "Do NOT guess" not in BASELINE_PROMPT


def test_chain_puts_retrieved_context_and_question_into_the_prompt(fake_embeddings):
    seen = {}

    def fake_llm(prompt_value):
        seen["prompt"] = prompt_value.to_string()
        return "  the answer  "

    store = build_index(CHUNKS, fake_embeddings)
    chain = build_chain(store.as_retriever(search_kwargs={"k": 1}), RunnableLambda(fake_llm))
    question = CHUNKS[1]["text"]  # exact text, so the offline-RL chunk is retrieved
    out = chain.invoke(question)

    assert out == "  the answer  "
    assert "[From: About offline RL]" in seen["prompt"]
    assert CHUNKS[1]["text"] in seen["prompt"]
    assert CHUNKS[0]["text"] not in seen["prompt"]   # k=1: only the closest chunk is shown
    assert f"Question: {question}" in seen["prompt"]
    assert seen["prompt"].startswith("<|system|>")


def test_chain_can_use_the_baseline_prompt(fake_embeddings):
    seen = {}
    store = build_index(CHUNKS, fake_embeddings)
    chain = build_chain(store.as_retriever(search_kwargs={"k": 1}),
                        RunnableLambda(lambda p: seen.setdefault("p", p.to_string())), BASELINE_PROMPT)
    chain.invoke("q")
    assert "rather than guessing" in seen["p"] and "Do NOT guess" not in seen["p"]
