"""Prompts, context formatting and the retrieval-augmented chain."""

# TinyLlama chat format. Kept as plain strings so they can be tested and compared.
BASELINE_PROMPT = """<|system|>
You are a helpful research assistant. Answer the question using ONLY the context provided below. If the answer is not contained in the context, say you don't know rather than guessing.
</s>
<|user|>
Context:
{context}

Question: {question}
</s>
<|assistant|>
"""

STRICT_PROMPT = """<|system|>
You are a careful research assistant. Answer the question using ONLY facts explicitly stated in the context below.
Do NOT guess names, dates, or attributions that are not directly stated in the context.
If a detail (such as who created something) is not explicitly stated in the context, say "the context does not specify this" instead of guessing.
Keep your answer concise and avoid repeating the same point multiple times.
</s>
<|user|>
Context:
{context}

Question: {question}
</s>
<|assistant|>
"""

TINYLLAMA_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def format_docs(docs) -> str:
    """Label each retrieved chunk with its paper so the model (and a reader) can see the source."""
    return "\n\n".join(f"[From: {d.metadata['paper_title']}]\n{d.page_content}" for d in docs)


def build_chain(retriever, llm, prompt_template: str = STRICT_PROMPT):
    """retriever -> prompt -> LLM -> text. `llm` is any LangChain runnable or callable."""
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnablePassthrough

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PromptTemplate.from_template(prompt_template)
        | llm
        | StrOutputParser()
    )


def load_tinyllama(repetition_penalty: float = 1.3, temperature: float = 0.3, max_new_tokens: int = 200):
    """The CPU-friendly generator used in the experiments. Downloads ~2 GB on first use."""
    import torch
    from langchain_huggingface import HuggingFacePipeline
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    tok = AutoTokenizer.from_pretrained(TINYLLAMA_ID)
    model = AutoModelForCausalLM.from_pretrained(TINYLLAMA_ID, torch_dtype=torch.float32)
    gen = pipeline("text-generation", model=model, tokenizer=tok, max_new_tokens=max_new_tokens,
                   temperature=temperature, do_sample=True,
                   repetition_penalty=repetition_penalty, return_full_text=False)
    return HuggingFacePipeline(pipeline=gen)
