"""Query expansion: ask the LLM for rephrasings, then filter out the junk a 1.1B model produces."""

EXPANSION_PROMPT = """<|system|>
You rewrite search queries as questions. Given a query, output exactly two rephrased QUESTIONS, one per line. Each line MUST end with a question mark. Do not explain anything. Do not answer the question.
</s>
<|user|>
Query: What causes model overfitting?
</s>
<|assistant|>
Why do machine learning models overfit?
What factors lead to overfitting in models?
</s>
<|user|>
Query: {query}
</s>
<|assistant|>
"""

BAD_STARTS = ("original", "query:", "two alternative", "alternative", "-", "•")
MIN_LINE_CHARS = 10


def parse_variations(generated: str, require_question: bool = True, limit: int = 2) -> list[str]:
    """Keep only lines that look like real rephrasings: long enough, not a label or a bullet
    echo, and (by default) ending in a question mark."""
    out = []
    for line in generated.split("\n"):
        line = line.strip("-• \n")
        if len(line) <= MIN_LINE_CHARS or line.lower().startswith(BAD_STARTS):
            continue
        if require_question and not line.endswith("?"):
            continue
        out.append(line)
    return out[:limit]


def expand_query(query: str, generate) -> list[str]:
    """The original query first, then up to two usable rephrasings. `generate(prompt) -> str`
    is the LLM call, passed in so this logic can be tested without a model."""
    return [query] + parse_variations(generate(EXPANSION_PROMPT.format(query=query)))
