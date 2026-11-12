from rl_paper_rag.expansion import EXPANSION_PROMPT, expand_query, parse_variations


def test_keeps_real_questions_and_drops_the_rest():
    text = ("Original query: what is PPO?\n"
            "What does proximal policy optimization do?\n"
            "- How does PPO limit policy updates?\n"
            "Alternative: something\n"
            "short?\n"
            "A statement that is not a question.\n")
    assert parse_variations(text) == ["What does proximal policy optimization do?",
                                      "How does PPO limit policy updates?"]


def test_at_most_two_variations():
    text = "\n".join(f"Question number {i} about RL?" for i in range(5))
    assert len(parse_variations(text)) == 2


def test_question_mark_can_be_made_optional():
    assert parse_variations("Explain how PPO works", require_question=False) == ["Explain how PPO works"]
    assert parse_variations("Explain how PPO works") == []


def test_empty_or_junk_output_gives_no_variations():
    assert parse_variations("") == []
    assert parse_variations("\n\n  \n") == []


def test_bullets_and_whitespace_are_stripped():
    assert parse_variations("•  Why does reward shaping help?  ") == ["Why does reward shaping help?"]


def test_expand_query_puts_the_original_first_and_uses_the_model_output():
    prompts = []

    def generate(prompt):
        prompts.append(prompt)
        return "How do actor-critic methods work?\nWhy use a critic network?"

    out = expand_query("actor critic", generate)
    assert out == ["actor critic", "How do actor-critic methods work?", "Why use a critic network?"]
    assert "Query: actor critic" in prompts[0]


def test_expand_query_still_returns_the_original_if_the_model_babbles():
    assert expand_query("original", lambda p: "I cannot help with that") == ["original"]


def test_the_prompt_has_a_slot_for_the_query():
    assert "{query}" in EXPANSION_PROMPT
