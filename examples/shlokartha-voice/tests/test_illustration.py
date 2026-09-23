from nodes.illustration import illustration_prompt


def test_prompt_is_landscape_led():
    prompt = illustration_prompt("Arjuna sees his kinsmen on the battlefield.")
    assert "landscape" in prompt.lower()


def test_prompt_strips_deity_terms():
    prompt = illustration_prompt("Lord Krishna speaks to the god Arjuna.")
    lowered = prompt.lower()
    assert "krishna" not in lowered and "god" not in lowered


def test_prompt_is_capped_in_length():
    prompt = illustration_prompt("word " * 500)
    assert len(prompt) <= 400


def test_empty_meaning_still_yields_a_usable_prompt():
    assert illustration_prompt("").strip() != ""
