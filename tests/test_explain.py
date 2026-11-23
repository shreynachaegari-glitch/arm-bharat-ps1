"""Experiment C (advice safety) cases for the Llama wrapper, using fake runners instead of the model."""
import sys
import textwrap

import pytest

from ps1.llm.explain import (
    ASSISTANT_HEADER,
    MAX_NEW_TOKENS,
    LlamaConfig,
    build_prompt,
    explain,
    seq_len_for,
)

TEMPLATE = "Reviewed template text."
PROMPT = build_prompt("tomato", "early blight", "dark rings on older leaves", "a photo cannot confirm it")
STATS = 'PyTorchObserver {"prompt_tokens":10,"generated_tokens":5}'


def fake_runner(tmp_path, body: str) -> tuple[str, ...]:
    script = tmp_path / "fake_llama.py"
    script.write_text(textwrap.dedent(body))
    return (sys.executable, str(script))


def cfg(command, timeout_s=5.0) -> LlamaConfig:
    return LlamaConfig(command=command, model_path="m.pte", tokenizer_path="t.model", timeout_s=timeout_s)


def echo_runner(tmp_path, generated: str):
    return fake_runner(tmp_path, f"""
        import sys
        prompt = [a for a in sys.argv if a.startswith("--prompt=")][0][len("--prompt="):]
        print(prompt + {generated!r} + "<|eot_id|>")
        print({STATS!r})
    """)


def test_safe_output_is_revealed(tmp_path):
    text = "The ringed brown spots look like early blight. A photo cannot confirm it."
    r = explain(cfg(echo_runner(tmp_path, text)), PROMPT, TEMPLATE)
    assert (r.source, r.checker_result, r.text) == ("llama", "pass", text)


@pytest.mark.parametrize("generated", ["Spray mancozeb now.", "This is definitely early blight.", "Use 2 grams."])
def test_unsafe_output_replaced_by_template(tmp_path, generated):
    r = explain(cfg(echo_runner(tmp_path, generated)), PROMPT, TEMPLATE)
    assert (r.source, r.checker_result, r.text) == ("template", "fail", TEMPLATE)
    assert generated in r.raw_output          # kept for the log, not shown


def test_timeout_falls_back(tmp_path):
    hang = fake_runner(tmp_path, "import time; time.sleep(10)")
    r = explain(cfg(hang, timeout_s=0.5), PROMPT, TEMPLATE)
    assert (r.source, r.checker_result, r.text) == ("template", "timeout", TEMPLATE)


def test_crash_falls_back(tmp_path):
    crash = fake_runner(tmp_path, "import sys; print('boom', file=sys.stderr); sys.exit(3)")
    r = explain(cfg(crash), PROMPT, TEMPLATE)
    assert (r.source, r.checker_result) == ("template", "error")


def test_missing_binary_falls_back():
    r = explain(cfg(("/nonexistent/llama_main",)), PROMPT, TEMPLATE)
    assert (r.source, r.checker_result) == ("template", "error")


def test_card_cause_reaches_the_prompt():
    p = build_prompt("tomato", "early blight", "rings", "a photo cannot confirm it", cause="a fungus")
    assert "cause=a fungus" in p


def test_prompt_ends_at_assistant_turn_and_bounds_output():
    assert PROMPT.rstrip().endswith(ASSISTANT_HEADER)
    assert seq_len_for(PROMPT) - MAX_NEW_TOKENS > 0


def test_invalid_utf8_from_runner_falls_back_instead_of_crashing(tmp_path):
    bad = fake_runner(tmp_path, f"""
        import sys
        sys.stdout.buffer.write(b"The spots look like early blight\\xe0\\x80.\\n")
        print({STATS!r})
    """)
    r = explain(cfg(bad), PROMPT, TEMPLATE)
    assert r.source in ("template", "llama")          # never an exception
    assert r.checker_result in ("fail", "pass", "error")


def test_timeout_kills_the_whole_process_group(tmp_path):
    marker = tmp_path / "child_alive"
    spawner = fake_runner(tmp_path, f"""
        import subprocess, sys, time
        subprocess.Popen([sys.executable, "-c", "import time, pathlib; time.sleep(1.5); pathlib.Path({str(marker)!r}).write_text('x')"])
        time.sleep(10)
    """)
    r = explain(cfg(spawner, timeout_s=0.5), PROMPT, TEMPLATE)
    assert r.checker_result == "timeout"
    import time
    time.sleep(2.0)
    assert not marker.exists()                         # grandchild was killed with the group


# --- 21 Nov: the card lists two possible causes; Llama must not pick one (seen on x86 with the real model) ---
LEAF_SPOT_CAUSES = (("septoria", "fung"), ("bacteri",))


def _gen(tmp_path, text: str) -> tuple[str, ...]:
    return fake_runner(tmp_path, f"""
        import sys
        prompt = [a for a in sys.argv if a.startswith("--prompt=")][0][len("--prompt="):]
        print(prompt + {text!r} + "<|eot_id|>")
        print({STATS!r})
    """)


def test_naming_only_one_of_two_possible_causes_falls_back(tmp_path):
    text = "You may be seeing a fungal leaf spot, likely caused by Septoria. A photo cannot be sure."
    r = explain(cfg(_gen(tmp_path, text)), PROMPT, TEMPLATE, cause_alternatives=LEAF_SPOT_CAUSES)
    assert (r.source, r.checker_result) == ("template", "fail") and r.text == TEMPLATE
    assert any("possible causes" in reason for reason in r.reasons)


def test_naming_both_possible_causes_passes(tmp_path):
    text = "You may be seeing a leaf spot from a fungus or from bacteria. A photo cannot tell which one it is."
    r = explain(cfg(_gen(tmp_path, text)), PROMPT, TEMPLATE, cause_alternatives=LEAF_SPOT_CAUSES)
    assert (r.source, r.checker_result) == ("llama", "pass")


def test_naming_no_cause_at_all_passes(tmp_path):
    text = "You may be seeing a leaf spot problem. A photo cannot tell for certain."
    r = explain(cfg(_gen(tmp_path, text)), PROMPT, TEMPLATE, cause_alternatives=LEAF_SPOT_CAUSES)
    assert r.source == "llama"


def test_prompt_speaks_to_the_farmer_and_keeps_all_causes():
    assert "'you'" in PROMPT and "photo cannot confirm" in PROMPT


# --- 21 Nov: rewrite-the-reviewed-text mode (real 1B model added facts in free explanation mode) -------------
REVIEWED = "The spots with rings look like early blight. A photo cannot confirm the disease."


def test_rewrite_prompt_carries_the_reviewed_text_and_forbids_additions():
    p = build_prompt("tomato", "early blight", "rings", "a photo cannot confirm it", reviewed=REVIEWED)
    assert f"reviewed_text={REVIEWED}" in p and "Add nothing new" in p


def test_preamble_and_quotes_are_removed_before_the_checker(tmp_path):
    text = 'Here is a rewritten version for the farmer:\n\n"You may be seeing early blight. A photo cannot confirm it."'
    r = explain(cfg(_gen(tmp_path, text)), PROMPT, TEMPLATE)
    assert (r.source, r.checker_result) == ("llama", "pass")
    assert r.text == "You may be seeing early blight. A photo cannot confirm it."


@pytest.mark.parametrize("text", [
    "I cannot provide advice on treating a plant disease. Can I help you with something else?",
    "You may be seeing early blight on the older leaves.",              # drops the photo warning
])
def test_refusals_and_missing_photo_warning_fall_back(tmp_path, text):
    r = explain(cfg(_gen(tmp_path, text)), PROMPT, TEMPLATE)
    assert (r.source, r.checker_result, r.text) == ("template", "fail", TEMPLATE)
