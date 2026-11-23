"""Buffered Llama explanation with checker + template fallback (v4 P0-2, P0-6; addenda #3).

Flow: build prompt from card fields -> run llama_main once (greedy, hard timeout)
-> capture the *whole* output off-screen -> checker -> reveal or template.
Nothing is shown to the farmer before the checker has seen the complete text.
"""
import contextlib
import os
import re
import shlex
import signal
import subprocess
import time
from dataclasses import dataclass, field

from ps1.llm.checker import check_llama_text, keeps_cause_alternatives

MAX_NEW_TOKENS = 120
DEFAULT_TIMEOUT_S = 20.0          # warm-path budget from addenda #3; tune after G2
CHARS_PER_TOKEN_ESTIMATE = 3.5    # Llama-3 tokenizer on English; only used to size seq_len

SYSTEM_RULES = (
    "You rewrite reviewed plant-health advice for a farmer in simple English. Use only the facts given. "
    "Do not add treatments, products, doses, numbers, weather or causes that are not given. "
    "Keep the warning that a photo cannot confirm the problem. Speak to the farmer as 'you'. "
    "Write two or three short sentences."
)
# 21 Nov, real model on x86: in free "explain" mode the 1B model added facts the card does not have (e.g.
# "bacterial spot, which is also caused by Septoria"); rewriting the reviewed text kept to the card.
PREAMBLE = re.compile(r"^\s*(here is|here's|sure|okay|rewritten)[^\n]*:\s*\n+", re.IGNORECASE)
OBSERVER = re.compile(r"PyTorchObserver (\{.*\})")
ASSISTANT_HEADER = "<|start_header_id|>assistant<|end_header_id|>"


@dataclass(frozen=True)
class LlamaConfig:
    """`command` is the runner prefix: native path on the Pi, `podman run ... llama_main` on x86."""
    command: tuple[str, ...]
    model_path: str
    tokenizer_path: str
    cpu_threads: int = 4
    timeout_s: float = DEFAULT_TIMEOUT_S


@dataclass(frozen=True)
class Explanation:
    text: str
    source: str                   # "llama" | "template"
    checker_result: str           # "pass" | "fail" | "timeout" | "error"
    reasons: tuple[str, ...] = field(default_factory=tuple)
    latency_s: float = 0.0
    raw_output: str = ""          # kept for logs/evaluation, never displayed


def build_prompt(crop: str, finding_label: str, symptoms: str, limits: str, cause: str = "",
                 reviewed: str = "") -> str:
    """With `reviewed` (the card's checked template) Llama only rewrites it in simpler words (the app path).

    Without it, the older card-fields prompt is built (kept for the G1 benchmark prompts). `cause` comes from
    the card so the model can explain it without inventing one (G1 observation).
    """
    if reviewed:
        facts, ask = f"reviewed_text={reviewed}", "Rewrite the reviewed_text for the farmer in simpler words. Add nothing new."
    else:
        cause_part = f"; cause={cause}" if cause else ""
        facts = f"crop={crop}; finding=likely {finding_label}{cause_part}; symptoms={symptoms}; limits={limits}."
        ask = "Explain to the farmer what you may be seeing and why it is only a likely match."
    return (
        f"<|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_RULES}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\nFACTS: {facts}\n{ask}<|eot_id|>"
        f"{ASSISTANT_HEADER}\n\n"
    )


def clean_generated(text: str) -> str:
    """Only removes text: a leading 'Here is a rewritten version:' line and wrapping quotation marks."""
    text = PREAMBLE.sub("", text, count=1).strip()
    if len(text) >= 2 and text[0] in "\"“" and text[-1] in "\"”":
        text = text[1:-1].strip()
    elif text[:1] in "\"“" and text.count('"') + text.count("“") + text.count("”") == 1:
        text = text[1:].strip()                     # an opening quote whose closing one was never generated
    return text


def seq_len_for(prompt: str) -> int:
    """llama_main's --seq_len is prompt + output; bound output to MAX_NEW_TOKENS."""
    return int(len(prompt) / CHARS_PER_TOKEN_ESTIMATE) + MAX_NEW_TOKENS


def extract_generated(stdout: str) -> str:
    """The runner echoes prompt then generated tokens, then prints stats; keep only the generation."""
    body = stdout.split("PyTorchObserver", 1)[0]
    body = body.rsplit(ASSISTANT_HEADER, 1)[-1]
    return body.replace("<|eot_id|>", "").replace("<|end_of_text|>", "").strip()


def explain(cfg: LlamaConfig, prompt: str, template_text: str, lang: str = "en",
            cause_alternatives: tuple[tuple[str, ...], ...] = ()) -> Explanation:
    cmd = [*cfg.command, f"--model_path={cfg.model_path}", f"--tokenizer_path={cfg.tokenizer_path}",
           f"--prompt={prompt}", f"--seq_len={seq_len_for(prompt)}", "--temperature=0",
           f"--cpu_threads={cfg.cpu_threads}"]
    start = time.monotonic()
    try:
        returncode, stdout, stderr = _run(cmd, cfg.timeout_s)
    except subprocess.TimeoutExpired:
        return Explanation(template_text, "template", "timeout", ("llama timed out",), time.monotonic() - start)
    except (OSError, subprocess.SubprocessError) as exc:
        return Explanation(template_text, "template", "error", (f"runner failed: {exc}",),
                           time.monotonic() - start)
    latency = time.monotonic() - start
    if returncode != 0 or not OBSERVER.search(stdout):
        reason = f"runner exit {returncode}: {shlex.quote(stderr[-200:])}"
        return Explanation(template_text, "template", "error", (reason,), latency, stdout)

    generated = clean_generated(extract_generated(stdout))
    verdict = check_llama_text(generated, lang=lang)
    if not verdict.passed:
        return Explanation(template_text, "template", "fail", verdict.reasons, latency, stdout)
    if not keeps_cause_alternatives(generated, cause_alternatives):
        return Explanation(template_text, "template", "fail", ("names only one of the card's possible causes",),
                           latency, stdout)
    return Explanation(generated, "llama", "pass", (), latency, stdout)


def _run(cmd: list[str], timeout_s: float) -> tuple[int, str, str]:
    """Run the runner in its own process group; on timeout kill the whole group, not just the parent.

    Output is decoded with errors="replace": a generation cut inside a multi-byte character must
    reach the checker (which then rejects or passes it), never crash the request.
    """
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    try:
        out, err = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        proc.communicate()
        raise
    return proc.returncode, out.decode("utf-8", errors="replace"), err.decode("utf-8", errors="replace")
