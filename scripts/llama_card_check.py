"""Screen 2 evidence: run the real Llama .pte once per disease card through the app path; save what was shown.

Greedy decoding (temperature 0), so each card's output is deterministic for a given prompt, model and runtime.
Writes runs/llama_cards/card_check.json for scripts/make_report.py. x86: the G1 container; Pi: native llama_main.
  python scripts/llama_card_check.py
"""
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ps1.cards import load_cards
from ps1.llm.checker import CHECKER_VERSION
from ps1.llm.explain import (
    SYSTEM_RULES,
    build_prompt,
    clean_generated,
    explain,
    extract_generated,
)
from ps1.pipeline import default_llama_config

OUT = ROOT / "runs/llama_cards/card_check.json"


def main() -> int:
    cfg = default_llama_config(ROOT)
    cards, _ = load_cards(allow_drafts=True)
    rows = []
    for label, card in sorted(cards.items()):
        if not card.actions:                         # the app does not call Llama for these (healthy)
            rows.append({"card": label, "called": False})
            continue
        prompt = build_prompt(card.crop, card.title.split(" (")[0].lower(), card.symptoms, card.limits, card.cause,
                              reviewed=card.template)
        r = explain(cfg, prompt, card.template, cause_alternatives=card.cause_alternatives)
        rows.append({"card": label, "card_version": card.card_version, "called": True, "shown_source": r.source,
                     "checker_result": r.checker_result, "reasons": list(r.reasons), "latency_s": round(r.latency_s, 2),
                     "shown_text": r.text,
                     "generated": clean_generated(extract_generated(r.raw_output)) if r.raw_output else ""})
        print(f"{label}: {r.source}/{r.checker_result} {list(r.reasons)} {r.latency_s:.1f}s")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"generated": datetime.now().isoformat(timespec="seconds"), "checker": CHECKER_VERSION,
                               "runner": list(cfg.command), "system_rules": SYSTEM_RULES, "rows": rows}, indent=1))
    print(OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
