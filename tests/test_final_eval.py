import hashlib
import json

import numpy as np
import pytest

from ps1.decision import DecisionConfig
from ps1.decision import decide as app_decide
from ps1.vision.abstain_oe import decide as sheet_decide
from ps1.vision.final_eval import (
    apply_quality,
    claim_lock,
    evaluate_gates,
    listing_problems,
    prepare_and_lock,
    problems_before_sealed_run,
    score_rows,
    size_problems,
    to_probs,
    verify_sealed_hashes,
)

CLASSES = ("tomato_early_blight", "tomato_healthy", "tomato_late_blight", "tomato_leaf_spot", "tomato_other_problem")
SHA = "a" * 64
DECISION_SHA = "d" * 64
OK_GATES = {"status": "confirmed", "confirmed_by": "Sreyesh", "pte_sha256": SHA, "decision_sha256": DECISION_SHA,
            "threshold": 0.78, "quality_thresholds_version": "q1-dev", "input_gate": "waived",
            "blocking": {"accepted_error": {"max": 0.30}, "false_accept_seen": {"max": 0.10}, "coverage": {"min": 0.25}},
            "report_only": {"false_accept_novel": {"max": 0.30}}}
PINS = {"pte_sha256": SHA, "decision_sha256": DECISION_SHA, "gates_sha256": "g" * 64, "manifest_sha256": "m" * 64,
        "quality_thresholds": "q1-dev"}


def _cfg(threshold: float = 0.78, sha: str = SHA, classes: tuple = CLASSES) -> DecisionConfig:
    return DecisionConfig("d", sha, classes, 1.35, threshold, "test")


def _problems(gates: dict, quality: str = "q1-dev", cfg: DecisionConfig | None = None, model_classes: tuple = CLASSES) -> list[str]:
    return problems_before_sealed_run(gates, cfg or _cfg(), SHA, quality, DECISION_SHA, model_classes)


def test_confirmed_gates_have_no_problems():
    assert _problems(OK_GATES) == []


@pytest.mark.parametrize("change,word", [
    ({"status": "draft"}, "confirmed"),
    ({"confirmed_by": ""}, "confirmed_by"),
    ({"pte_sha256": "b" * 64}, "sha256"),
    ({"decision_sha256": "b" * 64}, "decision"),
    ({"threshold": 0.7}, "threshold"),
    ({"quality_thresholds_version": "q9"}, "quality"),
    ({"input_gate": "pending"}, "input gate"),
])
def test_each_unmet_condition_blocks_the_sealed_run(change, word):
    assert any(word in p for p in _problems({**OK_GATES, **change}))


def test_placeholder_quality_thresholds_block_the_sealed_run():
    gates = {**OK_GATES, "quality_thresholds_version": "q0-placeholder"}
    assert any("placeholder" in p for p in _problems(gates, quality="q0-placeholder"))


def test_a_built_input_gate_blocks_until_this_harness_runs_it():
    # final_eval does not call the input gate, so a sealed number would describe a different app than the one shipped.
    assert any("does not run" in p for p in _problems({**OK_GATES, "input_gate": "built"}))


def test_a_gate_on_a_metric_that_does_not_exist_blocks_before_the_lock():
    gates = {**OK_GATES, "blocking": {**OK_GATES["blocking"], "acepted_error": {"max": 0.3}}}
    assert any("acepted_error" in p for p in _problems(gates))


def test_a_permuted_class_order_blocks_the_sealed_run():
    swapped = (CLASSES[1], CLASSES[0], *CLASSES[2:])
    assert any("class order" in p for p in _problems(OK_GATES, cfg=_cfg(classes=swapped)))


# ---- the lock: one pinned configuration, ever ----

def test_a_different_configuration_is_refused(tmp_path):
    lock = tmp_path / "LOCK.json"
    assert claim_lock(lock, PINS) is False
    with pytest.raises(SystemExit, match="already"):
        claim_lock(lock, {**PINS, "gates_sha256": "z" * 64})


def test_the_same_configuration_resumes_because_it_can_only_give_the_same_numbers(tmp_path):
    lock = tmp_path / "LOCK.json"
    claim_lock(lock, PINS)
    assert claim_lock(lock, PINS) is True
    assert json.loads(lock.read_text())["pins"] == PINS


# ---- checks before the lock ----

def _listing(tmp_path, image_bytes: bytes = b"leaf", recorded: bytes | None = None):
    """Same layout as data/sealed_test.sha256: `<split>  <sha256>  <relative path, may hold spaces>`."""
    (tmp_path / "a b").mkdir(exist_ok=True)
    (tmp_path / "a b" / "x y.jpg").write_bytes(image_bytes)
    sha = hashlib.sha256(recorded if recorded is not None else image_bytes).hexdigest()
    listing = tmp_path / "sealed.sha256"
    listing.write_text(f"final_external_test  {sha}  a b/x y.jpg\n")
    row = {"path": str((tmp_path / "a b" / "x y.jpg").resolve()), "sha256": sha}
    return listing, tmp_path, row


def _prepare(tmp_path, listing, root, rows, decode=lambda r: None, expected=None, manifest_count=1):
    lock = tmp_path / "out" / "LOCK.json"
    prepare_and_lock(lock, PINS, {"known": rows}, expected or {"known": len(rows)}, listing, root, manifest_count, decode)
    return lock


def test_verify_reads_the_real_three_column_format_and_paths_with_spaces(tmp_path):
    listing, root, row = _listing(tmp_path)
    assert verify_sealed_hashes(listing, root) == {row["path"]: row["sha256"]}


def test_verify_detects_a_changed_image(tmp_path):
    listing, root, _ = _listing(tmp_path, recorded=b"something else")
    with pytest.raises(SystemExit, match="changed"):
        verify_sealed_hashes(listing, root)


def test_an_empty_listing_verifies_nothing_and_is_refused(tmp_path):
    listing = tmp_path / "empty.sha256"
    listing.write_text("")
    with pytest.raises(SystemExit, match="empty"):
        verify_sealed_hashes(listing, tmp_path)


def test_listing_problems_catch_an_unlisted_row_a_changed_row_and_a_short_listing():
    listed = {"/d/a.jpg": "1" * 64}
    assert listing_problems(listed, [{"path": "/d/a.jpg", "sha256": "1" * 64}], 1) == []
    assert any("not in" in p for p in listing_problems(listed, [{"path": "/d/b.jpg", "sha256": "1" * 64}], 1))
    assert any("hash" in p for p in listing_problems(listed, [{"path": "/d/a.jpg", "sha256": "2" * 64}], 1))
    assert any("11019" in p for p in listing_problems(listed, [{"path": "/d/a.jpg", "sha256": "1" * 64}], 11019))


def test_size_problems_name_the_set():
    assert size_problems({"known": [1, 2]}, {"known": 2}) == []
    assert any("known" in p for p in size_problems({"known": [1]}, {"known": 259}))


@pytest.mark.parametrize("break_it", ["hash", "size", "decode", "unlisted"])
def test_every_fixable_problem_leaves_the_single_use_lock_unclaimed(tmp_path, break_it):
    listing, root, row = _listing(tmp_path, recorded=b"something else" if break_it == "hash" else None)
    rows, kwargs = [row], {}
    if break_it == "size":
        kwargs["expected"] = {"known": 259}
    if break_it == "unlisted":
        rows = [{**row, "path": "/elsewhere.jpg"}]
    if break_it == "decode":
        def bad(_r):
            raise OSError("truncated file")
        kwargs["decode"] = bad
    with pytest.raises(SystemExit):
        _prepare(tmp_path, listing, root, rows, **kwargs)
    assert not (tmp_path / "out" / "LOCK.json").exists()


def test_the_lock_is_claimed_after_all_checks_pass(tmp_path):
    listing, root, row = _listing(tmp_path)
    assert _prepare(tmp_path, listing, root, [row]).exists()


# ---- scoring ----

class FakeVision:
    def __init__(self, fail_on: str | None = None):
        self.calls: list[str] = []
        self.fail_on = fail_on

    def preprocess(self, path: str) -> str:
        return path

    def logits(self, batch: str) -> np.ndarray:
        if batch == self.fail_on:
            raise MemoryError("killed mid-run")
        self.calls.append(batch)
        return np.array([1.0, 2.0, 0.0, 0.0, 0.0])


class Quality:
    def __init__(self, ok: bool):
        self.ok = ok


def test_the_model_never_sees_a_photo_the_quality_gate_rejected():
    vision = FakeVision()
    rows = [{"path": "good.jpg"}, {"path": "bad.jpg"}]
    records = score_rows(vision, rows, lambda p: Quality(p == "good.jpg"))
    assert vision.calls == ["good.jpg"]         # same as the app: a rejected photo stops before the model
    assert records[1]["quality_ok"] is False and records[1]["logits"] is None


def test_scores_survive_a_crash_and_a_rerun_does_not_rescore(tmp_path):
    rows = [{"path": "a.jpg"}, {"path": "b.jpg"}, {"path": "c.jpg"}]
    scores = tmp_path / "scores.jsonl"
    with pytest.raises(MemoryError):
        score_rows(FakeVision(fail_on="b.jpg"), rows, lambda p: Quality(True), scores)
    assert [json.loads(x)["path"] for x in scores.read_text().splitlines()] == ["a.jpg"]

    second = FakeVision()
    records = score_rows(second, rows, lambda p: Quality(True), scores)
    assert second.calls == ["b.jpg", "c.jpg"] and [r["path"] for r in records] == ["a.jpg", "b.jpg", "c.jpg"]


def test_a_half_written_last_line_is_ignored_on_resume(tmp_path):
    scores = tmp_path / "scores.jsonl"
    scores.write_text(json.dumps({"path": "a.jpg", "quality_ok": True, "logits": [1, 2, 0, 0, 0]}) + '\n{"path": "b.j')
    vision = FakeVision()
    records = score_rows(vision, [{"path": "a.jpg"}, {"path": "b.jpg"}], lambda p: Quality(True), scores)
    assert vision.calls == ["b.jpg"] and len(records) == 2


def test_quality_rejected_records_can_never_be_accepted():
    records = [{"path": "a", "quality_ok": True, "logits": [9.0, 0.0, 0.0, 0.0, 0.0]},
               {"path": "b", "quality_ok": False, "logits": None}]
    probs = to_probs(records, _cfg())
    accepted, _ = sheet_decide(probs, 4, 0.5)
    assert accepted.tolist() == [True, False]


def test_apply_quality_does_not_modify_its_input():
    probs = np.array([[0.9, 0.025, 0.025, 0.025, 0.025], [0.9, 0.025, 0.025, 0.025, 0.025]])
    out = apply_quality(probs, np.array([True, False]), other_idx=4)
    assert out[1, 4] == 1.0 and probs[1, 0] == 0.9


def test_the_evaluation_rule_matches_the_app_decision_engine():
    rng = np.random.default_rng(0)
    logits = rng.normal(0, 3, size=(500, 5))
    cfg = _cfg(threshold=0.78)
    records = [{"path": str(i), "quality_ok": True, "logits": row.tolist()} for i, row in enumerate(logits)]
    accepted, _ = sheet_decide(to_probs(records, cfg), 4, cfg.threshold)
    app = [app_decide(row, cfg, CLASSES, SHA).label is not None for row in logits]
    assert accepted.tolist() == app


# ---- gates ----

def test_evaluate_gates_max_and_min_and_overall():
    row = {"accepted_error": {"rate": 0.2, "low": 0.1, "high": 0.3}, "false_accept_seen": {"rate": 0.15, "low": 0.1, "high": 0.2},
           "coverage": {"rate": 0.4, "low": 0.3, "high": 0.5}, "false_accept_novel": {"rate": 0.5, "low": 0.3, "high": 0.7}}
    out = evaluate_gates(row, OK_GATES)
    assert out["blocking"]["accepted_error"]["passed"] and out["blocking"]["coverage"]["passed"]
    assert not out["blocking"]["false_accept_seen"]["passed"]
    assert not out["report_only"]["false_accept_novel"]["passed"]
    assert out["all_blocking_passed"] is False


def test_nothing_accepted_fails_a_max_gate_rather_than_passing_it():
    row = {"accepted_error": {"rate": None, "low": 0.0, "high": 1.0}}
    gates = {"blocking": {"accepted_error": {"max": 0.3}}, "report_only": {}}
    assert evaluate_gates(row, gates)["all_blocking_passed"] is False
