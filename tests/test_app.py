"""Local offline web app (ps1/app): Screen 1 at once, Screen 2 after the checker, nothing leaves the device."""
import io
import re
from pathlib import Path

import cv2
import numpy as np
import pytest

from ps1.app.server import create_app
from tests.test_pipeline import cfg, fake_llama, logits_for

APP_DIR = Path(__file__).resolve().parents[1] / "ps1" / "app"


def png_bytes(blur: float = 0.0) -> bytes:
    img = np.random.default_rng(0).integers(60, 200, size=(480, 640, 3), dtype=np.uint8)
    if blur:
        img = cv2.GaussianBlur(img, (0, 0), sigmaX=blur)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def make(tmp_path, label="tomato_early_blight", llama=None, **kw):
    app = create_app(cfg(logits_for(label), llama=llama), upload_dir=tmp_path / "uploads", **kw)
    app.config["TESTING"] = True
    return app, app.test_client()


def diagnose(client, data: bytes, name="leaf.png", lang="en"):
    return client.post("/api/diagnose", data={"photo": (io.BytesIO(data), name), "lang": lang},
                       content_type="multipart/form-data")


def test_health_reports_model_draft_cards_and_languages(tmp_path):
    _, c = make(tmp_path)
    h = c.get("/api/health").get_json()
    assert h["model"] == "f" * 12 and h["decision"] == "d-test"
    assert h["cards_reviewed"] is False                 # drafts must be visible to the user
    assert h["languages"] == ["en"] and h["llama"] is False



def test_model_note_is_shown_on_the_page_and_in_health(tmp_path):
    note = "Model v8 (extra field data, after the sealed test). The submitted model is v7."
    _, c = make(tmp_path, model_note=note)
    assert note in c.get("/").get_data(as_text=True)    # a viewer can see which model answers
    assert c.get("/api/health").get_json()["model_note"] == note


def test_no_model_note_by_default(tmp_path):
    _, c = make(tmp_path)
    assert 'id="model-note"' not in c.get("/").get_data(as_text=True)
    assert c.get("/api/health").get_json()["model_note"] is None

def test_accepted_photo_gets_card_now_and_upload_is_deleted(tmp_path):
    app, c = make(tmp_path)
    r = diagnose(c, png_bytes())
    assert r.status_code == 200
    j = r.get_json()
    assert j["finding"]["label"] == "tomato_early_blight" and not j["finding"]["abstained"]
    assert j["title"] == "Early blight (likely)"
    assert j["screen1"]["card_id"] == "tomato_early_blight" and j["screen1"]["actions"]
    assert j["chemical_control_line"] and j["screen2_status"] == "none"
    assert "model_score" not in j["finding"]                       # never shown as a probability
    assert list((tmp_path / "uploads").iterdir()) == []             # the photo is not kept


def test_blurry_photo_abstains_without_a_card(tmp_path):
    _, c = make(tmp_path)
    j = diagnose(c, png_bytes(blur=8)).get_json()
    assert j["finding"]["abstained"] and j["finding"]["abstain_reason"] == "poor_quality"
    assert j["screen1"]["card_id"] is None and j["title"] is None


def test_unreadable_file_abstains_as_poor_quality(tmp_path):
    _, c = make(tmp_path)
    j = diagnose(c, b"not an image at all", name="leaf.jpg").get_json()
    assert j["finding"]["abstain_reason"] == "poor_quality"


@pytest.mark.parametrize("data,name,lang,code", [
    (None, None, "en", 400),                       # no photo
    (b"x", "leaf.exe", "en", 400),                 # not an image extension
    (b"x", "leaf.png", "te", 400),                 # no reviewed/drafted Telugu cards yet
    (b"x", "leaf.png", "fr", 400),                 # not a supported language
])
def test_bad_requests_are_rejected(tmp_path, data, name, lang, code):
    _, c = make(tmp_path)
    form = {"lang": lang}
    if data is not None:
        form["photo"] = (io.BytesIO(data), name)
    r = c.post("/api/diagnose", data=form, content_type="multipart/form-data")
    assert r.status_code == code and "error" in r.get_json()


def test_too_large_upload_is_rejected(tmp_path):
    _, c = make(tmp_path, max_upload_bytes=1000)
    r = diagnose(c, png_bytes())
    assert r.status_code == 413 and "error" in r.get_json()


def test_screen2_arrives_after_the_checker(tmp_path):
    text = "The ringed brown spots on older leaves look like early blight. A photo cannot tell for certain."
    app, c = make(tmp_path, llama=fake_llama(tmp_path, text))
    j = diagnose(c, png_bytes()).get_json()
    assert j["screen2_status"] == "pending" and j["screen1"]["explanation_source"] == "template"
    app.config["PS1_WORKER"].drain()
    e = c.get(f"/api/explain/{j['job']}").get_json()
    assert e["status"] == "done" and e["screen2"]["explanation"] == text
    assert e["screen2"]["actions"] == j["screen1"]["actions"]       # Llama never changes actions


def test_unsafe_llama_text_never_reaches_the_user(tmp_path):
    app, c = make(tmp_path, llama=fake_llama(tmp_path, "Spray mancozeb weekly."))
    j = diagnose(c, png_bytes()).get_json()
    app.config["PS1_WORKER"].drain()
    e = c.get(f"/api/explain/{j['job']}").get_json()
    assert "mancozeb" not in str(e).lower()
    assert e["screen2"]["explanation"] == j["screen1"]["explanation"]


def test_no_llama_call_when_nothing_was_named(tmp_path):
    app, c = make(tmp_path, label="tomato_other_problem", llama=fake_llama(tmp_path, "anything"))
    j = diagnose(c, png_bytes()).get_json()
    assert j["finding"]["abstain_reason"] == "unsupported_label" and j["screen2_status"] == "none"


def test_unknown_job_is_404(tmp_path):
    _, c = make(tmp_path)
    assert c.get("/api/explain/nope").status_code == 404


def test_page_and_static_files_load_nothing_from_the_internet(tmp_path):
    _, c = make(tmp_path)
    page = c.get("/")
    assert page.status_code == 200 and b"<form" in page.data
    for f in list(APP_DIR.glob("templates/*")) + list(APP_DIR.glob("static/*")):
        text = f.read_text(encoding="utf-8")
        assert not re.search(r"(https?:)?//[a-z0-9.-]+\.[a-z]{2,}", text, re.I), f"external URL in {f.name}"


def test_security_headers_allow_only_this_device(tmp_path):
    _, c = make(tmp_path)
    h = c.get("/").headers
    assert "default-src 'self'" in h["Content-Security-Policy"] and h["X-Frame-Options"] == "DENY"
    assert h["X-Content-Type-Options"] == "nosniff"


def test_new_photo_cancels_older_explanations_not_yet_started(tmp_path):
    import threading
    gate = threading.Event()
    text = "The ringed brown spots on older leaves look like early blight. A photo cannot tell for certain."
    app, c = make(tmp_path, llama=fake_llama(tmp_path, text))
    worker = app.config["PS1_WORKER"]
    worker.llama(gate.wait)                                  # hold the Llama lane busy
    first = diagnose(c, png_bytes()).get_json()               # Screen 1 still arrives while Llama is busy
    second = diagnose(c, png_bytes()).get_json()
    assert first["screen1"]["card_id"] == "tomato_early_blight"
    gate.set()
    worker.drain()
    assert c.get(f"/api/explain/{first['job']}").get_json()["status"] == "cancelled"
    assert c.get(f"/api/explain/{second['job']}").get_json()["status"] == "done"


def test_huge_image_is_rejected_before_decoding(tmp_path):
    from PIL import Image
    big = io.BytesIO()
    Image.new("L", (9000, 8000)).save(big, format="PNG")          # 72 MP of zeros: tiny file, huge decode
    _, c = make(tmp_path)
    r = diagnose(c, big.getvalue())
    assert r.status_code == 400 and "too large" in r.get_json()["error"]


def test_full_vision_queue_returns_503(tmp_path):
    _, c = make(tmp_path, max_pending_vision=0)
    r = diagnose(c, png_bytes())
    assert r.status_code == 503 and "busy" in r.get_json()["error"]
    assert list((tmp_path / "uploads").iterdir()) == []


def test_one_phone_does_not_cancel_another_phones_explanation(tmp_path):
    import threading
    gate = threading.Event()
    text = "The ringed brown spots on older leaves look like early blight. A photo cannot tell for certain."
    app, c = make(tmp_path, llama=fake_llama(tmp_path, text))
    worker = app.config["PS1_WORKER"]
    worker.llama(gate.wait)
    a = c.post("/api/diagnose", data={"photo": (io.BytesIO(png_bytes()), "a.png"), "lang": "en"},
               content_type="multipart/form-data", environ_base={"REMOTE_ADDR": "10.42.0.10"}).get_json()
    b = c.post("/api/diagnose", data={"photo": (io.BytesIO(png_bytes()), "b.png"), "lang": "en"},
               content_type="multipart/form-data", environ_base={"REMOTE_ADDR": "10.42.0.11"}).get_json()
    gate.set()
    worker.drain()
    assert c.get(f"/api/explain/{a['job']}").get_json()["status"] == "done"
    assert c.get(f"/api/explain/{b['job']}").get_json()["status"] == "done"
