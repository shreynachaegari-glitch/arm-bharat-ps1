"""Local offline web app: photo in, Screen 1 (reviewed card) at once, Screen 2 (checked Llama text) later.

    farmer's browser --photo--> POST /api/diagnose --> [vision lane] run_screen1 --> JSON (Screen 1)
                                                                          \\-> [llama lane] run_screen2 (buffered)
    farmer's browser --poll--> GET /api/explain/<job> --> pending | done (checked text) | none | error

Vision and Llama each run on their own single thread, in order, so Screen 1 never queues behind Llama; a new
photo cancels older Screen 2 jobs that have not started. The photo is deleted as soon as Screen 1 is built.
Nothing is fetched from the internet; the page ships its own CSS/JS. Usage (x86 dev, draft cards):
  python -m ps1.app --pte runs/v7_final/export/tomato_effnet_b0_t2_fp32_xnnpack.pte \\
      --decision runs/v7_final/export/decision.json --allow-drafts [--llama] [--host 0.0.0.0] [--port 5000]
"""
import argparse
import logging
import os
import tempfile
import threading
import uuid
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, get_args

from flask import Flask, jsonify, render_template, request
from PIL import Image
from werkzeug.exceptions import RequestEntityTooLarge

from ps1.contracts import Lang, Request
from ps1.pipeline import (
    PipelineConfig,
    Screen1Result,
    assumed_tomato_gate,
    needs_explanation,
    run_screen1,
    run_screen2,
)

log = logging.getLogger("ps1.app")

ALLOWED_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})
DEFAULT_MAX_UPLOAD_BYTES = 15 * 1024 * 1024          # a 12 MP phone JPEG is ~3-6 MB
SCREEN1_TIMEOUT_S = 60.0
MAX_JOBS_KEPT = 32
MAX_PENDING_VISION = 4                               # queued + running photos; more gets 503 (busy)
MAX_IMAGE_PIXELS = 64_000_000                        # 64 MP: allows 48-50 MP phone cameras; decode ~190 MB
HIDDEN_FINDING_FIELDS = frozenset({"model_score"})   # internal only, never shown as a probability (contracts)


class PipelineWorker:
    """Two single-thread lanes: vision (Screen 1) and Llama (Screen 2). Each lane runs in order.

    Screen 1 never waits behind a running Llama call (21 Nov x86: 5.1 s queued vs 0.34 s idle); the cost is
    some CPU sharing while both run, which only matters if the farmer sends a new photo mid-explanation.
    """

    def __init__(self) -> None:
        self._vision = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ps1-vision")
        self._llama = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ps1-llama")

    def vision(self, fn: Callable[..., Any], *args: Any) -> Future:
        return self._vision.submit(fn, *args)

    def llama(self, fn: Callable[..., Any], *args: Any) -> Future:
        return self._llama.submit(fn, *args)

    def drain(self) -> None:
        """Block until everything submitted so far has finished on both lanes."""
        self._vision.submit(lambda: None).result()
        self._llama.submit(lambda: None).result()


class JobTable:
    """Screen 2 futures by job id, bounded and thread-safe.

    A new photo cancels the not-yet-started Screen 2 jobs of the SAME client only (another phone on the hotspot
    keeps its own). Submitting and registering happen under one lock, so a slower request can never cancel a
    newer request's job (review 21 Nov).
    """

    def __init__(self, limit: int = MAX_JOBS_KEPT) -> None:
        self._jobs: OrderedDict[str, tuple[str, Future | None]] = OrderedDict()
        self._lock = threading.Lock()
        self._limit = limit

    def add(self, client: str, submit: Callable[[], Future | None]) -> str:
        job_id = uuid.uuid4().hex
        with self._lock:
            for owner, old in self._jobs.values():
                if owner == client and old is not None:
                    old.cancel()                     # no-op once running or done
            self._jobs[job_id] = (client, submit())
            while len(self._jobs) > self._limit:
                self._jobs.popitem(last=False)
        return job_id

    def get(self, job_id: str) -> tuple[bool, Future | None]:
        with self._lock:
            entry = self._jobs.get(job_id)
        return entry is not None, entry[1] if entry else None


class Gauge:
    """Counts queued + running vision jobs so a flood of uploads gets 503 instead of an unbounded queue."""

    def __init__(self, limit: int) -> None:
        self._n, self._limit, self._lock = 0, limit, threading.Lock()

    def try_acquire(self) -> bool:
        with self._lock:
            if self._n >= self._limit:
                return False
            self._n += 1
            return True

    def release(self, _: object = None) -> None:
        with self._lock:
            self._n -= 1


def _error(message: str, status: int):
    return jsonify({"error": message}), status


def _screen1_json(s1: Screen1Result, job_id: str, screen2_status: str) -> dict:
    finding = {k: v for k, v in s1.finding.model_dump(mode="json").items() if k not in HIDDEN_FINDING_FIELDS}
    return {"job": job_id, "finding": finding, "title": s1.card.title if s1.card else None,
            "screen1": s1.screen1.model_dump(mode="json"), "chemical_control_line": s1.chemical_control_line,
            "timings_ms": {k: round(v, 1) for k, v in s1.timings_ms.items()}, "screen2_status": screen2_status}


def _save_upload(storage, upload_dir: Path) -> Path:
    """Store the upload under a name we generate (never the client's), keeping only a checked suffix.

    On any failure the temp file is removed before the error propagates."""
    suffix = Path(storage.filename or "").suffix.lower()
    fd, name = tempfile.mkstemp(dir=upload_dir, suffix=suffix, prefix="photo_")
    os.close(fd)
    try:
        storage.save(name)
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise
    return Path(name)


def _too_many_pixels(path: Path) -> bool:
    """Read only the image header: a small PNG can decode to gigabytes (decode bomb on an 8 GB Pi)."""
    try:
        with Image.open(path) as img:
            width, height = img.size
    except (OSError, ValueError, Image.DecompressionBombError):
        return False                                   # unreadable: the quality gate rejects it as such
    return width * height > MAX_IMAGE_PIXELS


def create_app(cfg: PipelineConfig, *, upload_dir: Path | None = None,
               max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
               max_pending_vision: int = MAX_PENDING_VISION, model_note: str | None = None) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = max_upload_bytes
    upload_dir = Path(upload_dir or tempfile.mkdtemp(prefix="ps1_uploads_"))
    upload_dir.mkdir(parents=True, exist_ok=True)
    worker, jobs, vision_slots = PipelineWorker(), JobTable(), Gauge(max_pending_vision)
    app.config["PS1_WORKER"] = worker
    languages = sorted({card.lang for card in cfg.cards.values()})

    @app.after_request
    def security_headers(resp):
        resp.headers["Content-Security-Policy"] = ("default-src 'self'; img-src 'self' blob:; object-src 'none'; "
                                                   "base-uri 'self'; frame-ancestors 'none'; form-action 'self'")
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers["Referrer-Policy"] = "no-referrer"
        return resp

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_: RequestEntityTooLarge):
        return _error(f"photo is larger than {max_upload_bytes // (1024 * 1024) or 1} MB", 413)

    @app.get("/")
    def index():
        return render_template("index.html", cards_reviewed=cfg.cards_reviewed, languages=languages,
                               llama=cfg.llama is not None, model_note=model_note)

    @app.get("/api/health")
    def health():
        return jsonify({"model": cfg.vision.pte_sha256[:12], "decision": cfg.decision.version, "model_note": model_note,
                        "cards_reviewed": cfg.cards_reviewed, "languages": languages,
                        "llama": cfg.llama is not None,
                        "input_gate": "assumed (W6 not built)" if cfg.input_gate is assumed_tomato_gate else "model"})

    @app.post("/api/diagnose")
    def diagnose():
        lang = request.form.get("lang", "en")
        if lang not in get_args(Lang):
            return _error(f"unsupported language {lang!r}", 400)
        if lang not in languages:
            return _error(f"no advice cards in {lang!r} yet; available: {', '.join(languages)}", 400)
        storage = request.files.get("photo")
        if storage is None or not storage.filename:
            return _error("no photo attached", 400)
        if Path(storage.filename).suffix.lower() not in ALLOWED_SUFFIXES:
            return _error("the photo must be a JPEG, PNG, WebP or BMP image", 400)

        if not vision_slots.try_acquire():
            return _error("the assistant is busy; please try again in a moment", 503)
        path: Path | None = None
        future: Future | None = None
        try:
            path = _save_upload(storage, upload_dir)
            if _too_many_pixels(path):
                vision_slots.release()
                return _error("the photo is too large; please take it again at normal camera size", 400)
            req = Request(crop="tomato", lang=lang, image_path=str(path))
            future = worker.vision(run_screen1, req, cfg)
            future.add_done_callback(vision_slots.release)
            s1: Screen1Result = future.result(timeout=SCREEN1_TIMEOUT_S)
        except Exception:
            if future is None:
                vision_slots.release()                      # failed before the job was queued
            else:
                future.cancel()                             # timed out while still queued: drop it
            log.exception("screen 1 failed")
            return _error("the photo could not be checked; please try again", 500)
        finally:
            if path is not None:
                path.unlink(missing_ok=True)                # the photo is never kept

        wants_screen2 = cfg.llama is not None and needs_explanation(s1)
        job_id = jobs.add(request.remote_addr or "local",
                          lambda: worker.llama(run_screen2, s1, lang, cfg) if wants_screen2 else None)
        return jsonify(_screen1_json(s1, job_id, "pending" if wants_screen2 else "none"))

    @app.get("/api/explain/<job_id>")
    def explain(job_id: str):
        known, future = jobs.get(job_id)
        if not known:
            return _error("unknown or expired job", 404)
        if future is None:
            return jsonify({"status": "none", "screen2": None})
        if future.cancelled():
            return jsonify({"status": "cancelled", "screen2": None})
        if not future.done():
            return jsonify({"status": "pending", "screen2": None})
        try:
            screen2 = future.result()
        except Exception:
            log.exception("screen 2 failed")
            return jsonify({"status": "error", "screen2": None})
        if screen2 is None:
            return jsonify({"status": "none", "screen2": None})
        return jsonify({"status": "done", "screen2": screen2.model_dump(mode="json")})

    return app


def main() -> None:
    from waitress import serve

    from ps1.pipeline import build_config, default_llama_config

    ap = argparse.ArgumentParser(description="PS1 offline assistant: local web app")
    ap.add_argument("--pte", required=True)
    ap.add_argument("--decision", required=True)
    ap.add_argument("--allow-drafts", action="store_true", help="dev only: use unreviewed cards (shown as DRAFT)")
    ap.add_argument("--llama", action="store_true", help="enable the buffered Llama explanation (Screen 2)")
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to serve phones on the Pi's own hotspot")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--model-note", help="banner naming a non-default model (e.g. v8), so viewers know which answers")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = Path(__file__).resolve().parents[2]
    cfg = build_config(args.pte, args.decision, args.allow_drafts, default_llama_config(root) if args.llama else None)
    app = create_app(cfg, model_note=args.model_note)
    log.info("serving on http://%s:%d (model %s, cards %s, llama %s)", args.host, args.port,
             cfg.vision.pte_sha256[:12], "reviewed" if cfg.cards_reviewed else "DRAFT", cfg.llama is not None)
    serve(app, host=args.host, port=args.port, threads=4)


if __name__ == "__main__":
    main()
