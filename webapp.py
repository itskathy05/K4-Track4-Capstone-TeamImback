"""Browser demo for the fall detector (stdlib HTTP server, no extra dependencies).

Usage: python webapp.py  ->  open http://127.0.0.1:8000
Serves an MJPEG stream of the annotated video, live status as JSON, and live threshold tuning.
Binds to localhost only: it can open the webcam and accepts uploads.
"""
import argparse
import collections
import contextlib
import copy
import dataclasses
import json
import math
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2

from evaluate import score_clip
from fall_detector import FALL_ALERT, Config, FallDetector
from main import draw, process, run

ROOT = Path(__file__).parent
CLIPS = ROOT / "test_videos"
UPLOADS = ROOT / "uploads"
PAGE = ROOT / "web" / "index.html"
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MODELS = {name: ROOT / "models" / f"pose_landmarker_{name}.task" for name in ("lite", "full")}
MAX_UPLOAD = 200 * 1024 * 1024
# activities in the public UR Fall clips (labelled by watching them; see README)
ACTIVITY = {"adl-01": "crouch", "adl-02": "crouch", "adl-04": "bend down", "adl-05": "bend down",
            "adl-28": "bend down", "adl-07": "sit on armchair", "adl-08": "sit on armchair",
            "adl-09": "sit on bed edge", "adl-25": "sit on chair", "adl-10": "lie on bed",
            "adl-11": "lie on bed", "adl-36": "lie on bed", "adl-37": "lie on bed",
            "adl-19": "dark room (pose loss)", "adl-30": "on all fours", "adl-31": "on all fours"}
LIMITS = {"min_visibility": (0.0, 1.0), "smooth_alpha": (0.01, 1.0),
          "horizontal_angle": (1.0, 89.0), "upright_angle": (1.0, 89.0)}  # others: (0, 1000]


def list_sources():
    """Selectable sources: webcam, downloaded UR Fall clips, uploaded videos."""
    out = [{"id": "cam:0", "label": "Webcam 0", "group": "Webcam"},
           {"id": "cam:1", "label": "Webcam 1", "group": "Webcam"}]
    for p in sorted(CLIPS.glob("*/*-cam0.mp4")):
        name = p.stem.replace("-cam0", "")
        kind = "Falls" if name.startswith("fall") else "Daily activities"
        label = f"{name} - {'fall' if kind == 'Falls' else ACTIVITY.get(name, 'daily activity')} ({p.parent.name})"
        out.append({"id": p.relative_to(ROOT).as_posix(), "label": label, "group": kind})
    for p in sorted(UPLOADS.glob("*")):
        if p.suffix.lower() in VIDEO_EXT:
            out.append({"id": p.relative_to(ROOT).as_posix(), "label": p.name, "group": "Uploads"})
    return out


def resolve_source(source_id):
    """Map a source id to (path_or_cam_index, crop_right, label). Only listed sources are allowed."""
    for s in list_sources():
        if s["id"] == source_id:
            if source_id.startswith("cam:"):
                return source_id[4:], False, s["label"]
            return str(ROOT / source_id), source_id.startswith("test_videos/"), s["label"]
    raise ValueError(f"unknown source: {source_id!r}")


def resolve_model(name):
    if name not in MODELS:
        raise ValueError(f"unknown model: {name!r}")
    return str(MODELS[name])


def benchmark_clips(split):
    """[(clip name, path)] for 'tune', 'holdout' or 'all'."""
    if split not in ("tune", "holdout", "all"):
        raise ValueError(f"unknown split: {split!r}")
    pattern = "*/*-cam0.mp4" if split == "all" else f"{split}/*-cam0.mp4"
    return [(p.stem.replace("-cam0", ""), str(p)) for p in sorted(CLIPS.glob(pattern))]


def safe_upload_name(name):
    name = re.sub(r"[^\w.-]", "_", Path(name.replace("\\", "/")).name)
    if Path(name).suffix.lower() not in VIDEO_EXT:
        raise ValueError(f"unsupported video type: {name}")
    return name


def update_config(cfg, data):
    """Validate all fields first, then apply, so a bad request changes nothing."""
    fields = {f.name for f in dataclasses.fields(cfg)}
    clean = {}
    for key, value in data.items():
        if key not in fields:
            raise ValueError(f"unknown setting: {key}")
        try:
            v = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{key}: not a number") from None
        lo, hi = LIMITS.get(key, (0.0, 1000.0))
        if not (math.isfinite(v) and lo <= v <= hi and (v > 0 or key == "min_visibility")):
            raise ValueError(f"{key}: must be in ({lo}, {hi}]")
        clean[key] = v
    for key, v in clean.items():
        setattr(cfg, key, v)
    return cfg


def parse_results(md):
    """Rows of the markdown table that evaluate.py prints."""
    lines = [ln.strip() for ln in md.splitlines() if ln.strip().startswith("|")]
    if len(lines) < 2:
        return []
    cells = [[c.strip() for c in ln.strip("|").split("|")] for ln in lines]
    return [dict(zip(cells[0], row)) for row in cells[2:]]


class Benchmark:
    """Re-runs the dataset evaluation in the background with a frozen copy of the current thresholds."""

    def __init__(self, runner=None):
        self.runner = runner or (lambda path, model, cfg: run(path, model, crop_right=True, show=False, config=cfg))
        self.lock = threading.Lock()
        self.state = {"running": False, "split": None, "done": 0, "total": 0, "rows": [], "correct": 0}
        self._thread, self._cancel = None, threading.Event()

    def start(self, clips, model, cfg, split):
        self.cancel()
        frozen = copy.copy(cfg)
        with self.lock:
            self.state = {"running": True, "split": split, "done": 0, "total": len(clips), "rows": [], "correct": 0,
                          "config": dataclasses.asdict(frozen)}
        self._cancel = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(clips, model, frozen, self._cancel), daemon=True)
        self._thread.start()

    def cancel(self):
        if self._thread:
            self._cancel.set()
            self._thread.join(timeout=10)

    def join(self):
        if self._thread:
            self._thread.join()

    def snapshot(self):
        with self.lock:
            return {**self.state, "rows": list(self.state["rows"])}

    def _run(self, clips, model, cfg, cancel):
        try:
            for name, path in clips:
                if cancel.is_set():
                    break
                try:
                    history = self.runner(path, model, cfg)
                    row = score_clip(name, history) if history else {"clip": name, "result": "ERROR: no frames"}
                except (Exception, SystemExit) as e:
                    row = {"clip": name, "result": f"ERROR: {e}"}
                with self.lock:
                    self.state["rows"].append(row)
                    self.state["done"] += 1
                    self.state["correct"] += row["result"] == "OK"
        finally:
            with self.lock:
                self.state["running"] = False


class Engine:
    """Runs one source at a time in a worker thread; HTTP handlers read its latest frame and status."""

    def __init__(self):
        self.cfg = Config()  # shared with the running detector, so slider changes apply live
        self.cond = threading.Condition()
        self.jpg, self.frame_id = None, 0
        self.status = {"running": False, "label": None, "state": None, "error": None}
        self.series = collections.deque(maxlen=450)  # ~15 s at 30 fps
        self.events = collections.deque(maxlen=100)
        self.snapshots = collections.deque(maxlen=12)  # frames captured at alerts, newest last
        self._snap_id = 0
        self.detector = None
        self._stop, self._thread = threading.Event(), None
        self._control = threading.RLock()  # one start/stop at a time, so only one worker owns the camera

    def start(self, source_id, model="lite"):
        path, crop, label = resolve_source(source_id)
        model_path = resolve_model(model)
        with self._control:
            self.stop()
            with self.cond:
                self.series.clear()
                self.events.clear()
                self.status = {"running": True, "source": source_id, "label": label, "model": model,
                               "state": None, "error": None, "ended": False, "still_for": 0.0, "escalated": False}
            self._stop = threading.Event()
            self._thread = threading.Thread(target=self._run, args=(path, model_path, crop, self._stop), daemon=True)
            self._thread.start()

    def stop(self):
        with self._control:
            if self._thread:
                self._stop.set()
                self._thread.join(timeout=5)
                self._thread = None

    def snapshot(self):
        with self.cond:
            snaps = [{k: v for k, v in s.items() if k != "jpg"} for s in reversed(self.snapshots)]
            return {**self.status, "series": list(self.series), "events": list(self.events)[::-1],
                    "config": dataclasses.asdict(self.cfg), "snapshots": snaps}

    def snapshot_jpg(self, snap_id):
        with self.cond:
            return next((s["jpg"] for s in self.snapshots if s["id"] == snap_id), None)

    def _capture(self, kind, t, jpg):
        """Keep the frame of an alert for the gallery (call with self.cond held)."""
        self._snap_id += 1
        self.snapshots.append({"id": self._snap_id, "kind": kind, "t": round(t, 1), "label": self.status["label"],
                               "clock": time.strftime("%H:%M:%S"), "jpg": jpg})

    def _run(self, path, model_path, crop, stop):
        is_cam, t0, prev_state, fps, last = path.isdigit(), time.monotonic(), None, 0.0, time.monotonic()
        det, was_escalated = FallDetector(self.cfg), False
        self.detector = det
        try:
            with contextlib.closing(process(path, model_path, crop, det)) as stream:
                for t, frame, points, state, f in stream:
                    if stop.is_set():
                        break
                    if not is_cam:  # play files at their real speed
                        time.sleep(max(0.0, t - (time.monotonic() - t0)))
                    draw(frame, points, state, f, hud=False)  # the page shows state and readouts
                    jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
                    now = time.monotonic()
                    fps = 0.9 * fps + 0.1 / max(now - last, 1e-3)
                    last = now
                    feats = dataclasses.asdict(f) if f else None
                    with self.cond:
                        self.jpg, self.frame_id = jpg, self.frame_id + 1
                        self.status.update(state=state, t=round(t, 2), fps=round(fps, 1), features=feats,
                                           still_for=round(det.still_for, 1), escalated=det.escalated)
                        self.series.append({"t": round(t, 3), "state": state, **(feats or {})})
                        if state != prev_state:
                            self.events.append({"t": round(t, 2), "from": prev_state, "to": state,
                                                "clock": time.strftime("%H:%M:%S")})
                            if state == FALL_ALERT:
                                self._capture("Fall alert", t, jpg)
                            prev_state = state
                        if det.escalated and not was_escalated:
                            self.events.append({"t": round(t, 2), "from": FALL_ALERT, "to": "NO_MOVEMENT",
                                                "clock": time.strftime("%H:%M:%S")})
                            self._capture(f"No movement {self.cfg.still_alert:g}s", t, jpg)
                        was_escalated = det.escalated
                        self.cond.notify_all()
        except SystemExit as e:  # frames()/make_landmarker report bad sources this way
            with self.cond:
                self.status["error"] = str(e)
        except Exception as e:  # keep the server alive; show the problem in the page
            with self.cond:
                self.status["error"] = f"{type(e).__name__}: {e}"
        finally:
            with self.cond:
                self.status.update(running=False, ended=not stop.is_set())
                self.cond.notify_all()


class Handler(BaseHTTPRequestHandler):
    engine = None  # set by make_server
    bench = None

    def log_message(self, *args):  # keep the console quiet; errors still surface in the page
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _body(self, limit=1 << 16):
        n = int(self.headers.get("Content-Length") or 0)
        if n > limit:
            raise ValueError("request too large")
        return self.rfile.read(n)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/":
            body = PAGE.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif url.path == "/stream.mjpg":
            self._stream()
        elif url.path == "/api/status":
            self._json({**self.engine.snapshot(), "benchmark": self.bench.snapshot()})
        elif url.path.startswith("/api/snapshot/") and url.path.endswith(".jpg"):
            snap_id = url.path[len("/api/snapshot/"):-len(".jpg")]
            jpg = self.engine.snapshot_jpg(int(snap_id)) if snap_id.isdigit() else None
            if jpg is None:
                self._json({"error": "not found"}, 404)
            else:
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpg)))
                self.end_headers()
                self.wfile.write(jpg)
        elif url.path == "/api/sources":
            self._json(list_sources())
        elif url.path == "/api/results":
            self._json({split: parse_results(p.read_text(encoding="utf-8")) if p.exists() else []
                        for split, p in (("holdout", ROOT / "results_holdout.md"), ("tune", ROOT / "results_tune.md"))})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        url = urlparse(self.path)
        try:
            if url.path == "/api/start":
                req = json.loads(self._body() or b"{}")
                self.engine.start(req.get("source", ""), req.get("model", "lite"))
                self._json({"ok": True})
            elif url.path == "/api/benchmark":
                req = json.loads(self._body() or b"{}")
                split = req.get("split", "holdout")
                with self.engine.cond:
                    self.bench.start(benchmark_clips(split), resolve_model(req.get("model", "lite")),
                                     self.engine.cfg, split)
                self._json({"ok": True})
            elif url.path == "/api/stop":
                self.engine.stop()
                self._json({"ok": True})
            elif url.path == "/api/config":
                with self.engine.cond:
                    update_config(self.engine.cfg, json.loads(self._body() or b"{}"))
                self._json(dataclasses.asdict(self.engine.cfg))
            elif url.path == "/api/config/reset":
                with self.engine.cond:
                    update_config(self.engine.cfg, dataclasses.asdict(Config()))
                self._json(dataclasses.asdict(self.engine.cfg))
            elif url.path == "/api/upload":
                name = safe_upload_name(parse_qs(url.query).get("name", [""])[0])
                data = self._body(limit=MAX_UPLOAD)
                UPLOADS.mkdir(exist_ok=True)
                (UPLOADS / name).write_bytes(data)
                self._json({"ok": True, "id": f"uploads/{name}"})
            else:
                self._json({"error": "not found"}, 404)
        except (ValueError, json.JSONDecodeError) as e:
            self._json({"error": str(e)}, 400)

    def _stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        seen = -1
        try:
            while True:
                with self.engine.cond:
                    self.engine.cond.wait_for(lambda: self.engine.frame_id != seen, timeout=1.0)
                    jpg, fid = self.engine.jpg, self.engine.frame_id
                if jpg is None or fid == seen:
                    continue
                seen = fid
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpg)
                                 + jpg + b"\r\n")
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass  # browser closed the stream


def make_server(port=8000):
    Handler.engine, Handler.bench = Engine(), Benchmark()
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    return server


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args()
    srv = make_server(a.port)
    print(f"Fall detection demo: http://127.0.0.1:{a.port}  (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        Handler.engine.stop()
        srv.server_close()
