#!/usr/bin/env python
"""
Endpoints (all CORS-open):
  POST /analyze        multipart file=<mp4>  -> {"job_id","video_id"}   (encode runs in background)
  GET  /jobs/{id}      -> {"status": queued|running|done|error, "stage": "...", "error": ...}
  GET  /result/{id}    -> full profile (same shape as /api/videos/{id}) once status == done
  GET  /api/videos                      -> list (browse the corpus)
  GET  /api/videos/{video_id}           -> one video + timeline
  GET  /api/videos/{video_id}/similar   -> nearest neighbors
  GET  /health
"""

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(REPO_ROOT / "batch_encoding"))

import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from chunker import chunk_video
from stitcher import stitch
from reducer import reduce
import storage
from chunk_runner import encode_chunk

CFG = {"qdrant_path": "./qdrant_data", "timelines_dir": "./data/timelines",
       "work_dir": "./data/work/analyze", "videos_dir": ""}
VIDEO_EXTS = (".mp4", ".webm", ".mov", ".mkv", ".m4v", ".avi")

app = FastAPI(title="NeuroProfile analyze")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_JOBS: dict[str, dict] = {}
_JOBLOCK = threading.Lock()

# single shared embedded-Qdrant handle (one client per process), guarded by _DBLOCK
_DB = None
_DBLOCK = threading.Lock()


def get_db():
    global _DB
    if _DB is None:
        _DB = storage.QdrantVecDB(path=CFG["qdrant_path"])
    return _DB


def _sha(b: bytes, n: int = 12) -> str:
    return hashlib.sha1(b).hexdigest()[:n]


def _compute_moments(system_ts, timestamps, names, tiers, top_k=8):
    from scipy.signal import find_peaks
    out = []
    for s in range(system_ts.shape[1]):
        col = system_ts[:, s]
        mask = ~np.isnan(col)
        if mask.sum() < 3:
            continue
        filled = np.where(mask, col, np.nanmin(col[mask]))
        peaks, _ = find_peaks(filled, distance=3)
        for pk in peaks:
            out.append({"t": int(timestamps[pk]), "system": names[s],
                        "tier": tiers[s], "value": float(col[pk])})
    out.sort(key=lambda m: m["value"], reverse=True)
    return out[:top_k]


def _timeline_payload(system_ts, timestamps, valid):
    st = np.where(np.isfinite(system_ts), system_ts, 0.0)
    return {"times": [float(t) for t in timestamps],
            "system_ts": st.tolist(),
            "valid": [bool(v) for v in valid]}


def _slim(p):
    return {"video_id": p["video_id"], "title": p.get("title", p["video_id"]),
            "duration": p.get("duration"), "system_names": p.get("system_names"),
            "system_tiers": p.get("system_tiers"), "system_profile": p.get("system_profile")}


def _load_timeline(timeline_path):
    fp = Path(CFG["timelines_dir"]) / os.path.basename(timeline_path)
    if not fp.exists():
        raise HTTPException(404, f"timeline not found: {fp.name}")
    z = np.load(fp)
    system_ts = z["system_ts"].astype(float)
    ts = z["timestamps"].astype(float) if "timestamps" in z else np.arange(system_ts.shape[0], dtype=float)
    valid = z["valid"].astype(bool) if "valid" in z else ~np.isnan(system_ts).any(axis=1)
    return _timeline_payload(system_ts, ts, valid)


class _Canceled(Exception):
    """Raised inside a job when the client asked to cancel."""


def _run_job(job_id: str, mp4_path: Path, title: str, vid: str):
    t0 = time.time()

    def emit(msg: str, stage: str = None):
        """Append a timestamped line to the job log (and stdout), optionally set the stage."""
        line = f"[{time.time() - t0:5.1f}s] {msg}"
        print(f"[job {job_id}] {line}", flush=True)
        with _JOBLOCK:
            j = _JOBS.get(job_id)
            if j is None:
                return
            j.setdefault("log", []).append(line)
            j["log"] = j["log"][-40:]          # keep the tail bounded
            if stage is not None:
                j["stage"] = stage

    def check_cancel():
        with _JOBLOCK:
            if _JOBS.get(job_id, {}).get("cancel"):
                raise _Canceled()

    try:
        with _JOBLOCK:
            _JOBS[job_id]["status"] = "running"
        emit(f"reading upload · {mp4_path.stat().st_size / 1e6:.1f} MB", "reading upload")
        check_cancel()

        emit("splitting into 100s chunks…", "chunking")
        chunks = chunk_video(str(mp4_path), out_dir=str(Path(CFG["work_dir"]) / vid / "chunks"))
        items = sorted(chunks.items(), key=lambda kv: kv[1])
        emit(f"{len(items)} chunk(s) to encode")

        chunk_predictions = []
        for idx, (path, chunk_start) in enumerate(items):
            check_cancel()
            emit(f"encoding chunk {idx + 1}/{len(items)} on GPU (isolated subprocess)…",
                 f"encoding chunk {idx + 1}/{len(items)}")
            ct = time.time()
            preds, segments = encode_chunk(path)      # isolated subprocess; VRAM freed on exit
            emit(f"chunk {idx + 1}/{len(items)} done · {len(segments)} segments · "
                 f"{time.time() - ct:.0f}s")
            chunk_predictions.append((preds, segments, chunk_start))

        check_cancel()
        emit("stitching chunks into one timeline…", "stitching")
        timeline, timestamps = stitch(chunk_predictions)
        if timeline.shape[0] == 0:
            raise RuntimeError("no segments survived")

        emit(f"reducing {timeline.shape[0]} rows → 360 regions → 6 systems…", "reducing")
        out = reduce(timeline)
        duration = int(timestamps[-1] - timestamps[0] + 1)

        timeline_name = f"{vid.replace(':', '_')}.npz"
        Path(CFG["timelines_dir"]).mkdir(parents=True, exist_ok=True)
        np.savez(Path(CFG["timelines_dir"]) / timeline_name,
                 region_ts=out["region_ts"], system_ts=out["system_ts"],
                 timestamps=timestamps, valid=out["valid"])
        emit(f"saved timeline → {timeline_name}")

        moments = _compute_moments(out["system_ts"], timestamps,
                                   out["system_names"], out["system_tiers"])
        payload = storage.build_payload(
            video_id=vid, title=title, duration=duration,
            system_profile=out["system_profile"].tolist(),
            system_names=out["system_names"], system_tiers=out["system_tiers"],
            system_derived=out["system_derived"],
            moments=moments, timeline_path=timeline_name, transcript="",
        )
        emit("storing vector in Qdrant…", "storing")
        with _DBLOCK:
            get_db().upsert_video(vid, out["summary_vec"], payload=payload)

        result = dict(payload)
        result["timeline"] = _timeline_payload(out["system_ts"], timestamps, out["valid"])
        emit(f"done · {duration}s clip · {time.time() - t0:.0f}s total", "done")
        with _JOBLOCK:
            _JOBS[job_id].update(status="done", stage="done", result=result, video_id=vid)
    except _Canceled:
        emit("canceled by user", "canceled")
        with _JOBLOCK:
            _JOBS[job_id].update(status="canceled", stage="canceled")
    except Exception as e:
        traceback.print_exc()
        emit(f"ERROR: {type(e).__name__}: {e}", "error")
        with _JOBLOCK:
            _JOBS[job_id].update(status="error", error=f"{type(e).__name__}: {e}"[:300])


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(400, "empty upload")
    sha = _sha(data)
    vid = f"upload:{sha}"
    work = Path(CFG["work_dir"]) / vid
    work.mkdir(parents=True, exist_ok=True)
    mp4 = work / "input.mp4"
    mp4.write_bytes(data)

    # Persist the ORIGINAL file (with its audio) to the videos dir — on Colab this is a Drive
    # path — so /video/{vid} can serve it back for playback later and after a sync-down, not
    # just from the uploader's ephemeral in-browser blob. Named by the id stem so
    # _resolve_video_file finds it: video_id 'upload:<sha>' -> '<sha><ext>'.
    if CFG.get("videos_dir"):
        try:
            ext = Path(file.filename or "").suffix.lower()
            if ext not in VIDEO_EXTS:
                ext = ".mp4"
            vdir = Path(CFG["videos_dir"])
            vdir.mkdir(parents=True, exist_ok=True)
            dest = vdir / f"{sha}{ext}"
            if not dest.exists():
                dest.write_bytes(data)
            print(f"[analyze] saved playback copy -> {dest}", flush=True)
        except Exception as e:
            print(f"[analyze] WARN could not save playback copy: {e}", flush=True)

    job_id = uuid.uuid4().hex[:12]
    title = file.filename or vid
    with _JOBLOCK:
        _JOBS[job_id] = {"status": "queued", "stage": "queued", "video_id": vid,
                         "title": title, "log": [], "cancel": False}
    threading.Thread(target=_run_job, args=(job_id, mp4, title, vid), daemon=True).start()
    return {"job_id": job_id, "video_id": vid}


@app.post("/jobs/{job_id}/cancel")
def job_cancel(job_id: str):
    """Ask a running job to stop. It aborts at the next checkpoint (between chunks / stages);
    a chunk already on the GPU finishes first, then the job stops."""
    with _JOBLOCK:
        j = _JOBS.get(job_id)
        if not j:
            raise HTTPException(404, "unknown job")
        if j["status"] in ("done", "error", "canceled"):
            return {"job_id": job_id, "status": j["status"], "note": "already finished"}
        j["cancel"] = True
    return {"job_id": job_id, "status": "canceling"}


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    with _JOBLOCK:
        j = _JOBS.get(job_id)
    if not j:
        raise HTTPException(404, "unknown job")
    return {k: v for k, v in j.items() if k != "result"}


@app.get("/result/{job_id}")
def job_result(job_id: str):
    with _JOBLOCK:
        j = _JOBS.get(job_id)
    if not j:
        raise HTTPException(404, "unknown job")
    if j["status"] != "done":
        raise HTTPException(409, f"job not done (status={j['status']})")
    return j["result"]


@app.get("/api/videos")
def api_videos():
    db = get_db()
    out, offset = [], None
    with _DBLOCK:
        while True:
            pts, offset = db.client.scroll(db.videos_v1, limit=256, offset=offset,
                                           with_payload=True, with_vectors=False)
            out.extend(_slim(p.payload) for p in pts)
            if offset is None:
                break
    return out


@app.get("/api/videos/{video_id}")
def api_video(video_id: str):
    db = get_db()
    with _DBLOCK:
        p = db.get_video(video_id)
    if not p:
        raise HTTPException(404, "not found")
    payload = dict(p)
    payload["timeline"] = _load_timeline(payload["timeline_path"])
    return payload


@app.get("/api/videos/{video_id}/similar")
def api_similar(video_id: str, k: int = 6):
    from qdrant_client import models as qm  # noqa: F401
    db = get_db()
    with _DBLOCK:
        recs = db.client.retrieve(db.videos_v1, ids=[storage._pid(video_id)], with_vectors=True)
        if not recs:
            raise HTTPException(404, "not found")
        vec = recs[0].vector
        hits = db.search_similar(vec, limit=k, exclude_id=video_id)
    return [{"score": round(float(h["score"]), 3), **_slim(h["payload"])} for h in hits]


def _resolve_video_file(video_id: str):
    """Map a video_id ('file:<stem>' or 'upload:<sha>') to a local mp4 in videos_dir, for
    playback. Returns None (-> 404) when no dir/file is configured."""
    d = CFG.get("videos_dir")
    if not d or not Path(d).exists():
        return None
    stem = video_id.split(":", 1)[1] if ":" in video_id else video_id
    for ext in VIDEO_EXTS:
        p = Path(d) / f"{stem}{ext}"
        if p.exists():
            return p
    for p in Path(d).glob(f"{stem}.*"):
        if p.suffix.lower() in VIDEO_EXTS and not p.name.endswith(".fs.mp4"):
            return p
    return None


def _faststart_file(fp: Path) -> Path:
    """Return a streamable (faststart) copy of fp, remuxing + caching on first use.

    iPhone/most mp4s put the `moov` atom at the END of the file. That plays fine from disk
    (whole file in memory) but over HTTP range the browser can't line up the audio track and
    silently drops it — video plays, no sound. `-movflags +faststart -c copy` moves the index
    to the front (fast, no re-encode) and fixes streaming. Cached next to the source, keyed by
    mtime, so the remux happens once. Falls back to the original if ffmpeg is missing or fails.
    webm/mkv stream fine and are returned as-is."""
    try:
        if fp.suffix.lower() not in (".mp4", ".m4v", ".mov"):
            return fp
        cache = fp.with_name(fp.name + ".fs.mp4")
        if cache.exists() and cache.stat().st_mtime >= fp.stat().st_mtime:
            return cache
        if shutil.which("ffmpeg") is None:
            return fp
        tmp = fp.with_name(fp.name + ".fs.tmp.mp4")
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", str(fp), "-c", "copy", "-movflags", "+faststart", str(tmp)],
            capture_output=True)
        if r.returncode == 0 and tmp.exists():
            tmp.replace(cache)
            print(f"[video] faststart-remuxed {fp.name} -> {cache.name}", flush=True)
            return cache
        if tmp.exists():
            tmp.unlink()
        return fp
    except Exception as e:
        print(f"[video] faststart remux failed for {fp}: {e}", flush=True)
        return fp


@app.get("/video/{video_id}")
def api_video_file(video_id: str):
    fp = _resolve_video_file(video_id)
    if not fp:
        raise HTTPException(404, "no local video file for this id")
    return FileResponse(_faststart_file(fp), media_type="video/mp4")


@app.delete("/api/videos/{video_id}")
def api_delete(video_id: str):
    """Remove a clip everywhere: the Qdrant vector, its timeline .npz, and the video file
    (+ faststart cache). Idempotent-ish: 404 only if the vector isn't in the DB."""
    db = get_db()
    with _DBLOCK:
        p = db.get_video(video_id)
        if not p:
            raise HTTPException(404, "not found")
        db.delete_video(video_id)
    # timeline .npz
    tp = p.get("timeline_path")
    if tp:
        try:
            (Path(CFG["timelines_dir"]) / os.path.basename(tp)).unlink(missing_ok=True)
        except Exception as e:
            print(f"[delete] timeline: {e}", flush=True)
    # video file + its faststart cache
    fp = _resolve_video_file(video_id)
    if fp:
        for f in (fp, fp.with_name(fp.name + ".fs.mp4")):
            try:
                f.unlink(missing_ok=True)
            except Exception as e:
                print(f"[delete] video: {e}", flush=True)
    print(f"[delete] removed {video_id}", flush=True)
    return {"deleted": video_id}


@app.get("/health")
def health():
    return {"ok": True, "qdrant": CFG["qdrant_path"], "jobs": len(_JOBS)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--qdrant-path", default="./qdrant_data")
    ap.add_argument("--timelines-dir", default="./data/timelines")
    ap.add_argument("--work-dir", default="./data/work/analyze")
    ap.add_argument("--videos-dir", default="", help="folder with corpus mp4s (synced playback)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    CFG.update(qdrant_path=a.qdrant_path, timelines_dir=a.timelines_dir, work_dir=a.work_dir,
               videos_dir=a.videos_dir)
    Path(CFG["work_dir"]).mkdir(parents=True, exist_ok=True)
    get_db()  # open the single shared client up front (fail fast if the path is bad)
    print(f"[analyze] qdrant={CFG['qdrant_path']} timelines={CFG['timelines_dir']}")
    import uvicorn
    uvicorn.run(app, host=a.host, port=a.port)