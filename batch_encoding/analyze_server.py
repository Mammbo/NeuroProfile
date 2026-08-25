#!/usr/bin/env python
"""
analyze_server.py — LIVE backend for NeuroProfile, runs INSIDE Colab (GPU), exposed via a
cloudflared tunnel. Accepts an uploaded mp4, encodes it with the same isolated per-chunk
pipeline as batch_encode, stores it in Qdrant, and serves the result.

This is the compute layer the dashboard upload form and the Chrome extension both POST to.

Endpoints (all CORS-open):
  POST /analyze        multipart file=<mp4>  -> {"job_id","video_id"}   (encode runs in background)
  GET  /jobs/{id}      -> {"status": queued|running|done|error, "stage": "...", "error": ...}
  GET  /result/{id}    -> full profile (same shape as /api/videos/{id}) once status == done
  GET  /api/videos                      -> list (browse the corpus)
  GET  /api/videos/{video_id}           -> one video + timeline
  GET  /api/videos/{video_id}/similar   -> nearest neighbors
  GET  /health

Run (inside Colab, from repo root, with HF_TOKEN + HF_HOME set):
  python batch_encoding/analyze_server.py \
      --qdrant-path   /content/drive/MyDrive/neuroprofile/qdrant_data \
      --timelines-dir /content/drive/MyDrive/neuroprofile/data/timelines
Then expose port 8000 with cloudflared (see the notebook cell).

NOTE: embedded Qdrant allows ONE client per process, so this holds a single shared QdrantVecDB
guarded by a lock. Do NOT also run batch_encode against the same qdrant_path at the same time.
"""
import argparse
import hashlib
import os
import sys
import threading
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

from chunker import chunk_video
from stitcher import stitch
from reducer import reduce
import storage
from chunk_runner import encode_chunk

CFG = {"qdrant_path": "./qdrant_data", "timelines_dir": "./data/timelines",
       "work_dir": "./data/work/analyze"}

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


def _run_job(job_id: str, mp4_path: Path, title: str, vid: str):
    def stage(s):
        with _JOBLOCK:
            _JOBS[job_id]["stage"] = s
    try:
        with _JOBLOCK:
            _JOBS[job_id]["status"] = "running"
        stage("chunking")
        chunks = chunk_video(str(mp4_path), out_dir=str(Path(CFG["work_dir"]) / vid / "chunks"))
        items = sorted(chunks.items(), key=lambda kv: kv[1])

        chunk_predictions = []
        for idx, (path, chunk_start) in enumerate(items):
            stage(f"encoding chunk {idx + 1}/{len(items)}")
            preds, segments = encode_chunk(path)      # isolated subprocess; VRAM freed on exit
            chunk_predictions.append((preds, segments, chunk_start))

        stage("reducing")
        timeline, timestamps = stitch(chunk_predictions)
        if timeline.shape[0] == 0:
            raise RuntimeError("no segments survived")
        out = reduce(timeline)
        duration = int(timestamps[-1] - timestamps[0] + 1)

        timeline_name = f"{vid.replace(':', '_')}.npz"
        Path(CFG["timelines_dir"]).mkdir(parents=True, exist_ok=True)
        np.savez(Path(CFG["timelines_dir"]) / timeline_name,
                 region_ts=out["region_ts"], system_ts=out["system_ts"],
                 timestamps=timestamps, valid=out["valid"])

        moments = _compute_moments(out["system_ts"], timestamps,
                                   out["system_names"], out["system_tiers"])
        payload = storage.build_payload(
            video_id=vid, title=title, duration=duration,
            system_profile=out["system_profile"].tolist(),
            system_names=out["system_names"], system_tiers=out["system_tiers"],
            system_derived=out["system_derived"],
            moments=moments, timeline_path=timeline_name, transcript="",
        )
        stage("storing")
        with _DBLOCK:
            get_db().upsert_video(vid, out["summary_vec"], payload=payload)

        result = dict(payload)
        result["timeline"] = _timeline_payload(out["system_ts"], timestamps, out["valid"])
        with _JOBLOCK:
            _JOBS[job_id].update(status="done", stage="done", result=result, video_id=vid)
    except Exception as e:
        traceback.print_exc()
        with _JOBLOCK:
            _JOBS[job_id].update(status="error", error=f"{type(e).__name__}: {e}"[:300])


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(400, "empty upload")
    vid = f"upload:{_sha(data)}"
    work = Path(CFG["work_dir"]) / vid
    work.mkdir(parents=True, exist_ok=True)
    mp4 = work / "input.mp4"
    mp4.write_bytes(data)

    job_id = uuid.uuid4().hex[:12]
    title = file.filename or vid
    with _JOBLOCK:
        _JOBS[job_id] = {"status": "queued", "stage": "queued", "video_id": vid, "title": title}
    threading.Thread(target=_run_job, args=(job_id, mp4, title, vid), daemon=True).start()
    return {"job_id": job_id, "video_id": vid}


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


@app.get("/health")
def health():
    return {"ok": True, "qdrant": CFG["qdrant_path"], "jobs": len(_JOBS)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--qdrant-path", default="./qdrant_data")
    ap.add_argument("--timelines-dir", default="./data/timelines")
    ap.add_argument("--work-dir", default="./data/work/analyze")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    CFG.update(qdrant_path=a.qdrant_path, timelines_dir=a.timelines_dir, work_dir=a.work_dir)
    Path(CFG["work_dir"]).mkdir(parents=True, exist_ok=True)
    get_db()  # open the single shared client up front (fail fast if the path is bad)
    print(f"[analyze] qdrant={CFG['qdrant_path']} timelines={CFG['timelines_dir']}")
    import uvicorn
    uvicorn.run(app, host=a.host, port=a.port)
