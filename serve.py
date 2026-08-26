#!/usr/bin/env python
"""
serve.py — the OFFLINE read-only API for the NeuroProfile dashboard.

No GPU, no torch, no tribev2: it just reads a corpus that was already encoded (Qdrant
vectors + timeline .npz on disk) and serves it to the Next.js dashboard. This is the safe
demo path. For live uploads run batch_encoding/analyze_server.py on a GPU box instead —
it speaks the same read API plus /analyze.

    python serve.py --qdrant-path ./qdrant_data \
                    --timelines-dir ./data/timelines \
                    --videos-dir ./data/videos            # :8000

Endpoints (all CORS-open):
  GET    /api/videos                     list the corpus
  GET    /api/videos/{id}                one clip + its timeline
  GET    /api/videos/{id}/similar        nearest neighbours by cosine
  DELETE /api/videos/{id}                drop the vector + .npz + video file
  GET    /video/{id}                     the original media, faststart-remuxed for streaming
  GET    /health
"""
import argparse
import os
import sys
from pathlib import Path

# reducer.py (imported transitively by nothing here, but storage's default paths are
# relative) plus every other entry point runs from the repo root. Keep the preamble.
REPO_ROOT = Path(__file__).resolve().parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT / "backend"))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

import storage
from serving import (VIDEO_EXTS, faststart_file, load_timeline, media_type_for,
                     probe_codecs, resolve_video_file, slim)

CFG = {"qdrant_path": "./qdrant_data", "timelines_dir": "./data/timelines", "videos_dir": ""}

app = FastAPI(title="NeuroProfile serve (offline)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_DB = None


def get_db():
    """One embedded-Qdrant client per process — the embedded store holds a single-process
    lock, so analyze_server.py and serve.py cannot share a qdrant_data/ dir at once."""
    global _DB
    if _DB is None:
        _DB = storage.QdrantVecDB(path=CFG["qdrant_path"])
    return _DB


@app.get("/api/videos")
def api_videos():
    db = get_db()
    out, offset = [], None
    while True:
        pts, offset = db.client.scroll(db.videos_v1, limit=256, offset=offset,
                                       with_payload=True, with_vectors=False)
        out.extend(slim(p.payload) for p in pts)
        if offset is None:
            break
    out.sort(key=lambda v: (v.get("title") or "").lower())
    return out


@app.get("/api/videos/{video_id}")
def api_video(video_id: str):
    p = get_db().get_video(video_id)
    if not p:
        raise HTTPException(404, "not found")
    payload = dict(p)
    try:
        payload["timeline"] = load_timeline(CFG["timelines_dir"], payload["timeline_path"])
    except FileNotFoundError as e:
        raise HTTPException(404, f"timeline not found: {e}")
    return payload


@app.get("/api/videos/{video_id}/similar")
def api_similar(video_id: str, k: int = 6):
    db = get_db()
    recs = db.client.retrieve(db.videos_v1, ids=[storage._pid(video_id)], with_vectors=True)
    if not recs:
        raise HTTPException(404, "not found")
    hits = db.search_similar(recs[0].vector, limit=k, exclude_id=video_id)
    return [{"score": round(float(h["score"]), 3), **slim(h["payload"])} for h in hits]


@app.get("/video/{video_id}")
def api_video_file(video_id: str):
    fp = resolve_video_file(CFG["videos_dir"], video_id)
    if not fp:
        raise HTTPException(404, "no local video file for this id")
    served = faststart_file(fp)
    return FileResponse(served, media_type=media_type_for(served))


@app.delete("/api/videos/{video_id}")
def api_delete(video_id: str):
    """Remove a clip everywhere: Qdrant vector, timeline .npz, media file (+ faststart
    cache). 404 only if the vector isn't in the DB."""
    db = get_db()
    p = db.get_video(video_id)
    if not p:
        raise HTTPException(404, "not found")
    db.delete_video(video_id)
    tp = p.get("timeline_path")
    if tp:
        try:
            (Path(CFG["timelines_dir"]) / os.path.basename(tp)).unlink(missing_ok=True)
        except OSError as e:
            print(f"[delete] timeline: {e}", flush=True)
    fp = resolve_video_file(CFG["videos_dir"], video_id)
    if fp:
        for f in (fp, fp.with_name(fp.name + ".fs.mp4")):
            try:
                f.unlink(missing_ok=True)
            except OSError as e:
                print(f"[delete] video: {e}", flush=True)
    print(f"[delete] removed {video_id}", flush=True)
    return {"deleted": video_id}


@app.get("/health")
def health():
    return {"ok": True, "mode": "offline", "qdrant": CFG["qdrant_path"]}


def prewarm_videos():
    """Build the streamable copy of every media file up front.

    Without this the first click on an AV1/Opus clip blocks for however long the transcode
    takes. Do it at startup instead, so the demo never stalls mid-presentation."""
    d = CFG["videos_dir"]
    if not d or not Path(d).exists():
        print("[prewarm] no --videos-dir, nothing to do")
        return
    files = [p for p in sorted(Path(d).iterdir())
             if p.suffix.lower() in VIDEO_EXTS and not p.name.endswith(".fs.mp4")]
    for p in files:
        v, a = probe_codecs(p)
        out = faststart_file(p)
        print(f"[prewarm] {p.name}  ({v}/{a}) -> {out.name}")
    print(f"[prewarm] {len(files)} file(s) ready")


def check_videos():
    """Print, for every clip in the store, whether a playable media file resolves for it.

    A clip encoded as video_id 'file:<stem>' only plays if '<stem>.<ext>' is sitting in
    --videos-dir. This is the fastest way to see which of the corpus is silent-in-the-UI
    after a Drive sync-down. Returns the number of clips with no file."""
    db = get_db()
    rows, offset, missing = [], None, 0
    while True:
        pts, offset = db.client.scroll(db.videos_v1, limit=256, offset=offset,
                                       with_payload=True, with_vectors=False)
        rows.extend(p.payload for p in pts)
        if offset is None:
            break
    if not rows:
        print("[check] the store is empty — nothing encoded at "
              f"{CFG['qdrant_path']}")
        return 0
    print(f"[check] {len(rows)} clip(s) · videos_dir={CFG['videos_dir'] or '(unset!)'}")
    for p in sorted(rows, key=lambda r: r["video_id"]):
        vid = p["video_id"]
        fp = resolve_video_file(CFG["videos_dir"], vid)
        tl = Path(CFG["timelines_dir"]) / os.path.basename(p.get("timeline_path") or "")
        marks = "video OK " if fp else "NO VIDEO "
        marks += "timeline OK" if tl.name and tl.exists() else "NO TIMELINE"
        if not fp:
            missing += 1
        print(f"  [{marks}] {vid}")
        if not fp:
            stem = vid.split(":", 1)[1] if ":" in vid else vid
            print(f"      put the original at: {CFG['videos_dir'] or '<--videos-dir>'}/{stem}.mp4")
    print(f"[check] {missing} clip(s) have no playable file" if missing
          else "[check] every clip has a playable file")
    return missing


def seed_mock(n=4):
    """Write n obviously-fake clips into the configured store so the dashboard has
    something to render without a GPU. Titles are prefixed MOCK so nobody can mistake
    synthetic noise for a real prediction (constraint #8)."""
    import numpy as np
    sys.path.insert(0, str(REPO_ROOT / "backend"))
    from reducer import system_names, system_tiers, system_derived
    from serving import compute_moments

    rng = np.random.default_rng(0)
    Path(CFG["timelines_dir"]).mkdir(parents=True, exist_ok=True)
    db = get_db()
    for i in range(n):
        vid = f"mock:clip_{i:02d}"
        T = 60 + 30 * i
        ts = np.arange(T, dtype=float)
        system_ts = np.cumsum(rng.normal(0, 0.3, size=(T, 6)), axis=0)
        system_ts[T // 3:T // 3 + 4] = np.nan          # a dropped-segment gap, like the real thing
        region_ts = np.repeat(system_ts, 60, axis=1)
        valid = ~np.isnan(system_ts).all(axis=1)
        name = f"mock_clip_{i:02d}.npz"
        np.savez(Path(CFG["timelines_dir"]) / name, region_ts=region_ts,
                 system_ts=system_ts, timestamps=ts, valid=valid)
        payload = storage.build_payload(
            video_id=vid, title=f"MOCK — synthetic clip {i:02d}", duration=T,
            system_profile=np.nanmean(system_ts, axis=0).tolist(),
            system_names=system_names, system_tiers=system_tiers,
            system_derived=system_derived,
            moments=compute_moments(system_ts, ts, system_names, system_tiers),
            timeline_path=name, transcript="")
        db.upsert_video(vid, np.nanmean(region_ts, axis=0), payload=payload)
        print(f"[mock] seeded {vid} ({T}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--qdrant-path", default="./qdrant_data")
    ap.add_argument("--timelines-dir", default="./data/timelines")
    ap.add_argument("--videos-dir", default="", help="folder with the corpus media files")
    ap.add_argument("--prewarm", action="store_true",
                    help="build every streamable video copy at startup instead of on first play")
    ap.add_argument("--check", action="store_true",
                    help="report which stored clips have a playable file, then exit")
    ap.add_argument("--mock", action="store_true",
                    help="seed synthetic clips first, so the UI has something without a GPU")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    CFG.update(qdrant_path=a.qdrant_path, timelines_dir=a.timelines_dir, videos_dir=a.videos_dir)
    get_db()  # fail fast on a bad path
    if a.check:
        sys.exit(0 if check_videos() == 0 else 1)
    if a.mock:
        seed_mock()
    if a.prewarm:
        prewarm_videos()
    print(f"[serve] offline · qdrant={CFG['qdrant_path']} timelines={CFG['timelines_dir']} "
          f"videos={CFG['videos_dir'] or '(none)'}")
    import uvicorn
    uvicorn.run(app, host=a.host, port=a.port)
