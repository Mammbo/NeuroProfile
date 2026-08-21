#Daniel Alvarez
# 8/19/26  (updated 8/21/26 — per-chunk subprocess isolation; audio/text on CPU, video on GPU)
"""
batch_encode.py — headless corpus grinder for NeuroProfile.

Runs a list of URLs / files / text through the full pipeline and lands each one in
Qdrant, unattended. One bad clip is logged and skipped; it never kills the run.

    resolve_source -> chunk_video -> [encode_chunk (ISOLATED SUBPROCESS)]*chunks
      -> stitch -> reduce -> save .npz -> build_payload -> upsert_video

Usage (on Colab, run from repo root; feed FILE PATHS, not URLs — Colab IPs are YouTube-blocked):
    python batch_encoding/batch_encode.py --corpus /content/corpus.txt \
        --qdrant-path   /content/drive/MyDrive/neuroprofile/qdrant_data \
        --timelines-dir /content/drive/MyDrive/neuroprofile/data/timelines
    python batch_encoding/batch_encode.py --corpus /content/corpus.txt --limit 2
    python batch_encoding/batch_encode.py --corpus /content/corpus.txt --overwrite
"""
import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

# --- run from repo root: reducer.py does np.load("ica/...") at import time ----
REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(REPO_ROOT / "batch_encoding"))

import numpy as np

import input_handler as ih
from chunker import chunk_video
from stitcher import stitch
from reducer import reduce
import storage
from chunk_runner import encode_chunk   # spawns _encode_worker.py per chunk (isolated)


# helpers
def _now():
    return datetime.now(timezone.utc).isoformat()

# Cookies FILE (only used for the URL route; on Colab you feed files, so it's unused).
COOKIES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yt_cookies.txt")

def _ytdlp_opts(extra_argv):
    from yt_dlp import parse_options
    argv = ["--js-runtimes", "deno", "--remote-components", "ejs:github"]
    if os.path.exists(COOKIES_FILE):
        argv += ["--cookies", COOKIES_FILE]
    return parse_options(argv + extra_argv).ydl_opts


def _sha(s: str, n: int = 12) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:n]


def _log(logf: Path, record: dict):
    record["ts"] = _now()
    with open(logf, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"[{record.get('status','?'):>7}] {record.get('video_id','')}  "
          f"{record.get('title','')[:60]}"
          + (f"  ({record['error']})" if record.get("error") else ""), flush=True)


#  media in
def download_url(url: str, work: Path, max_bytes: int):
    """Returns (path, title, ytid). 480p cap + merge to mp4. (URL route — local use only.)"""
    import yt_dlp
    ydl_opts = _ytdlp_opts([
        "-f", "bv*[height<=480]+ba/b[height<=480]/b",
        "--merge-output-format", "mp4",
        "-o", str(work / "%(id)s.%(ext)s"),
        "--max-filesize", str(max_bytes),
        "--no-playlist",
    ])
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        path = Path(ydl.prepare_filename(info)).with_suffix(".mp4")
    if not path.exists():
        raise RuntimeError("download produced no file (size cap hit or unavailable format)")
    return str(path), info.get("title") or url, info.get("id")


def resolve_source(source: str, work: Path, max_bytes: int):
    """Normalize one corpus line into (video_id, title, media_kind, (route, payload))."""
    if source.startswith(("http://", "https://")):
        ih.validate_url(source)                      # SSRF + extractor allowlist
        import yt_dlp
        with yt_dlp.YoutubeDL(_ytdlp_opts(["--no-playlist", "--skip-download"])) as ydl:
            info = ydl.extract_info(source, download=False)
        vid = f"url:{info.get('id') or _sha(source)}"
        return vid, (info.get("title") or source), "video", ("__URL__", source)

    p = Path(source)
    if p.exists():
        head = p.open("rb").read(16)
        norm = ih.validate_file(str(p), head, p.stat().st_size)  # raises on bad type/size
        vid = f"file:{p.stem}"
        return vid, p.name, norm.media_kind, ("__FILE__", str(p))

    # not a URL, not a path -> literal text input
    norm = ih.validate_text(source)
    return f"text:{_sha(source)}", norm.title, "text", ("__TEXT__", norm.text)


# encode
def compute_moments(system_ts, timestamps, names, tiers, top_k=8):
    from scipy.signal import find_peaks
    moments = []
    for s in range(system_ts.shape[1]):
        col = system_ts[:, s]
        mask = ~np.isnan(col)
        if mask.sum() < 3:
            continue
        filled = np.where(mask, col, np.nanmin(col[mask]))
        peaks, _ = find_peaks(filled, distance=3)
        for pk in peaks:
            moments.append({
                "t": int(timestamps[pk]),
                "system": names[s],
                "tier": tiers[s],
                "value": float(col[pk]),
            })
    moments.sort(key=lambda m: m["value"], reverse=True)
    return moments[:top_k]


def encode_chunks_isolated(items, diag):
    """Encode each chunk in its own subprocess. Constant VRAM regardless of chunk count."""
    chunk_predictions = []
    for idx, (path, chunk_start) in enumerate(items):
        preds, segments = encode_chunk(path)          # child process; VRAM freed on exit
        if idx == 1 and not diag["printed"] and segments:
            s0 = segments[0].start
            verdict = ("chunk-local: stitcher OK" if abs(s0) < abs(s0 - chunk_start)
                       else "ABSOLUTE: fix stitcher (remove chunk_start)")
            print(f"  >>> segment.start check: chunk1 seg[0].start={s0} "
                  f"chunk_start={chunk_start} => {verdict}", flush=True)
            diag["printed"] = True
        chunk_predictions.append((preds, segments, chunk_start))
    return chunk_predictions


def encode_one(source, db, cfg, diag):
    vid, title, media_kind, (route, payload_src) = resolve_source(
        source, cfg.work_dir, cfg.max_bytes)

    if not cfg.overwrite and db.get_video(vid) is not None:
        return {"status": "skip", "video_id": vid, "title": title, "reason": "exists"}

    if route == "__TEXT__":
        # Text-only input isn't supported in isolated (video-worker) mode. The demo corpus is
        # video files; feed those. (Re-add a text worker mode later if needed.)
        return {"status": "skip", "video_id": vid, "title": title,
                "reason": "text input not supported in isolated batch mode; feed video files"}

    per = cfg.work_dir / _sha(vid)
    per.mkdir(parents=True, exist_ok=True)

    if route == "__URL__":
        media_path, title, _ = download_url(payload_src, per, cfg.max_bytes)
    else:  # __FILE__
        media_path = payload_src

    chunks = chunk_video(media_path, out_dir=str(per / "chunks"))
    items = sorted(chunks.items(), key=lambda kv: kv[1])
    chunk_predictions = encode_chunks_isolated(items, diag)

    timeline, timestamps = stitch(chunk_predictions)
    if timeline.shape[0] == 0:
        return {"status": "empty", "video_id": vid, "title": title,
                "error": "no segments survived"}
    out = reduce(timeline)
    duration = int(timestamps[-1] - timestamps[0] + 1)

    # persist timeline (.npz). Store the BASENAME in the payload so it's relocatable
    # (Colab <-> local); the reader resolves it against its own timelines dir.
    timeline_name = f"{_sha(vid)}.npz"
    timeline_path = cfg.timelines_dir / timeline_name
    np.savez(timeline_path,
             region_ts=out["region_ts"], system_ts=out["system_ts"],
             timestamps=timestamps, valid=out["valid"])

    moments = ([] if cfg.no_moments else
               compute_moments(out["system_ts"], timestamps,
                               out["system_names"], out["system_tiers"]))

    dbpayload = storage.build_payload(
        video_id=vid, title=title, duration=duration,
        system_profile=out["system_profile"].tolist(),
        system_names=out["system_names"], system_tiers=out["system_tiers"],
        system_derived=out["system_derived"],
        moments=moments, timeline_path=timeline_name,
        transcript="",   # captured empty in isolated mode (dashboard doesn't require it)
    )
    db.upsert_video(vid, out["summary_vec"], payload=dbpayload)

    return {"status": "ok", "video_id": vid, "title": title,
            "seconds": int(out["valid"].sum()), "duration": duration,
            "n_moments": len(moments), "chunks": len(items)}


# driver
def load_corpus(args):
    sources = list(args.sources)
    if args.corpus:
        for line in Path(args.corpus).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                sources.append(line)
    if args.limit:
        sources = sources[:args.limit]
    return sources


def main():
    ap = argparse.ArgumentParser(description="Batch-encode a corpus into Qdrant.")
    ap.add_argument("sources", nargs="*", help="URLs / file paths / text")
    ap.add_argument("--corpus", help="file with one source per line (# = comment)")
    ap.add_argument("--limit", type=int, help="only the first N sources")
    ap.add_argument("--overwrite", action="store_true", help="re-encode existing ids")
    ap.add_argument("--no-moments", action="store_true", help="store moments=[]")
    ap.add_argument("--qdrant-path", default="./qdrant_data")
    ap.add_argument("--work-dir", default="./data/work")
    ap.add_argument("--timelines-dir", default="./data/timelines")
    ap.add_argument("--max-filesize-gb", type=float, default=2.0,
                    help="URL download cap (GB)")
    args = ap.parse_args()

    cfg = argparse.Namespace(
        overwrite=args.overwrite, no_moments=args.no_moments,
        work_dir=Path(args.work_dir), timelines_dir=Path(args.timelines_dir),
        max_bytes=int(args.max_filesize_gb * 2**30),
    )
    cfg.work_dir.mkdir(parents=True, exist_ok=True)
    cfg.timelines_dir.mkdir(parents=True, exist_ok=True)
    logf = cfg.timelines_dir.parent / "encode_log.jsonl"

    sources = load_corpus(args)
    if not sources:
        sys.exit("no sources — pass args or --corpus")
    print(f"Corpus: {len(sources)} source(s). Log -> {logf}", flush=True)
    print("Mode: per-chunk subprocess isolation | audio+text=CPU, video=GPU", flush=True)

    db = storage.QdrantVecDB(path=args.qdrant_path)
    diag = {"printed": False}

    ok = fail = 0
    t0 = time.time()
    for i, src in enumerate(sources, 1):
        print(f"\n=== [{i}/{len(sources)}] {src[:80]}", flush=True)
        try:
            rec = encode_one(src, db, cfg, diag)
        except Exception as e:
            # a worker OOM/failure arrives here as CalledProcessError; log and continue
            rec = {"status": "error", "title": src,
                   "error": f"{type(e).__name__}: {e}"[:200]}
            traceback.print_exc()
        rec.setdefault("source", src)
        _log(logf, rec)
        ok += rec["status"] in ("ok", "skip")
        fail += rec["status"] not in ("ok", "skip")

    dt = time.time() - t0
    print(f"\nDone. ok/skip={ok}  failed={fail}  in {dt/60:.1f} min. Details in {logf}",
          flush=True)


if __name__ == "__main__":
    main()