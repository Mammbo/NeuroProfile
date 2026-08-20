#!/usr/bin/env python
"""
test_encode.py — encode smoke test, with PER-CHUNK PROCESS ISOLATION.
Usage (from repo root):
    python batch_encoding/test_encode.py <clip.mp4 | url>
Use a clip > 100 s so the chunker makes >=2 chunks and the segment.start check prints.
Env: NP_TEXT_CPU=1 forces the text backbone to CPU inside each worker (safety valve only).
"""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(REPO_ROOT / "batch_encoding"))

import numpy as np
import torch

from chunker import chunk_video
from stitcher import stitch
from reducer import reduce
from chunk_runner import encode_chunk

TEXT_CPU = bool(os.environ.get("NP_TEXT_CPU"))


def preflight():
    if not torch.cuda.is_available():
        print("  preflight: NO CUDA — encode would fall back to CPU (hours). Need a GPU.")
        return
    free, total = torch.cuda.mem_get_info()
    print(f"  preflight: GPU OK — {free/2**30:.1f} GiB free / {total/2**30:.1f} GiB total")
    print("  strategy: per-chunk subprocess isolation — constant VRAM regardless of chunk count")


def download(source, work):
    """480p cap + merge to mp4 (only used if a URL is passed; on Colab feed a FILE)."""
    import glob
    import subprocess
    cookies = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yt_cookies.txt")
    cmd = ["yt-dlp", "--js-runtimes", "deno", "--remote-components", "ejs:github",
           "--retries", "infinite", "--fragment-retries", "infinite",
           "--no-continue", "--no-part", "--merge-output-format", "mp4"]
    if os.path.exists(cookies):
        cmd += ["--cookies", cookies]
    cmd += ["-f", "bv*[height<=480]+ba/b[height<=480]/b", "-o", str(work / "dl.%(ext)s"), source]
    print("RUNNING:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    hits = glob.glob(str(work / "dl.*"))
    if not hits:
        sys.exit("download produced no file")
    return hits[0]


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python batch_encoding/test_encode.py <clip.mp4 | url>")
    source = sys.argv[1]

    preflight()

    work = REPO_ROOT / "data" / "work" / "smoke"
    work.mkdir(parents=True, exist_ok=True)
    video_path = download(source, work) if source.startswith(("http://", "https://")) else source

    print(f"Chunking {video_path} ...")
    chunks = chunk_video(video_path, out_dir=str(work / "chunks"))
    items = sorted(chunks.items(), key=lambda kv: kv[1])
    print(f"  {len(items)} chunk(s): starts = {[s for _, s in items]}")

    chunk_predictions = []
    for idx, (path, chunk_start) in enumerate(items):
        print(f"\n--- chunk {idx}  start={chunk_start}s  {path}  (isolated subprocess)")
        preds, segments = encode_chunk(path, text_cpu=TEXT_CPU)   # child process; VRAM freed on exit
        print(f"  preds.shape = {preds.shape}   (expect (n_kept, 20484))")
        print(f"  segments kept = {len(segments)}")

        if idx == 1 and segments:
            s0 = segments[0].start
            verdict = ("chunk-local (stitcher OK)" if abs(s0) < abs(s0 - chunk_start)
                       else "ABSOLUTE — stitcher double-counts, remove chunk_start")
            print(f"  >>> segment.start check: chunk1 seg[0].start = {s0}  "
                  f"vs chunk_start = {chunk_start}  =>  {verdict}")

        chunk_predictions.append((preds, segments, chunk_start))

    print("\nStitching + reducing...")
    timeline, timestamps = stitch(chunk_predictions)
    print(f"  timeline = {timeline.shape}   timestamps = {timestamps.shape}")
    if timeline.shape[0] == 0:
        sys.exit("EMPTY timeline — no segments survived. Investigate before batching.")
    out = reduce(timeline)
    print(f"  region_ts={out['region_ts'].shape}  system_ts={out['system_ts'].shape}  "
          f"summary_vec={out['summary_vec'].shape}")
    print(f"  system_profile = {np.round(out['system_profile'], 3).tolist()}")
    print(f"  systems: {out['system_names']}")

    print("\nSMOKE TEST PASSED.")


if __name__ == "__main__":
    main()