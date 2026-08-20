#!/usr/bin/env python
"""
test_encode.py — the OOM gate for the 8 GB RTX 4060.

CORRECTED strategy (TRIBE is trimodal; the VIDEO backbone is the bottleneck):
  - video backbone (VJEPA2-giant) -> GPU in fp16   <-- the fix. CPU was ~98 s/segment.
  - text  backbone (Llama-3.2-3B) -> CPU            (cheap workload; frees the card)
  - audio backbone (Wav2Vec-BERT) -> CPU
  - brain-encoder head            -> GPU (tiny)

So the GPU holds only VJEPA2 (~4 GB) + the head. Text/audio on CPU are minutes, not
hours. If text-on-CPU eats too much RAM (fp32 Llama ~13 GB), apply the bf16 patch noted
in the handoff.

Config uses only keys that exist in neuralset==0.0.2; any key the checkpoint schema
rejects is auto-dropped and the load retried, so a pydantic `extra_forbidden` can't kill
the run.

Usage:
    python batch_encoding/test_encode.py <path-to-clip.mp4 | url>
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT / "backend"))

import numpy as np
import torch

from chunker import chunk_video
from stitcher import stitch
from reducer import reduce


def _vram(tag):
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 2**30
        resv = torch.cuda.memory_reserved() / 2**30
        free, total = torch.cuda.mem_get_info()
        print(f"[vram {tag}] allocated={alloc:.2f} reserved={resv:.2f} "
              f"free={free/2**30:.2f} / {total/2**30:.2f} GiB")


def preflight():
    if not torch.cuda.is_available():
        print("  preflight: NO CUDA — video would fall back to CPU (hours). Need a GPU.")
        return
    free, total = torch.cuda.mem_get_info()
    print(f"  preflight: GPU OK — {free/2**30:.1f} GiB free / {total/2**30:.1f} GiB total")
    print("  strategy: all backbones on GPU, one resident at a time (gpu_swap)")


def build_cfg():
    """ALL backbones on GPU; gpu_swap keeps only one resident at a time (sequential).
    image_feature.* keys are defensive; load_model() drops whatever the schema rejects."""
    return {
        "data.text_feature.device": "cpu",
        "data.text_feature.batch_size": 8,
        "data.audio_feature.device": "cpu",
        "data.video_feature.image.device": "cuda",
        "data.video_feature.image.batch_size": 1,
        "data.image_feature.image.device": "cuda",
        "data.image_feature.image.batch_size": 1,
    }


def load_model(cfg):
    """Load with sequential GPU swapping; auto-drop any config key the schema forbids."""
    try:
        from gpu_swap import enable_sequential_gpu
        enable_sequential_gpu()
    except Exception as e:
        print(f"[gpu_swap] WARNING not enabled ({e}) — all-cuda may OOM")
    from tribev2 import TribeModel
    try:
        from pydantic import ValidationError
    except Exception:
        ValidationError = tuple()

    cfg = dict(cfg)
    for _ in range(8):
        try:
            return TribeModel.from_pretrained(
                "facebook/tribev2", device="auto", config_update=dict(cfg)
            )
        except ValidationError as e:
            dropped = []
            for err in e.errors():
                if err.get("type") == "extra_forbidden":
                    key = ".".join(str(p) for p in err["loc"])
                    if key in cfg:
                        del cfg[key]
                        dropped.append(key)
            if not dropped:
                raise
            print(f"  [cfg] schema rejected -> dropped: {dropped}")
    raise RuntimeError("could not satisfy the checkpoint config schema after retries")


def download(source, work):
    """480p cap + merge to mp4 — smaller/faster download than 4K."""
    import subprocess
    import glob
    cookies = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yt_cookies.txt")
    cmd = ["yt-dlp", "--js-runtimes", "deno", "--remote-components", "ejs:github",
           "--retries", "infinite", "--fragment-retries", "infinite",
           "--no-continue", "--no-part",
           "--merge-output-format", "mp4",
           "--cookies", cookies,
           "-f", "bv*[height<=480]+ba/b[height<=480]/b",
           "-o", str(work / "dl.%(ext)s"), source]
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

    os.environ["NLTK_ALLOW_PROXIED_URLOPEN"] = "1"
    import nltk
    for r in ["punkt_tab", "punkt"]:
        nltk.download(r, quiet=True)

    print("Loading TribeModel...")
    model = load_model(build_cfg())
    _vram("after from_pretrained")

    work = REPO_ROOT / "data" / "work" / "smoke"
    work.mkdir(parents=True, exist_ok=True)
    video_path = download(source, work) if source.startswith(("http://", "https://")) else source

    print(f"Chunking {video_path} ...")
    chunks = chunk_video(video_path, out_dir=str(work / "chunks"))
    items = sorted(chunks.items(), key=lambda kv: kv[1])
    print(f"  {len(items)} chunk(s): starts = {[s for _, s in items]}")

    chunk_predictions = []
    for idx, (path, chunk_start) in enumerate(items):
        print(f"\n--- chunk {idx}  start={chunk_start}s  {path}")
        events = model.get_events_dataframe(video_path=path)
        preds, segments = model.predict(events)
        _vram(f"after predict chunk {idx}")
        print(f"  preds.shape = {preds.shape}   (expect (n_kept, 20484))")
        print(f"  segments kept = {len(segments)}")

        if idx == 1 and segments:
            s0 = segments[0].start
            verdict = ("chunk-local (stitcher OK)" if abs(s0) < abs(s0 - chunk_start)
                       else "ABSOLUTE — stitcher double-counts, remove chunk_start")
            print(f"  >>> segment.start check: chunk1 seg[0].start = {s0}  "
                  f"vs chunk_start = {chunk_start}  =>  {verdict}")

        chunk_predictions.append((preds, segments, chunk_start))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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

    _vram("final")
    print("\nSMOKE TEST PASSED.")


if __name__ == "__main__":
    main()