#!/usr/bin/env python
"""
_encode_worker.py — encode ONE chunk in an isolated process, then EXIT.
Usage:
    python batch_encoding/_encode_worker.py <chunk.mp4> <out.npz> [--text-cpu]
Writes <out.npz> with:  preds (n, 20484) float32,  starts (n,) float32.
Exit code 0 = success.
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# reducer.py does np.load("ica/...") at import time -> run from repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT / "backend"))

import numpy as np
import torch


def _vram(tag):
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"[worker vram {tag}] reserved={torch.cuda.memory_reserved()/2**30:.2f} "
              f"free={free/2**30:.2f} / {total/2**30:.2f} GiB", flush=True)


def build_cfg(text_cpu):
    """All backbones on cuda fp16 (single chunk fits). --text-cpu = safety valve."""
    return {
        "data.text_feature.device": "cpu" if text_cpu else "cuda",
        "data.text_feature.batch_size": 8,
        "data.audio_feature.device": "cuda",
        "data.video_feature.image.device": "cuda",
        "data.video_feature.image.batch_size": 2,
        "data.image_feature.image.device": "cuda",
        "data.image_feature.image.batch_size": 2,
    }


def load_model(cfg):
    """Load the model; auto-drop any config key the checkpoint schema forbids."""
    from tribev2 import TribeModel
    try:
        from pydantic import ValidationError
    except Exception:
        ValidationError = tuple()
    cfg = dict(cfg)
    for _ in range(8):
        try:
            return TribeModel.from_pretrained(
                "facebook/tribev2", device="auto", config_update=dict(cfg))
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
            print(f"  [cfg] schema rejected -> dropped: {dropped}", flush=True)
    raise RuntimeError("could not satisfy the checkpoint config schema after retries")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("chunk", help="path to one chunk .mp4")
    ap.add_argument("out", help="output .npz path")
    ap.add_argument("--text-cpu", action="store_true", help="text backbone on CPU (safety valve)")
    args = ap.parse_args()

    os.environ["NLTK_ALLOW_PROXIED_URLOPEN"] = "1"
    import nltk
    for r in ["punkt_tab", "punkt"]:
        nltk.download(r, quiet=True)

    model = load_model(build_cfg(args.text_cpu))
    _vram("after from_pretrained")

    events = model.get_events_dataframe(video_path=args.chunk)
    preds, segments = model.predict(events)
    _vram("after predict")

    starts = np.array([float(s.start) for s in segments], dtype=np.float32)
    np.savez(args.out, preds=np.asarray(preds, dtype=np.float32), starts=starts)
    print(f"[worker] {args.chunk}: preds={tuple(np.asarray(preds).shape)} kept={len(segments)}",
          flush=True)


if __name__ == "__main__":
    main()