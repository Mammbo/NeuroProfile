"""
chunk_runner.py — run ONE chunk's encode in an isolated subprocess and get its results back.

Public API:
    encode_chunk(chunk_path, text_cpu=False) -> (preds: np.ndarray (n, 20484),
                                                 segments: list[_Seg])  # _Seg has .start
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

WORKER = Path(__file__).resolve().parent / "_encode_worker.py"


class _Seg:
    """Lightweight stand-in for neuralset's Segment — stitch() only reads .start."""
    __slots__ = ("start",)

    def __init__(self, start):
        self.start = float(start)


def encode_chunk(chunk_path, text_cpu=False, timeout=None):
    """Encode one chunk in a fresh process. Returns (preds, [_Seg, ...])."""
    chunk_path = str(chunk_path)
    out = chunk_path + ".preds.npz"
    cmd = [sys.executable, str(WORKER), chunk_path, out]
    if text_cpu:
        cmd.append("--text-cpu")
    subprocess.run(cmd, check=True, timeout=timeout)  # raises CalledProcessError on failure
    with np.load(out) as z:
        preds = z["preds"]
        starts = z["starts"]
    try:
        os.remove(out)
    except OSError:
        pass
    return preds, [_Seg(s) for s in starts]