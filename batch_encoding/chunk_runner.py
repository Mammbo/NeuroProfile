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

    tail = []
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        for line in proc.stdout:            # stream + capture the tail
            sys.stdout.write(line)
            sys.stdout.flush()
            tail.append(line.rstrip("\n"))
            tail[:] = tail[-25:]
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        raise RuntimeError(f"chunk encode timed out after {timeout}s")

    if proc.returncode != 0:
        ctx = "\n".join(tail[-12:])
        if proc.returncode in (-9, 137):    # SIGKILL from the OS OOM-killer
            raise RuntimeError(
                "worker was killed by the OS (SIGKILL) — this is almost always out of memory. "
                "Text now runs on GPU by default (fp16), which avoids the ~12 GiB CPU-RAM Llama "
                "load; if it still happens the GPU itself is too small for one chunk. "
                f"Last worker output:\n{ctx}")
        raise RuntimeError(f"chunk encode failed (exit {proc.returncode}).\nLast output:\n{ctx}")

    with np.load(out) as z:
        preds = z["preds"]
        starts = z["starts"]
    try:
        os.remove(out)
    except OSError:
        pass
    return preds, [_Seg(s) for s in starts]