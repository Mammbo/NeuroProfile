"""
chunker_test.py — 100 s windows, 10 s overlap.

Hermetic: every input is an ffmpeg `testsrc` clip generated into a tmpdir, so nothing here
depends on the corpus or on a GPU. Tiny frame size / low fps keeps the re-encode fast.

The thing that actually matters downstream is the *absolute start time* attached to each
chunk: the stitcher adds it to `segment.start` to place a row on the global clock
(constraint #5 — never align by array position). So these tests assert the start-time map,
not just the file count.
"""
import os
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
from chunker import chunk_video  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                                reason="chunker shells out to ffmpeg/ffprobe")


def _make_clip(path, seconds):
    """A `seconds`-long silent test pattern. 64x64 @ 5 fps so encoding stays cheap."""
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", f"testsrc=size=64x64:rate=5:duration={seconds}",
         "-pix_fmt", "yuv420p", str(path)],
        check=True)
    return str(path)


def _duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def test_short_clip_is_one_chunk_at_zero(tmp_path):
    """A clip shorter than one window must not be split, and must start at 0."""
    src = _make_clip(tmp_path / "short.mp4", 30)
    chunks = chunk_video(src, out_dir=str(tmp_path / "chunks"))

    assert len(chunks) == 1
    assert list(chunks.values()) == [0]


def test_150s_clip_splits_at_the_overlap_step(tmp_path):
    """150 s → two windows. The step is window - overlap = 90, so the second chunk's
    absolute start is 90 (NOT 100) — that 10 s of overlap is what lets the stitcher
    average across a seam."""
    src = _make_clip(tmp_path / "long.mp4", 150)
    chunks = chunk_video(src, out_dir=str(tmp_path / "chunks"))

    starts = sorted(chunks.values())
    assert len(chunks) == 2
    assert starts == [0, 90]


def test_chunk_files_exist_and_are_readable(tmp_path):
    """Every returned path must be a real, probe-able video — chunk_runner feeds these
    straight to the model in a subprocess, where a silently-empty file is a bad failure."""
    src = _make_clip(tmp_path / "long.mp4", 150)
    chunks = chunk_video(src, out_dir=str(tmp_path / "chunks"))

    for path, start in sorted(chunks.items(), key=lambda kv: kv[1]):
        assert os.path.exists(path), path
        assert os.path.getsize(path) > 0, path
        d = _duration(path)
        # first window is a full 100 s; the last is the 60 s remainder (150 - 90).
        expected = 100 if start == 0 else 60
        assert d == pytest.approx(expected, abs=1.0), f"{path}: {d}s, wanted ~{expected}s"


def test_windows_cover_the_whole_clip(tmp_path):
    """No gap: consecutive starts advance by exactly the step, and the final window
    reaches the end of the source."""
    src = _make_clip(tmp_path / "long.mp4", 250)
    chunks = chunk_video(src, out_dir=str(tmp_path / "chunks"))

    starts = sorted(chunks.values())
    assert starts == [0, 90, 180]
    assert starts[-1] + _duration(
        [p for p, s in chunks.items() if s == starts[-1]][0]) == pytest.approx(250, abs=1.0)


def test_custom_window_and_overlap(tmp_path):
    """The 100/10 default is just a default — the caller can retune it. 110 s at 40/10
    gives starts 0/30/60/90; the durations are deliberately off the window boundaries so a
    ffprobe duration that reads 110.04 instead of 110.00 can't flip the chunk count."""
    src = _make_clip(tmp_path / "long.mp4", 110)
    chunks = chunk_video(src, window=40, overlap=10, out_dir=str(tmp_path / "chunks"))

    assert sorted(chunks.values()) == [0, 30, 60, 90]


def test_webm_input_chunks(tmp_path):
    """The Chrome extension uploads MediaRecorder webm (vp8/opus), not mp4. The backend
    writes those bytes to a file called input.mp4, but chunk_video is ffmpeg-based and reads
    by content, so the container mismatch must not matter. Chunks come out as mp4."""
    src = tmp_path / "capture.webm"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", "testsrc=size=64x64:rate=5:duration=120",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=120",
         "-c:v", "libvpx", "-b:v", "200k", "-c:a", "libopus", "-shortest", str(src)],
        check=True)

    chunks = chunk_video(str(src), out_dir=str(tmp_path / "chunks"))
    assert sorted(chunks.values()) == [0, 90]
    for path in chunks:
        assert os.path.getsize(path) > 0
        assert _duration(path) > 0


def test_mislabeled_extension_still_chunks(tmp_path):
    """analyze_server writes every upload to 'input.mp4' whatever it really is. Prove that
    a webm wearing an .mp4 name goes through — this is the exact extension upload path."""
    real_webm = tmp_path / "capture.webm"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", "testsrc=size=64x64:rate=5:duration=30",
         "-c:v", "libvpx", "-b:v", "200k", str(real_webm)], check=True)
    lying = tmp_path / "input.mp4"
    lying.write_bytes(real_webm.read_bytes())

    chunks = chunk_video(str(lying), out_dir=str(tmp_path / "chunks"))
    assert list(chunks.values()) == [0]
    assert os.path.getsize(list(chunks)[0]) > 0
