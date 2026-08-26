"""
serving_test.py — the media path in backend/serving.py.

These are the bits that decide whether a clip actually plays in a browser, which is not
something the Qdrant/reducer tests can catch.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
from serving import faststart_file, probe_codecs, media_type_for  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                                reason="the media path shells out to ffmpeg/ffprobe")


def _clip(path, vcodec="libx264", acodec="aac", seconds=2):
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", f"testsrc=size=64x64:rate=10:duration={seconds}",
         "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
         "-c:v", vcodec, "-pix_fmt", "yuv420p", "-c:a", acodec, "-shortest", str(path)],
        check=True)
    return Path(path)


def test_probe_codecs_reads_both_streams(tmp_path):
    fp = _clip(tmp_path / "a.mp4")
    assert probe_codecs(fp) == ("h264", "aac")


def test_probe_codecs_on_a_non_media_file_is_none(tmp_path):
    junk = tmp_path / "not-a-video.mp4"
    junk.write_bytes(b"nope")
    assert probe_codecs(junk) == (None, None)


def test_web_safe_source_is_copied_not_reencoded(tmp_path):
    """h264 + aac is already playable everywhere — the remux must only move the moov atom,
    never spend time re-encoding."""
    fp = _clip(tmp_path / "safe.mp4")
    out = faststart_file(fp)

    assert out.name == "safe.mp4.fs.mp4"
    assert probe_codecs(out) == ("h264", "aac")


def test_opus_audio_is_transcoded_to_aac(tmp_path):
    """yt-dlp's best mp4 is Opus-in-mp4. Safari can't decode it and some Chrome builds play
    the video with no sound, so the cached copy has to carry aac."""
    fp = _clip(tmp_path / "opus.mp4", acodec="libopus")
    assert probe_codecs(fp)[1] == "opus"

    out = faststart_file(fp)
    vcodec, acodec = probe_codecs(out)
    assert acodec == "aac", "opus survived into the served copy — Safari will be silent"
    assert vcodec == "h264"


def test_result_is_cached_and_reused(tmp_path):
    fp = _clip(tmp_path / "safe.mp4")
    first = faststart_file(fp)
    stamp = first.stat().st_mtime_ns
    assert faststart_file(fp) == first
    assert first.stat().st_mtime_ns == stamp, "cache was rebuilt on the second call"


def test_cache_is_rebuilt_when_the_source_changes(tmp_path):
    fp = _clip(tmp_path / "safe.mp4")
    cached = faststart_file(fp)
    size_before = cached.stat().st_size

    _clip(fp, seconds=4)                         # same path, longer source
    os.utime(fp, None)
    rebuilt = faststart_file(fp)
    assert rebuilt.stat().st_size != size_before


def test_webm_passes_through_untouched(tmp_path):
    """webm already streams fine; remuxing it into mp4 would break the extension's captures."""
    fp = tmp_path / "capture.webm"
    fp.write_bytes(b"\x1a\x45\xdf\xa3whatever")
    assert faststart_file(fp) == fp
    assert not (tmp_path / "capture.webm.fs.mp4").exists()


def test_a_broken_file_falls_back_to_the_original(tmp_path):
    fp = tmp_path / "broken.mp4"
    fp.write_bytes(b"not really an mp4")
    assert faststart_file(fp) == fp


@pytest.mark.parametrize("name,expected", [
    ("a.mp4", "video/mp4"), ("a.m4v", "video/mp4"), ("a.webm", "video/webm"),
    ("a.mkv", "video/x-matroska"), ("a.mov", "video/quicktime"),
])
def test_media_type_for(name, expected):
    assert media_type_for(Path(name)) == expected
