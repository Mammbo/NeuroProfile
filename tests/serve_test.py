"""
serve_test.py — the offline read-only API (serve.py).

Everything here runs against a throwaway corpus built in a tmpdir: a real embedded-Qdrant
store, a real timeline .npz, and a real (tiny) mp4. No GPU, no model — that is the whole
point of the offline mode.
"""
import importlib.util
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest
from fastapi.testclient import TestClient

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "backend"))
import storage  # noqa: E402


def _load_serve():
    """serve.py lives at the repo root, not on the path as a package."""
    spec = importlib.util.spec_from_file_location("serve_mod", os.path.join(REPO_ROOT, "serve.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SYSTEMS = ["audiovisual_integration", "social_sts_tpj", "visual_motion",
           "auditory", "dmn_scene_medial_parietal", "affect_reward"]
TIERS = ["high", "high", "moderate", "moderate", "moderate", "low"]


@pytest.fixture
def corpus(tmp_path):
    """Two clips in a fresh store. Clip A also gets an mp4 on disk so /video/{id} can
    serve it; clip B deliberately has none, to prove that path 404s instead of blowing up."""
    qdir, tdir, vdir = tmp_path / "qdrant", tmp_path / "timelines", tmp_path / "videos"
    tdir.mkdir(); vdir.mkdir()

    serve = _load_serve()
    serve.CFG.update(qdrant_path=str(qdir), timelines_dir=str(tdir), videos_dir=str(vdir))
    serve._DB = None
    db = serve.get_db()

    rng = np.random.default_rng(7)
    for i, vid in enumerate(("file:clip_a", "file:clip_b")):
        T = 40 + i * 10
        ts = np.arange(T, dtype=float)
        system_ts = rng.normal(size=(T, 6))
        system_ts[5:8] = np.nan                      # a dropped-segment gap (constraint #5)
        valid = ~np.isnan(system_ts).all(axis=1)
        name = f"{vid.replace(':', '_')}.npz"
        np.savez(tdir / name, region_ts=rng.normal(size=(T, 360)), system_ts=system_ts,
                 timestamps=ts, valid=valid)
        db.upsert_video(vid, rng.normal(size=360), payload=storage.build_payload(
            video_id=vid, title=f"clip {i}", duration=T,
            system_profile=np.nanmean(system_ts, axis=0).tolist(),
            system_names=SYSTEMS, system_tiers=TIERS, system_derived=[True] * 5 + [False],
            moments=[], timeline_path=name, transcript=""))

    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
                        "-i", "testsrc=size=64x64:rate=5:duration=2",
                        "-pix_fmt", "yuv420p", str(vdir / "clip_a.mp4")], check=True)

    yield serve, TestClient(serve.app), tmp_path
    serve._DB = None


def test_health(corpus):
    _, client, _ = corpus
    assert client.get("/health").json()["ok"] is True


def test_list_videos(corpus):
    _, client, _ = corpus
    r = client.get("/api/videos")
    assert r.status_code == 200
    ids = [v["video_id"] for v in r.json()]
    assert sorted(ids) == ["file:clip_a", "file:clip_b"]
    # the list projection stays slim — no timeline shipped to the sidebar
    assert "timeline" not in r.json()[0]
    assert r.json()[0]["system_tiers"] == TIERS


def test_get_one_video_includes_timeline_and_valid_mask(corpus):
    _, client, _ = corpus
    r = client.get("/api/videos/file:clip_a")
    assert r.status_code == 200
    body = r.json()
    tl = body["timeline"]
    assert len(tl["times"]) == len(tl["system_ts"]) == len(tl["valid"]) == 40
    assert len(tl["system_ts"][0]) == 6
    # dropped seconds survive as valid=False, and are zero-filled (NaN isn't JSON)
    assert tl["valid"][5:8] == [False, False, False]
    assert tl["system_ts"][5] == [0.0] * 6
    assert all(isinstance(x, float) for x in tl["times"])


def test_missing_video_404s(corpus):
    _, client, _ = corpus
    assert client.get("/api/videos/file:nope").status_code == 404


def test_similar_excludes_self(corpus):
    _, client, _ = corpus
    r = client.get("/api/videos/file:clip_a/similar")
    assert r.status_code == 200
    assert [n["video_id"] for n in r.json()] == ["file:clip_b"]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg to make the fixture mp4")
def test_video_file_served_and_faststart_cached(corpus):
    serve, client, tmp_path = corpus
    r = client.get("/video/file:clip_a")
    assert r.status_code == 200
    assert r.headers["content-type"] == "video/mp4"
    assert len(r.content) > 0
    # the remux is cached next to the source, so the second request is free
    assert (tmp_path / "videos" / "clip_a.mp4.fs.mp4").exists()
    assert client.get("/video/file:clip_a").status_code == 200


def test_video_file_missing_404s(corpus):
    _, client, _ = corpus
    assert client.get("/video/file:clip_b").status_code == 404


def test_webm_is_not_served_as_mp4(corpus):
    """Extension captures arrive as webm; announcing them as video/mp4 makes Chrome
    refuse to decode them."""
    serve, client, tmp_path = corpus
    (tmp_path / "videos" / "clip_b.webm").write_bytes(b"\x1a\x45\xdf\xa3fake")
    r = client.get("/video/file:clip_b")
    assert r.status_code == 200
    assert r.headers["content-type"] == "video/webm"


def test_delete_removes_vector_timeline_and_file(corpus):
    serve, client, tmp_path = corpus
    npz = tmp_path / "timelines" / "file_clip_a.npz"
    mp4 = tmp_path / "videos" / "clip_a.mp4"
    assert npz.exists()

    assert client.delete("/api/videos/file:clip_a").status_code == 200
    assert client.get("/api/videos/file:clip_a").status_code == 404
    assert [v["video_id"] for v in client.get("/api/videos").json()] == ["file:clip_b"]
    assert not npz.exists()
    if shutil.which("ffmpeg"):
        assert not mp4.exists()
        assert not mp4.with_name(mp4.name + ".fs.mp4").exists()
    # deleting again is a 404, not a crash
    assert client.delete("/api/videos/file:clip_a").status_code == 404


def test_bracketed_youtube_stem_is_glob_escaped(corpus):
    """Corpus ids come from yt-dlp filenames — "Dog of Wisdom II [TnlakHr-O4w]". In the
    fallback lookup that "[...]" is a glob character class, so an unescaped pattern would
    match "Dog of Wisdom II T.mp4" and hand back somebody else's clip."""
    serve, _, tmp_path = corpus
    vdir = tmp_path / "videos"
    (vdir / "Dog of Wisdom II T.mp4").write_bytes(b"\x00" * 8)   # only the decoy exists

    assert serve.resolve_video_file(str(vdir), "file:Dog of Wisdom II [TnlakHr-O4w]") is None

    real = vdir / "Dog of Wisdom II [TnlakHr-O4w].mp4"
    real.write_bytes(b"\x00" * 8)
    got = serve.resolve_video_file(str(vdir), "file:Dog of Wisdom II [TnlakHr-O4w]")
    assert got == real
