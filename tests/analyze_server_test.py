"""
analyze_server_test.py — the live backend's non-GPU surface.

The encode itself needs CUDA + tribev2 and can't run here, so `_run_job` is stubbed out.
What IS testable locally is the seam the Chrome extension depends on: a webm upload has to
be accepted, persisted with its real extension, and served back with a webm content type.
"""
import hashlib
import importlib.util
import io
import os
import sys

import pytest
from fastapi.testclient import TestClient

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "backend"))

WEBM = b"\x1a\x45\xdf\xa3" + b"\x00" * 512          # matroska/webm magic + filler
MP4 = b"\x00\x00\x00\x20ftypisom" + b"\x00" * 512


@pytest.fixture
def server(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "analyze_mod", os.path.join(REPO_ROOT, "batch_encoding", "analyze_server.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    mod.CFG.update(qdrant_path=str(tmp_path / "qdrant"),
                   timelines_dir=str(tmp_path / "timelines"),
                   work_dir=str(tmp_path / "work"),
                   videos_dir=str(tmp_path / "videos"))
    # the real job needs a GPU; we only care that /analyze accepted and persisted the upload
    monkeypatch.setattr(mod, "_run_job", lambda *a, **k: None)
    yield mod, TestClient(mod.app), tmp_path


def _post(client, name, data, ctype):
    return client.post("/analyze", files={"file": (name, io.BytesIO(data), ctype)})


def test_webm_upload_is_accepted_and_kept_as_webm(server):
    """The extension posts capture.webm. If the playback copy loses the .webm extension it
    gets served as video/mp4 later and Chrome refuses to decode it."""
    mod, client, tmp_path = server
    r = _post(client, "capture.webm", WEBM, "video/webm")
    assert r.status_code == 200

    sha = hashlib.sha1(WEBM).hexdigest()[:12]
    assert r.json()["video_id"] == f"upload:{sha}"
    assert (tmp_path / "videos" / f"{sha}.webm").read_bytes() == WEBM


def test_uploaded_webm_is_served_with_a_webm_content_type(server):
    mod, client, tmp_path = server
    vid = _post(client, "capture.webm", WEBM, "video/webm").json()["video_id"]

    r = client.get(f"/video/{vid}")
    assert r.status_code == 200
    assert r.headers["content-type"] == "video/webm"
    assert r.content == WEBM


def test_mp4_upload_keeps_its_extension(server):
    mod, client, tmp_path = server
    sha = hashlib.sha1(MP4).hexdigest()[:12]
    assert _post(client, "clip.mp4", MP4, "video/mp4").status_code == 200
    assert (tmp_path / "videos" / f"{sha}.mp4").exists()


def test_unknown_extension_falls_back_to_mp4(server):
    """A filename we don't recognise still has to land somewhere ffmpeg can find it."""
    mod, client, tmp_path = server
    sha = hashlib.sha1(MP4).hexdigest()[:12]
    assert _post(client, "clip.weird", MP4, "application/octet-stream").status_code == 200
    assert (tmp_path / "videos" / f"{sha}.mp4").exists()


def test_empty_upload_is_rejected(server):
    _, client, _ = server
    assert _post(client, "capture.webm", b"", "video/webm").status_code == 400


def test_job_lifecycle_and_cancel(server):
    mod, client, _ = server
    job_id = _post(client, "capture.webm", WEBM, "video/webm").json()["job_id"]

    s = client.get(f"/jobs/{job_id}")
    assert s.status_code == 200 and s.json()["status"] == "queued"
    assert "result" not in s.json()                 # the tail endpoint must stay small

    assert client.get(f"/result/{job_id}").status_code == 409     # not done yet
    assert client.post(f"/jobs/{job_id}/cancel").json()["status"] == "canceling"
    assert mod._JOBS[job_id]["cancel"] is True

    assert client.get("/jobs/nope").status_code == 404
    assert client.post("/jobs/nope/cancel").status_code == 404


def test_health(server):
    _, client, _ = server
    assert client.get("/health").json()["ok"] is True
