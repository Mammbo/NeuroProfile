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
    # The real job needs a GPU, so the endpoint tests only care that /analyze accepted and
    # persisted the upload. Keep a handle on the real thing for the tests that drive its
    # non-GPU stages directly.
    mod._real_run_job = mod._run_job
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


# --- POST /analyze_url — the Chrome extension's path -------------------------------------
#
# The extension sends the tab's link and the backend fetches it with yt-dlp, so the whole
# clip is analysed at source quality instead of whatever a tab recording caught. yt-dlp is
# stubbed: what's under test is validation, the id convention, and the already-encoded
# short-circuit.

import sys as _sys  # noqa: E402
_sys.path.insert(0, os.path.join(REPO_ROOT, "backend"))
import input_handler as ih  # noqa: E402
import fetcher  # noqa: E402


@pytest.fixture
def url_server(server, monkeypatch):
    """server, plus a probe_url stub and a recorder in place of the encode thread."""
    mod, client, tmp_path = server
    calls = []
    monkeypatch.setattr(mod, "_run_job", lambda *a, **k: calls.append((a, k)))
    monkeypatch.setattr(mod, "probe_url", lambda url: {
        "video_id": "url:abc123", "title": "A Clip", "ytid": "abc123", "duration": 42})
    return mod, client, tmp_path, calls


def test_analyze_url_starts_a_job(url_server):
    mod, client, _, calls = url_server
    r = client.post("/analyze_url", data={"url": "https://example.com/watch?v=abc123"})
    assert r.status_code == 200

    body = r.json()
    assert body["video_id"] == "url:abc123"
    assert body["title"] == "A Clip"
    assert body["exists"] is False
    assert body["job_id"]

    # the job must be handed the URL, not a path — that's what selects the download stage
    assert calls and calls[0][1]["url"] == "https://example.com/watch?v=abc123"
    assert mod._JOBS[body["job_id"]]["video_id"] == "url:abc123"


def test_analyze_url_short_circuits_an_already_encoded_clip(url_server):
    """Re-submitting the same tab should cost nothing. Ids match batch_encode.py, so a clip
    already in the corpus is recognised rather than re-encoded."""
    mod, client, _, calls = url_server
    mod.get_db().upsert_video("url:abc123", [0.0] * 360, payload={"video_id": "url:abc123"})

    body = client.post("/analyze_url", data={"url": "https://example.com/x"}).json()
    assert body == {"video_id": "url:abc123", "title": "A Clip", "exists": True, "job_id": None}
    assert calls == [], "an already-encoded clip must not spawn an encode"


def test_analyze_url_rejects_an_empty_url(url_server):
    _, client, _, _ = url_server
    assert client.post("/analyze_url", data={"url": "   "}).status_code == 400


def test_analyze_url_rejects_what_the_ssrf_guard_rejects(server, monkeypatch):
    """A refused URL is the caller's fault -> 400, and no job is created."""
    mod, client, _ = server
    monkeypatch.setattr(mod, "_run_job", lambda *a, **k: None)

    def boom(url):
        raise ih.InputError("URL host resolves to a non-public address")
    monkeypatch.setattr(mod, "probe_url", boom)

    r = client.post("/analyze_url", data={"url": "http://169.254.169.254/latest/meta-data"})
    assert r.status_code == 400
    assert "non-public" in r.json()["detail"]
    assert mod._JOBS == {}


def test_analyze_url_reports_a_fetch_failure_as_502(server, monkeypatch):
    """We accepted the URL but the site refused us — that's upstream's fault, not the
    caller's, and the Colab bot-check message has to survive to the popup."""
    mod, client, _ = server
    monkeypatch.setattr(mod, "_run_job", lambda *a, **k: None)

    def boom(url):
        raise fetcher.FetchError("the site blocked this download as automated traffic. "
                                 "If the backend is on Colab this is expected")
    monkeypatch.setattr(mod, "probe_url", boom)

    r = client.post("/analyze_url", data={"url": "https://example.com/x"})
    assert r.status_code == 502
    assert "Colab" in r.json()["detail"]


def test_analyze_url_job_is_cancellable_like_an_upload(url_server):
    mod, client, _, _ = url_server
    job_id = client.post("/analyze_url", data={"url": "https://example.com/x"}).json()["job_id"]
    assert client.post(f"/jobs/{job_id}/cancel").json()["status"] == "canceling"
    assert mod._JOBS[job_id]["cancel"] is True


def test_run_job_download_stage(server, monkeypatch, tmp_path):
    """The download branch inside _run_job — the only genuinely new code on the encode path.

    Everything after the download needs CUDA, so chunk_video is made to raise: that stops the
    job right after the stage under test and lets us assert on the log, the persisted playback
    copy, and the error handling, all on a laptop.
    """
    mod, client, tmpdir = server

    src = tmp_path / "downloaded.mp4"
    src.write_bytes(b"\x00\x00\x00\x20ftypisom" + b"\x00" * 4096)

    def fake_download(url, work, max_bytes=None, height=None, validate=True):
        assert validate is False, "the URL was already validated by probe_url; don't redo it"
        return str(src), "Downloaded Title", "abc123"

    monkeypatch.setattr(mod, "download_url", fake_download)
    monkeypatch.setattr(mod, "chunk_video",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no GPU here")))

    job_id = "j1"
    mod._JOBS[job_id] = {"status": "queued", "stage": "queued", "video_id": "url:abc123",
                         "title": "placeholder", "log": [], "cancel": False}
    mod._real_run_job(job_id, None, "placeholder", "url:abc123",
                      url="https://example.com/x")

    job = mod._JOBS[job_id]
    log = "\n".join(job["log"])
    assert "downloading with yt-dlp" in log
    assert "downloaded downloaded.mp4" in log
    # the real title comes from yt-dlp, not from whatever the caller guessed
    assert job["title"] == "Downloaded Title"
    # the playback copy is named by the id stem so serving.resolve_video_file finds it
    assert (tmpdir / "videos" / "abc123.mp4").read_bytes() == src.read_bytes()
    # and the downstream failure is reported, not swallowed
    assert job["status"] == "error" and "no GPU here" in job["error"]


def test_run_job_reports_a_download_failure(server, monkeypatch):
    """A yt-dlp failure mid-job has to land in the job log as the actionable line, not as a
    bare traceback the popup can't use."""
    mod, client, _ = server

    def boom(url, work, **kw):
        raise fetcher.FetchError("the site blocked this download as automated traffic")
    monkeypatch.setattr(mod, "download_url", boom)

    job_id = "j2"
    mod._JOBS[job_id] = {"status": "queued", "stage": "queued", "video_id": "url:x",
                         "title": "t", "log": [], "cancel": False}
    mod._real_run_job(job_id, None, "t", "url:x", url="https://example.com/x")

    assert mod._JOBS[job_id]["status"] == "error"
    assert "blocked this download" in mod._JOBS[job_id]["error"]
