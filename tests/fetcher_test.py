"""
fetcher_test.py — the yt-dlp URL route (backend/fetcher.py).

yt-dlp is stubbed throughout. These tests are about the seam we own — the SSRF/allowlist
guard running first, ids matching batch_encode.py's convention, and yt-dlp's wall-of-text
errors collapsing to one actionable line — not about yt-dlp itself.
"""
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
import fetcher  # noqa: E402
import input_handler as ih  # noqa: E402


class _FakeYDL:
    """Stands in for yt_dlp.YoutubeDL as a context manager."""
    info = {"id": "abc123", "title": "A Clip", "duration": 42}
    raises = None
    last_opts = None
    outdir = "."

    def __init__(self, opts):
        type(self).last_opts = opts

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def extract_info(self, url, download=False):
        if type(self).raises:
            raise type(self).raises
        return type(self).info

    def prepare_filename(self, info):
        return os.path.join(type(self).outdir, f"{info.get('id', 'x')}.webm")


@pytest.fixture
def fake_ytdlp(monkeypatch):
    """Install a fake `yt_dlp` module and neutralise the network-touching validation."""
    _FakeYDL.raises = None
    _FakeYDL.info = {"id": "abc123", "title": "A Clip", "duration": 42}
    _FakeYDL.outdir = "."

    mod = types.ModuleType("yt_dlp")
    mod.YoutubeDL = _FakeYDL
    mod.parse_options = lambda argv: types.SimpleNamespace(ydl_opts={"argv": list(argv)})
    monkeypatch.setitem(sys.modules, "yt_dlp", mod)
    monkeypatch.setattr(ih, "validate_url", lambda url, **kw: None)
    return _FakeYDL


def test_probe_url_uses_the_batch_encode_id_convention(fake_ytdlp):
    """Ids must match batch_encode.py's `url:<id>`. If they drift, the same video encoded
    through the extension and through the corpus grinder lands as two separate clips."""
    meta = fetcher.probe_url("https://example.com/watch?v=abc123")
    assert meta == {"video_id": "url:abc123", "title": "A Clip",
                    "ytid": "abc123", "duration": 42}


def test_probe_url_falls_back_to_a_url_hash_when_the_site_gives_no_id(fake_ytdlp):
    fake_ytdlp.info = {"title": "No Id Here"}
    meta = fetcher.probe_url("https://example.com/x")
    assert meta["video_id"] == f"url:{fetcher.sha('https://example.com/x')}"


def test_probe_url_runs_the_ssrf_guard_before_touching_the_network(monkeypatch):
    """validate_url is the SSRF guard + extractor allowlist. If it ever stops running first,
    a crafted URL reaches yt-dlp — so yt_dlp must not even be importable here."""
    monkeypatch.setitem(sys.modules, "yt_dlp", None)     # any use would explode
    with pytest.raises(ih.InputError):
        fetcher.probe_url("file:///etc/passwd")


@pytest.mark.parametrize("url", ["http://127.0.0.1:8000/x", "http://localhost/x"])
def test_private_addresses_are_refused(url):
    with pytest.raises(ih.InputError):
        fetcher.probe_url(url)


def test_bot_check_becomes_an_actionable_message(fake_ytdlp):
    """The Colab case. A raw yt-dlp traceback in a popup is useless — the message has to say
    what to do about it."""
    fake_ytdlp.raises = RuntimeError("ERROR: Sign in to confirm you're not a bot")
    with pytest.raises(fetcher.FetchError) as e:
        fetcher.probe_url("https://example.com/x")
    assert "Colab" in str(e.value) and "yt_cookies.txt" in str(e.value)


@pytest.mark.parametrize("raw,expect", [
    ("ERROR: Private video", "private"),
    ("ERROR: Video unavailable", "private"),
    ("ERROR: Requested format is not available", "downloadable video stream"),
])
def test_other_errors_are_summarised(fake_ytdlp, raw, expect):
    fake_ytdlp.raises = RuntimeError(raw)
    with pytest.raises(fetcher.FetchError) as e:
        fetcher.probe_url("https://example.com/x")
    assert expect in str(e.value)


def test_download_caps_height_and_size(fake_ytdlp, tmp_path):
    """The corpus was encoded at 480p and TRIBE downsamples hard anyway, so a bigger
    download buys nothing but wall-clock — and an uncapped one can fill a Colab disk."""
    fake_ytdlp.outdir = str(tmp_path)
    (tmp_path / "abc123.mp4").write_bytes(b"x")

    path, title, ytid = fetcher.download_url("https://example.com/x", tmp_path,
                                             max_bytes=1234, height=480, validate=False)
    argv = fake_ytdlp.last_opts["argv"]
    assert "bv*[height<=480]+ba/b[height<=480]/b" in argv
    assert "1234" in argv
    assert "--no-playlist" in argv
    assert "--merge-output-format" in argv and "mp4" in argv
    assert (path, title, ytid) == (str(tmp_path / "abc123.mp4"), "A Clip", "abc123")


def test_download_reports_a_missing_file_rather_than_a_bad_path(fake_ytdlp, tmp_path):
    """yt-dlp reports the size cap as a normal skip, not an exception — an absent file is
    the only signal that the cap was hit."""
    fake_ytdlp.outdir = str(tmp_path)                    # nothing written
    with pytest.raises(fetcher.FetchError, match="size cap"):
        fetcher.download_url("https://example.com/x", tmp_path, validate=False)


def test_download_validates_by_default(fake_ytdlp, monkeypatch, tmp_path):
    """validate=False is only for a caller that already ran probe_url. The default must
    still check."""
    seen = []
    monkeypatch.setattr(ih, "validate_url", lambda url, **kw: seen.append(url))
    fake_ytdlp.outdir = str(tmp_path)
    (tmp_path / "abc123.mp4").write_bytes(b"x")

    fetcher.download_url("https://example.com/x", tmp_path)
    assert seen == ["https://example.com/x"]


def test_cookies_file_is_passed_through_only_when_it_exists(fake_ytdlp, tmp_path, monkeypatch):
    """A cookies export is the documented way around a bot check, so it has to reach yt-dlp
    when present — and never appear when it isn't."""
    monkeypatch.setattr(fetcher, "COOKIES_FILE", str(tmp_path / "absent.txt"))
    assert "--cookies" not in fetcher.ytdlp_opts([])["argv"]

    cookies = tmp_path / "yt_cookies.txt"
    cookies.write_text("# Netscape HTTP Cookie File\n")
    monkeypatch.setattr(fetcher, "COOKIES_FILE", str(cookies))
    argv = fetcher.ytdlp_opts([])["argv"]
    assert "--cookies" in argv and str(cookies) in argv
