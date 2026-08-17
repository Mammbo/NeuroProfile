# Daniel Alvarez
# 8/16/26
# tests for input_handler.py and app.py

import io, sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
import input_handler as ih
import app as appmod

client = TestClient(appmod.app)


def _resolver_to(ip):
    return lambda host, port: [(0, 0, 0, "", (ip, 0))]


@pytest.mark.parametrize("ip", ["127.0.0.1", "169.254.169.254", "10.0.0.1",
                                "192.168.1.5", "::1", "0.0.0.0"])
def test_ssrf_rejects_nonpublic(ip):
    with pytest.raises(ih.InputError):
        ih._assert_public_host("evil.test", resolver=_resolver_to(ip))


def test_ssrf_allows_public():
    ih._assert_public_host("ok.test", resolver=_resolver_to("8.8.8.8"))


def test_url_bad_scheme():
    with pytest.raises(ih.InputError):
        ih.validate_url("file:///etc/passwd", resolver=_resolver_to("8.8.8.8"))


def test_named_extractor_accepts_youtube():
    assert ih._named_extractor_supports("https://www.youtube.com/watch?v=dQw4w9WgXcq")


def test_named_extractor_rejects_random_site():
    assert not ih._named_extractor_supports("https://not-a-real-video-host.example/x")


def test_validate_url_youtube_ok():
    out = ih.validate_url("https://www.youtube.com/watch?v=dQw4w9WgXcq",
                          resolver=_resolver_to("8.8.8.8"))
    assert out.source_kind == "url"


@pytest.mark.parametrize("head,kind", [
    (b"\x00\x00\x00\x20ftypmp42", "video"),
    (b"\x1aE\xdf\xa3\x00\x00\x00\x00\x00\x00\x00\x00", "video"),
    (b"RIFF\x00\x00\x00\x00WAVEfmt ", "audio"),
    (b"ID3\x03\x00\x00\x00\x00\x00\x00", "audio"),
    (b"OggS\x00\x02\x00\x00\x00\x00\x00\x00", "audio"),
    (b"hello world this is plain text", "text"),
    (b"\x89PNG\r\n\x1a\n\x00\x00\x00\x00", None),
])
def test_sniff(head, kind):
    assert ih.sniff_media_kind(head) == kind


def test_validate_file_size_limit():
    with pytest.raises(ih.InputError):
        ih.validate_file("/tmp/x", b"ID3\x03\x00\x00", ih.MAX_UPLOAD_BYTES + 1)


def test_text_tagged_as_text():
    out = ih.validate_text("the quick brown fox")
    assert out.source_kind == "text" and out.media_kind == "text"


def test_text_empty_rejected():
    with pytest.raises(ih.InputError):
        ih.validate_text("   ")


def test_text_too_long_rejected():
    with pytest.raises(ih.InputError):
        ih.validate_text("a" * (ih.MAX_TEXT_CHARS + 1))


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_submit_text():
    r = client.post("/submit", data={"text": "hello there"})
    assert r.status_code == 200 and r.json()["source_kind"] == "text"


def test_submit_file():
    f = {"file": ("clip.bin", io.BytesIO(b"ID3\x03\x00\x00\x00\x00\x00\x00rest"), "application/octet-stream")}
    r = client.post("/submit", files=f)
    assert r.status_code == 200 and r.json()["media_kind"] == "audio"


def test_submit_rejects_png_file():
    f = {"file": ("x.png", io.BytesIO(b"\x89PNG\r\n\x1a\n....."), "image/png")}
    r = client.post("/submit", files=f)
    assert r.status_code == 400


def test_submit_requires_exactly_one():
    assert client.post("/submit", data={}).status_code == 400
    assert client.post("/submit", data={"text": "x", "url": "y"}).status_code == 400