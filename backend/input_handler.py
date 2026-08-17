# Daniel Alvarez
# 8/16/26
# validate and normalize an input (url, file, or text) before it hits the pipeline

import ipaddress
import socket
from dataclasses import dataclass, asdict
from urllib.parse import urlparse

MAX_UPLOAD_BYTES = 500 * 1024 * 1024 * 1024
MAX_TEXT_CHARS = 200_000
ALLOWED_SCHEMES = {"http", "https"}

class InputError(ValueError):
    pass


@dataclass
class NormalizedInput:
    source_kind: str
    media_kind: str | None
    url: str | None = None
    path: str | None = None
    text: str | None = None
    title: str | None = None

    def dict(self):
        return asdict(self)


def _assert_public_host(hostname, resolver=socket.getaddrinfo):
    # block loopback / internal targets so a url can't point back at our network
    try:
        infos = resolver(hostname, None)
    except socket.gaierror as e:
        raise InputError(f"could not resolve host: {hostname}") from e
    for info in infos:
        ip = info[4][0]
        addr = ipaddress.ip_address(ip)
        if not addr.is_global or addr.is_multicast:
            raise InputError("URL host resolves to a non-public address")


def _named_extractor_supports(url):
    # accept any real site yt-dlp knows, skip the generic catch-all
    from yt_dlp.extractor import gen_extractor_classes
    for ie in gen_extractor_classes():
        if ie.ie_key() == "Generic":
            continue
        try:
            if ie.suitable(url) and ie.working():
                return True
        except Exception:
            continue
    return False


def validate_url(url, allow_generic=False, resolver=socket.getaddrinfo):
    p = urlparse(url)
    if p.scheme not in ALLOWED_SCHEMES:
        raise InputError(f"unsupported scheme: {p.scheme or '(none)'}")
    if not p.hostname:
        raise InputError("URL has no host")
    _assert_public_host(p.hostname, resolver=resolver)
    if not allow_generic and not _named_extractor_supports(url):
        raise InputError("no supported extractor for this URL")
    return NormalizedInput(source_kind="url", media_kind=None, url=url, title=url)


def sniff_media_kind(head: bytes):
    # check magic bytes instead of trusting the extension
    if len(head) >= 12:
        if head[4:8] == b"ftyp":
            return "video"
        if head[:4] == b"\x1aE\xdf\xa3":
            return "video"
        if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
            return "audio"
        if head[:4] == b"RIFF" and head[8:12] == b"AVI ":
            return "video"
    if head[:3] == b"ID3" or head[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return "audio"
    if head[:4] == b"OggS":
        return "audio"
    if head[:4] == b"fLaC":
        return "audio"
    try:
        head.decode("utf-8")
        return "text"
    except UnicodeDecodeError:
        return None


def validate_file(path, head: bytes, size_bytes: int):
    if size_bytes > MAX_UPLOAD_BYTES:
        raise InputError(f"file too large: {size_bytes} > {MAX_UPLOAD_BYTES} bytes")
    kind = sniff_media_kind(head)
    if kind is None:
        raise InputError("unrecognized file type (expected video, audio, or text)")
    return NormalizedInput(source_kind="file", media_kind=kind, path=path, title=path)


def validate_text(text):
    text = (text or "").strip()
    if not text:
        raise InputError("text is empty")
    if len(text) > MAX_TEXT_CHARS:
        raise InputError(f"text too long: {len(text)} > {MAX_TEXT_CHARS} chars")
    return NormalizedInput(source_kind="text", media_kind="text", text=text,
                           title=text[:60])