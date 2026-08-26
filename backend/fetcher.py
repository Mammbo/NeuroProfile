"""
fetcher.py — pull media down from a URL with yt-dlp.

Shared by `batch_encoding/batch_encode.py` (the corpus grinder) and
`batch_encoding/analyze_server.py` (`POST /analyze_url`, which is what the Chrome extension
uses). No torch and no model, so it lives on the CPU side and is testable locally.

Every entry point here goes through `input_handler.validate_url` first: that is the SSRF
guard (a URL must not resolve to a private/loopback address) and the extractor allowlist
(yt-dlp must have a real named extractor for the site, not just the generic fallback).

**JS runtime.** The opts request `--js-runtimes deno`, inherited from batch_encode.py.
Without deno on PATH yt-dlp still works but warns that some formats may be missing, so
install it on any host that will do real downloads.

**Colab caveat.** Colab egress IPs are blocked by YouTube, so a download from a Colab GPU
host will usually fail with a bot check no matter how well-formed the request is. Two ways
around it: drop a cookies export at `batch_encoding/yt_cookies.txt` (picked up automatically,
and gitignored — it is a live Google session credential, never commit it), or feed file paths
instead of URLs. `UnsupportedHostError` carries that explanation to the caller.
"""
import hashlib
import os
from pathlib import Path

import input_handler as ih

REPO_ROOT = Path(__file__).resolve().parent.parent

# yt-dlp cookies export, if the owner has one. Optional, and gitignored — see module docstring.
COOKIES_FILE = str(REPO_ROOT / "batch_encoding" / "yt_cookies.txt")

# 480p is what the corpus was encoded at, and TRIBE's video backbone downsamples hard anyway,
# so a bigger download buys nothing but wall-clock.
DEFAULT_HEIGHT = 480
DEFAULT_MAX_BYTES = 2 * 1024 ** 3


class FetchError(RuntimeError):
    """A URL we accepted but could not actually retrieve."""


def sha(s: str, n: int = 12) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:n]


def ytdlp_opts(extra_argv):
    """Build ydl_opts from CLI-style argv, so the flags read the same as a `yt-dlp` command."""
    from yt_dlp import parse_options
    argv = ["--js-runtimes", "deno", "--remote-components", "ejs:github"]
    if os.path.exists(COOKIES_FILE):
        argv += ["--cookies", COOKIES_FILE]
    return parse_options(argv + extra_argv).ydl_opts


def _friendly(url: str, exc: Exception) -> FetchError:
    """Turn yt-dlp's wall of text into something a popup can show in one line."""
    msg = str(exc)
    low = msg.lower()
    if "sign in to confirm" in low or "bot" in low or "429" in low or "captcha" in low:
        return FetchError(
            "the site blocked this download as automated traffic. If the backend is on Colab "
            "this is expected — Colab egress IPs are blocked by YouTube. Put a cookies export "
            "at batch_encoding/yt_cookies.txt on the backend host, or upload the file directly.")
    if "private" in low or "members-only" in low or "unavailable" in low:
        return FetchError("this video is private, members-only, or unavailable.")
    if "no video formats" in low or "requested format" in low:
        return FetchError("no downloadable video stream at that URL.")
    return FetchError(msg.strip().splitlines()[-1][:300] if msg.strip() else repr(exc))


def probe_url(url: str) -> dict:
    """Validate a URL and read its metadata WITHOUT downloading.

    Returns {"video_id", "title", "ytid", "duration"}. `video_id` uses the same
    `url:<id>` convention as batch_encode.py, so the two paths share ids and an already-
    encoded clip is recognised rather than re-encoded.

    Raises input_handler.InputError on a URL we refuse, FetchError on one we can't read.
    """
    ih.validate_url(url)                       # SSRF guard + extractor allowlist
    import yt_dlp
    try:
        with yt_dlp.YoutubeDL(ytdlp_opts(["--no-playlist", "--skip-download"])) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as e:
        raise _friendly(url, e) from e
    if not info:
        raise FetchError("no metadata at that URL")
    ytid = info.get("id") or sha(url)
    return {"video_id": f"url:{ytid}", "title": info.get("title") or url,
            "ytid": ytid, "duration": info.get("duration")}


def download_url(url: str, work, max_bytes: int = DEFAULT_MAX_BYTES,
                 height: int = DEFAULT_HEIGHT, validate: bool = True):
    """Download one video to `work/` and return (path, title, ytid).

    Capped at `height` and merged to mp4. `validate=False` is only for a caller that has
    already run probe_url on the same URL and does not want to pay for the check twice.
    """
    if validate:
        ih.validate_url(url)
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    import yt_dlp
    opts = ytdlp_opts([
        "-f", f"bv*[height<={height}]+ba/b[height<={height}]/b",
        "--merge-output-format", "mp4",
        "-o", str(work / "%(id)s.%(ext)s"),
        "--max-filesize", str(max_bytes),
        "--no-playlist",
    ])
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            path = Path(ydl.prepare_filename(info)).with_suffix(".mp4")
    except Exception as e:
        raise _friendly(url, e) from e
    if not path.exists():
        # yt-dlp reports the size cap as a normal skip, not an exception
        raise FetchError("download produced no file — the size cap was hit, or no format "
                         "matched")
    return str(path), (info.get("title") or url), (info.get("id") or sha(url))
