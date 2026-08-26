"""
serving.py — read-side helpers shared by the two HTTP front-ends.
"""
import glob as _glob
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

# Extensions we will serve back for playback. webm is here because the Chrome extension
# uploads MediaRecorder webm, not mp4.
VIDEO_EXTS = (".mp4", ".webm", ".mov", ".mkv", ".m4v", ".avi")


def slim(p: dict) -> dict:
    """The list-view projection of a payload — enough for the sidebar, no timeline."""
    return {"video_id": p["video_id"], "title": p.get("title", p["video_id"]),
            "duration": p.get("duration"), "system_names": p.get("system_names"),
            "system_tiers": p.get("system_tiers"), "system_profile": p.get("system_profile")}


def timeline_payload(system_ts, timestamps, valid) -> dict:
    """JSON-safe timeline. NaN is not valid JSON, so dropped seconds go out as 0.0 and the
    `valid` mask (constraint #5) is what the UI must use to grey them out."""
    st = np.where(np.isfinite(system_ts), system_ts, 0.0)
    return {"times": [float(t) for t in timestamps],
            "system_ts": st.tolist(),
            "valid": [bool(v) for v in valid]}


def load_timeline(timelines_dir, timeline_path) -> dict:
    """Read one timeline .npz. `timeline_path` in the payload is a BASENAME — timelines are
    relocatable between Drive and local — so resolve it against our own dir.
    Raises FileNotFoundError if it isn't there."""
    fp = Path(timelines_dir) / os.path.basename(timeline_path)
    if not fp.exists():
        raise FileNotFoundError(fp.name)
    z = np.load(fp)
    system_ts = z["system_ts"].astype(float)
    ts = (z["timestamps"].astype(float) if "timestamps" in z
          else np.arange(system_ts.shape[0], dtype=float))
    valid = (z["valid"].astype(bool) if "valid" in z
             else ~np.isnan(system_ts).any(axis=1))
    return timeline_payload(system_ts, ts, valid)


def compute_moments(system_ts, timestamps, names, tiers, top_k=8):
    """Local peaks per system, strongest first. Times come from `timestamps` (segment.start),
    never from the row index — rows can be missing (constraint #5)."""
    from scipy.signal import find_peaks
    out = []
    for s in range(system_ts.shape[1]):
        col = system_ts[:, s]
        mask = ~np.isnan(col)
        if mask.sum() < 3:
            continue
        filled = np.where(mask, col, np.nanmin(col[mask]))
        peaks, _ = find_peaks(filled, distance=3)
        for pk in peaks:
            out.append({"t": int(timestamps[pk]), "system": names[s],
                        "tier": tiers[s], "value": float(col[pk])})
    out.sort(key=lambda m: m["value"], reverse=True)
    return out[:top_k]


def resolve_video_file(videos_dir, video_id):
    """Map a video_id ('file:<stem>' / 'upload:<sha>') to a playable file in videos_dir.
    Returns None (-> 404) when nothing matches."""
    if not videos_dir or not Path(videos_dir).exists():
        return None
    stem = video_id.split(":", 1)[1] if ":" in video_id else video_id
    for ext in VIDEO_EXTS:
        p = Path(videos_dir) / f"{stem}{ext}"
        if p.exists():
            return p
    # escape the stem: corpus ids come from yt-dlp filenames like "Title [abc123]", and an
    # unescaped "[...]" is a glob character class that silently matches nothing.
    for p in Path(videos_dir).glob(f"{_glob.escape(stem)}.*"):
        if p.suffix.lower() in VIDEO_EXTS and not p.name.endswith(".fs.mp4"):
            return p
    return None


WEB_SAFE_VIDEO = {"h264", "avc1"}
WEB_SAFE_AUDIO = {"aac", "mp3"}


def probe_codecs(fp: Path):
    """(video_codec, audio_codec) for fp; either may be None. Returns (None, None) if we
    can't probe, which callers treat as "just copy it and hope"."""
    if shutil.which("ffprobe") is None:
        return None, None
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,codec_name",
             "-of", "csv=p=0", str(fp)], capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            return None, None
        v = a = None
        for line in r.stdout.splitlines():
            parts = line.strip().split(",")
            if len(parts) != 2:
                continue
            name, kind = parts
            if kind == "video" and v is None:
                v = name
            elif kind == "audio" and a is None:
                a = name
        return v, a
    except (subprocess.SubprocessError, OSError):
        return None, None


def faststart_file(fp: Path) -> Path:
    try:
        if fp.suffix.lower() not in (".mp4", ".m4v", ".mov"):
            return fp
        cache = fp.with_name(fp.name + ".fs.mp4")
        if cache.exists() and cache.stat().st_mtime >= fp.stat().st_mtime:
            return cache
        if shutil.which("ffmpeg") is None:
            return fp

        vcodec, acodec = probe_codecs(fp)
        vargs = (["-c:v", "copy"] if vcodec is None or vcodec in WEB_SAFE_VIDEO
                 else ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                       "-pix_fmt", "yuv420p"])
        if acodec is None:
            aargs = ["-c:a", "copy"]
        elif acodec in WEB_SAFE_AUDIO:
            aargs = ["-c:a", "copy"]
        else:
            aargs = ["-c:a", "aac", "-b:a", "128k"]
        if "libx264" in vargs or "aac" in aargs:
            print(f"[video] transcoding {fp.name} ({vcodec}/{acodec}) for browser playback — "
                  f"this happens once, then it is cached", flush=True)

        tmp = fp.with_name(fp.name + ".fs.tmp.mp4")
        r = subprocess.run(["ffmpeg", "-y", "-i", str(fp), *vargs, *aargs,
                            "-movflags", "+faststart", str(tmp)], capture_output=True)
        if r.returncode == 0 and tmp.exists():
            tmp.replace(cache)
            print(f"[video] cached streamable copy {fp.name} -> {cache.name}", flush=True)
            return cache
        if tmp.exists():
            tmp.unlink()
        print(f"[video] remux failed for {fp.name}, serving the original:\n"
              f"{r.stderr.decode('utf-8', 'replace')[-500:]}", flush=True)
        return fp
    except Exception as e:
        print(f"[video] faststart remux failed for {fp}: {e}", flush=True)
        return fp


def media_type_for(fp: Path) -> str:
    """webm must not be announced as video/mp4 or Chrome refuses to decode it."""
    return {".webm": "video/webm", ".mkv": "video/x-matroska",
            ".mov": "video/quicktime", ".avi": "video/x-msvideo"}.get(
                fp.suffix.lower(), "video/mp4")
