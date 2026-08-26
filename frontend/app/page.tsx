"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  getVideos, getVideo, getSimilar, getApiBase, setApiBase, videoSrc, deleteVideo,
  type VideoSummary, type VideoFull, type Neighbor,
} from "@/lib/api";
import { fmtTime } from "@/lib/color";
import Sidebar from "@/components/Sidebar";
import Uploader from "@/components/Uploader";
import CarpetPlot from "@/components/CarpetPlot";
import SystemProfile from "@/components/SystemProfile";
import Neighbors from "@/components/Neighbors";

export default function Page() {
  const [videos, setVideos] = useState<VideoSummary[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [video, setVideo] = useState<VideoFull | null>(null);
  const [neighbors, setNeighbors] = useState<Neighbor[]>([]);
  const [cur, setCur] = useState(0);
  const [dur, setDur] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [apiInput, setApiInput] = useState("");

  // playable source for the selected clip. corpus clips resolve to serve.py /video/{id};
  // uploaded clips pass a blob: URL straight from the picked File. videoOk: null=loading,
  // true=has a real video (video is the clock), false=no file (fall back to a synthetic clock).
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [videoOk, setVideoOk] = useState<boolean | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const fallbackRef = useRef<string | null>(null);   // backend copy to try if the blob fails

  const rafRef = useRef<number | null>(null);
  const lastRef = useRef<number>(0);

  const select = useCallback(async (id: string, srcOverride?: string) => {
    setActiveId(id);
    setPlaying(false);
    setCur(0);
    setVideoOk(null);
    // Just-uploaded clip: play the instant in-browser blob, but keep the backend's persisted
    // copy (served from Drive) as a fallback in case the blob can't play.
    fallbackRef.current = srcOverride ? videoSrc(id) : null;
    setVideoUrl(srcOverride ?? videoSrc(id));
    try {
      const [v, nb] = await Promise.all([getVideo(id), getSimilar(id)]);
      setVideo(v);
      setNeighbors(nb);
      const lastT = v.timeline.times.length ? v.timeline.times[v.timeline.times.length - 1] + 1 : 0;
      setDur(v.duration || lastT || 0);
    } catch (e) {
      setError(String(e));
    }
  }, []);

  const loadCorpus = useCallback(async () => {
    setError(null);
    try {
      const vids = await getVideos();
      setVideos(vids);
      if (vids.length) select(vids[0].video_id);
      else {
        setVideo(null);
        setError("No videos yet. Upload one above, or encode a corpus / run serve.py --mock.");
      }
    } catch (e) {
      setVideo(null);
      setError(`Backend not reachable at ${getApiBase()} (${e}). Paste your Colab tunnel URL above.`);
    }
  }, [select]);

  useEffect(() => {
    setApiInput(getApiBase());
    loadCorpus();
  }, [loadCorpus]);

  const applyApiBase = useCallback(() => {
    const clean = apiInput.trim();
    if (!clean) return;
    setApiBase(clean);
    setApiInput(getApiBase());
    loadCorpus();
  }, [apiInput, loadCorpus]);

  // clock: when a real video is present it IS the clock (rAF samples currentTime for a
  // smooth playhead). Otherwise advance a synthetic clock so the playhead still sweeps.
  useEffect(() => {
    if (!playing) {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      return;
    }
    const v = videoRef.current;
    if (videoOk && v) {
      const loop = () => {
        setCur(v.currentTime);
        rafRef.current = requestAnimationFrame(loop);
      };
      rafRef.current = requestAnimationFrame(loop);
      return () => { if (rafRef.current) cancelAnimationFrame(rafRef.current); };
    }
    lastRef.current = performance.now();
    const loop = (ts: number) => {
      const dt = (ts - lastRef.current) / 1000;
      lastRef.current = ts;
      setCur((c) => {
        const nc = Math.min(dur, c + dt);
        if (nc >= dur) setPlaying(false);
        return nc;
      });
      rafRef.current = requestAnimationFrame(loop);
    };
    rafRef.current = requestAnimationFrame(loop);
    return () => { if (rafRef.current) cancelAnimationFrame(rafRef.current); };
  }, [playing, dur, videoOk]);

  const seek = (t: number) => {
    const clamped = Math.max(0, Math.min(dur, t));
    const v = videoRef.current;
    if (videoOk && v) v.currentTime = clamped;
    setCur(clamped);
  };

  const togglePlay = () => {
    const v = videoRef.current;
    if (videoOk && v) { v.paused ? v.play() : v.pause(); }
    else setPlaying((p) => !p);
  };

  const onUploaded = useCallback(async (id: string, src?: string) => {
    try {
      setVideos(await getVideos());
    } catch { /* ignore */ }
    select(id, src);
  }, [select]);

  const onDelete = useCallback(async () => {
    if (!activeId) return;
    if (!window.confirm(`Delete "${video?.title ?? activeId}" from the record? This removes it from the database and its stored video.`)) return;
    try {
      await deleteVideo(activeId);
      const vids = await getVideos();
      setVideos(vids);
      if (vids.length) {
        select(vids[0].video_id);
      } else {
        setVideo(null);
        setActiveId(null);
        setError("No videos left.");
      }
    } catch (e) {
      setError(`Delete failed: ${e}`);
    }
  }, [activeId, video, select]);

  return (
    <div className="page">
      <div className="head">
        <div className="brand">
          <h1>NeuroProfile</h1>
          <div className="tag">predicted cortical-response profiles for online video</div>
        </div>
        <div className="headright">
          <div className="apibox">
            <span className="apilabel">backend</span>
            <input
              className="apiinput mono"
              value={apiInput}
              onChange={(e) => setApiInput(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter") applyApiBase(); }}
              placeholder="https://….trycloudflare.com"
              spellCheck={false}
            />
            <button className="apiconnect" onClick={applyApiBase}>connect</button>
          </div>
          <div className="meta">{videos.length} clips · TRIBE v2 · CC BY-NC</div>
        </div>
      </div>

      <div className="disclaimer">
        Population-average predictions from a group model — not a measurement of any
        individual&rsquo;s brain, and not a claim about how a video affects you. Systems are
        tiered by confidence.
      </div>

      {error && <div className="err">{error}</div>}
      <hr className="rule" />

      {(video || videos.length > 0 || !error) && (
        <div className="body">
          <div>
            <Uploader onDone={onUploaded} />
            <Sidebar videos={videos} activeId={activeId} onSelect={select} />
          </div>

          {video ? (
            <div className="detail">
              <div className="detailhead">
                <div className="title">{video.title}</div>
                <div className="right">
                  {!videoOk && (
                    <button className="playbtn" onClick={togglePlay}>
                      {playing ? "❚❚" : "▶"}
                    </button>
                  )}
                  <span className="time">{fmtTime(cur)} / {fmtTime(dur)}</span>
                  <button className="delbtn" onClick={onDelete} title="Delete this clip from the record">
                    delete
                  </button>
                </div>
              </div>

              {videoUrl && (
                <div className="videowrap" style={{ display: videoOk ? "block" : "none" }}>
                  <video
                    key={videoUrl}
                    ref={videoRef}
                    src={videoUrl}
                    className="vid"
                    controls
                    playsInline
                    preload="metadata"
                    onLoadedMetadata={(e) => {
                      setVideoOk(true);
                      const d = e.currentTarget.duration;
                      if (isFinite(d) && d > 0) setDur(d);
                    }}
                    onError={() => {
                      // blob failed to play? try the backend's persisted copy once.
                      if (fallbackRef.current && fallbackRef.current !== videoUrl) {
                        const fb = fallbackRef.current;
                        fallbackRef.current = null;
                        setVideoOk(null);
                        setVideoUrl(fb);
                      } else {
                        setVideoOk(false);
                      }
                    }}
                    onPlay={() => setPlaying(true)}
                    onPause={() => setPlaying(false)}
                    onEnded={() => setPlaying(false)}
                  />
                </div>
              )}

              <div className="kicker" style={{ marginBottom: 12 }}>
                Response over time — systems × seconds
              </div>
              <CarpetPlot
                timeline={video.timeline}
                systemNames={video.system_names || []}
                systemTiers={video.system_tiers || []}
                duration={dur}
                currentTime={cur}
                onSeek={seek}
              />

              <hr className="rule" style={{ margin: "34px 0 28px" }} />

              <div className="cols">
                <div>
                  <div className="kicker" style={{ marginBottom: 16 }}>System profile — whole clip</div>
                  <SystemProfile
                    systemNames={video.system_names || []}
                    systemProfile={video.system_profile || []}
                  />
                </div>
                <div>
                  <div className="kicker" style={{ marginBottom: 16 }}>Nearest neighbors</div>
                  <Neighbors neighbors={neighbors} onSelect={select} />
                </div>
              </div>
            </div>
          ) : (
            <div className="detail">
              <div className="empty">
                No clip selected. Upload a video to analyze it on the live backend.
              </div>
            </div>
          )}
        </div>
      )}

      <div className="foot">
        TRIBE v2 is CC BY-NC — research/portfolio demo only. Output is 1 Hz, cortical surface
        (20,484 vertices) reduced to 360 regions → 6 functional systems.
      </div>
    </div>
  );
}