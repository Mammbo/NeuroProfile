"use client";
import { useEffect, useRef } from "react";
import type { Timeline } from "@/lib/api";
import { divColor } from "@/lib/color";
import { prettyName, tierAbbrev } from "@/lib/labels";

interface Props {
  timeline: Timeline;
  systemNames: string[];
  systemTiers: string[];
  duration: number;
  currentTime: number;
  onSeek: (t: number) => void;
}

const ROW_H = 22;
const GAP = "#eeece6"; // dropped second

export default function CarpetPlot({
  timeline,
  systemNames,
  systemTiers,
  duration,
  currentTime,
  onSeek,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const cv = canvasRef.current;
    if (!cv) return;
    const T = timeline.system_ts.length;
    const S = T > 0 ? timeline.system_ts[0].length : systemNames.length;
    if (T === 0) return;

    const vals: number[] = [];
    for (let t = 0; t < T; t++) {
      if (!timeline.valid[t]) continue;
      for (let s = 0; s < S; s++) vals.push(Math.abs(timeline.system_ts[t][s]));
    }
    vals.sort((a, b) => a - b);
    const scale = Math.max(0.4, vals[Math.floor(vals.length * 0.98)] || 1);

    const dpr = window.devicePixelRatio || 1;
    const cssW = cv.clientWidth || 760;
    cv.width = Math.max(T, cssW) * dpr;
    cv.height = S * ROW_H * dpr;
    cv.style.height = `${S * ROW_H}px`;
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    ctx.scale(dpr, dpr);
    const W = cv.width / dpr;
    const colW = W / T;

    for (let t = 0; t < T; t++) {
      for (let s = 0; s < S; s++) {
        if (!timeline.valid[t]) {
          ctx.fillStyle = GAP;
          ctx.fillRect(t * colW, s * ROW_H, colW + 0.6, ROW_H - 1);
          continue;
        }
        const dim = systemTiers[s] === "low" ? 0.6 : 1.0;
        ctx.fillStyle = divColor((timeline.system_ts[t][s] / scale) * dim);
        ctx.fillRect(t * colW, s * ROW_H, colW + 0.6, ROW_H - 1);
      }
    }
  }, [timeline, systemNames, systemTiers]);

  const playheadPct = duration ? (currentTime / duration) * 100 : 0;
  const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    onSeek(((e.clientX - r.left) / r.width) * duration);
  };

  return (
    <div>
      <div className="carpet">
        <div className="rowlabels">
          {systemNames.map((n, i) => (
            <div className="rowlabel" key={i} title={`confidence: ${systemTiers[i] || "medium"}`}>
              <span className="tier">{tierAbbrev(systemTiers[i])}</span>
              <span className="rowname">{prettyName(n)}</span>
            </div>
          ))}
        </div>
        <div className="carpet-holder">
          <canvas ref={canvasRef} onClick={handleClick} style={{ cursor: "pointer" }} />
          <div className="playhead" style={{ left: `${playheadPct}%` }} />
        </div>
      </div>
      <div className="axis">
        <span>0s</span>
        <span>{Math.round(duration)}s</span>
      </div>
      <div className="legend">
        <span>below avg</span>
        <span className="bar" />
        <span>above avg</span>
        <span className="spacer">shaded gap = second dropped by the model</span>
      </div>
    </div>
  );
}
