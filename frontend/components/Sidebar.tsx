"use client";
import type { VideoSummary } from "@/lib/api";

interface Props {
  videos: VideoSummary[];
  activeId: string | null;
  onSelect: (id: string) => void;
}

export default function Sidebar({ videos, activeId, onSelect }: Props) {
  return (
    <div className="rail">
      <div className="kicker" style={{ marginBottom: 14 }}>Corpus</div>
      <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
        {videos.map((v) => (
          <button
            key={v.video_id}
            className={`clip${v.video_id === activeId ? " active" : ""}`}
            onClick={() => onSelect(v.video_id)}
          >
            <span className="dot" />
            <span className="label">{v.title}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
