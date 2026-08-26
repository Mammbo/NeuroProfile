"use client";
import type { Neighbor } from "@/lib/api";

interface Props {
  neighbors: Neighbor[];
  onSelect: (id: string) => void;
}

export default function Neighbors({ neighbors, onSelect }: Props) {
  if (!neighbors.length) return <div className="empty">need ≥2 videos for neighbors</div>;
  return (
    <div>
      {neighbors.map((n) => (
        <div className="nb" key={n.video_id} onClick={() => onSelect(n.video_id)}>
          <span className="score">{n.score.toFixed(2)}</span>
          <span className="nt">{n.title}</span>
        </div>
      ))}
    </div>
  );
}
