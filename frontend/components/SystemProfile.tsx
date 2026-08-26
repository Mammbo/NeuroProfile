"use client";
import { prettyName } from "@/lib/labels";

interface Props {
  systemNames: string[];
  systemProfile: number[];
}

export default function SystemProfile({ systemNames, systemProfile }: Props) {
  const mx = Math.max(0.1, ...systemProfile.map((p) => Math.abs(p)));
  return (
    <div>
      {systemProfile.map((p, i) => {
        const w = Math.min(50, (Math.abs(p) / mx) * 50);
        const left = p >= 0 ? 50 : 50 - w;
        return (
          <div className="sys" key={i}>
            <div className="sysname">{prettyName(systemNames[i] || `System ${i}`)}</div>
            <div className="barrow">
              <div className="mid" />
              <div className="v" style={{ left: `${left}%`, width: `${w}%` }} />
            </div>
            <div className="num">{p >= 0 ? "+" : ""}{p.toFixed(2)}</div>
          </div>
        );
      })}
    </div>
  );
}
