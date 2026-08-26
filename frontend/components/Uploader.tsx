"use client";
import { useRef, useState } from "react";
import { analyze, getJob, cancelJob } from "@/lib/api";

// Uploads a video to the live backend (analyze_server.py), polls the job, streams its log,
// and calls onDone(video_id, blobUrl) when the encode finishes. onDone gets a blob: URL for
// the picked file so the clip plays immediately, synced to the carpet. No-op vs plain serve.py.
export default function Uploader({ onDone }: { onDone: (id: string, src?: string) => void }) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [status, setStatus] = useState("");
  const [log, setLog] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const jobRef = useRef<string | null>(null);
  const canceledRef = useRef(false);

  async function onPick(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    e.target.value = "";
    if (!f) return;
    setBusy(true);
    setLog([]);
    setStatus("uploading…");
    canceledRef.current = false;
    jobRef.current = null;
    try {
      const { job_id, video_id } = await analyze(f);
      jobRef.current = job_id;
      for (;;) {
        await new Promise((r) => setTimeout(r, 2000));
        const jb = await getJob(job_id);
        if (jb.log) setLog(jb.log);
        setStatus(jb.stage || jb.status);
        if (jb.status === "done") {
          onDone(video_id, URL.createObjectURL(f));
          break;
        }
        if (jb.status === "canceled") { setStatus("canceled"); break; }
        if (jb.status === "error") { setStatus("error: " + (jb.error || "unknown")); break; }
      }
    } catch (err) {
      setStatus(String(err instanceof Error ? err.message : err));
    }
    setBusy(false);
    jobRef.current = null;
  }

  async function onCancel() {
    if (!jobRef.current || canceledRef.current) return;
    canceledRef.current = true;
    setStatus("canceling…");
    await cancelJob(jobRef.current);
  }

  return (
    <div style={{ marginBottom: 18 }}>
      <input ref={inputRef} type="file" accept="video/*" style={{ display: "none" }} onChange={onPick} disabled={busy} />
      {!busy ? (
        <button className="uploadbtn" onClick={() => inputRef.current?.click()}>
          Analyze a video
        </button>
      ) : (
        <div className="analyzing">
          <div className="analyzing-row">
            <span className="analyzing-label">{status || "analyzing…"}</span>
            <button className="cancelbtn" onClick={onCancel} disabled={canceledRef.current}>
              cancel
            </button>
          </div>
          {log.length > 0 && (
            <div className="joblog mono">
              {log.slice(-6).map((l, i) => (
                <div key={i} className="joblog-line">{l}</div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}