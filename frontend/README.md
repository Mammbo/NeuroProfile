# NeuroProfile: Dashboard

React/TypeScript dashboard for NeuroProfile. Renders the carpet plot (systems × time,
playhead-synced to the video), the  whole clip system profile with confidence tiers, and
nearest neighbour retrieval. Reads from either FastAPI serving layer: `backend/serve.py`
(offline)
or `batch_encoding/analyze_server.py` (live GPU).

## Run

1. Start an API. Offline, from the repo root, with a synced `qdrant_data/` and
   `data/timelines/`:
   ```bash
   python backend/serve.py --videos-dir ./data/videos --prewarm   # http://localhost:8000
   python backend/serve.py --mock                                 # or: four synthetic clips
   ```
2. Start this app:
   ```bash
   npm install
   npm run dev
   ```

**You do not need a `.env.local`.** The API base resolves at runtime:
`localStorage["np_api"]` (whatever you paste into the header's **backend** box) beats
`NEXT_PUBLIC_API_BASE` beats `http://localhost:8000`.

Production build: `npm run build && npm start`.

## Structure

- `app/page.tsx` orchestration: fetching, the playback clock, seek, delete, layout.
- `components/Sidebar.tsx` — the corpus list
- `components/Uploader.tsx` — live upload with the streaming stage log and cancel
- `components/CarpetPlot.tsx` — the canvas carpet plot and playhead overlay
- `components/SystemProfile.tsx` — whole-clip bars with tier labels
- `components/Neighbors.tsx` — nearest neighbours by cosine
- `lib/api.ts` — typed client for the serving endpoints
- `lib/color.ts` — diverging colormap + `fmtTime`
- `lib/labels.ts` — display names for the six systems

No CSS framework and no chart library: the carpet plot is hand-drawn on a `<canvas>` with a
positioned playhead overlay, and the styles are plain CSS in `app/globals.css`.
