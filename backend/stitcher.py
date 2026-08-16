# Daniel Alvarez
# sticher.py
# takes an input of pred and segments and stiches them together.
import typing
import numpy as np
def stitch(chunk_predictions):
    """
    chunk_results: list of pred,segments, one per chunk in time order
    
    returns:
        timeline: np.ndarray, shape (T, 20484), one row per second, gaps = NaN
        timestamps: np.ndarray, shape(T, ) the absolute second for each row
    """

    rows = {} 
    # flatten every row to (absolute_time, vector).
    for preds, segments, chunk_start in chunk_predictions:
        for i, seg in enumerate(segments):
            t = int(round(chunk_start + seg.start))
             # Because chunks are in time order, a plain overwrite keeps the *later* chunk's
            # copy in the overlap. If you prefer earlier-chunk copies, use `if t not in rows`.
            rows[t] = preds[i]
    if not rows:
        return np.empty((0, 20484)), np.empty((0,))
    
    # assemble by timestamp into a dense array; missing seconds stay as NaN gaps.
    t_min, t_max = min(rows), max(rows)
    n_vertices = next(iter(rows.values())).shape[0]
    T = t_max - t_min + 1

    timeline = np.full((T, n_vertices), np.nan, dtype=np.float32)
    timestamps = np.arange(t_min, t_max + 1)

    for t, vec in rows.items():
        timeline[t - t_min] = vec

    return timeline, timestamps