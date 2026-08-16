# tests/stitcher_test.py
import numpy as np
from backend.stitcher import stitch

class FakeSeg:
    def __init__(self, start):
        self.start = start

def make_chunk(chunk_start, n_seconds, fill):
    preds = np.full((n_seconds, 4), fill, dtype=np.float32)
    segments = [FakeSeg(s) for s in range(n_seconds)]
    return (preds, segments, chunk_start)

def test_stitch_two_overlapping_chunks(): 
    chunkA = make_chunk(chunk_start=0,  n_seconds=100, fill=1.0)
    chunkB = make_chunk(chunk_start=90, n_seconds=100, fill=2.0)
    timeline, timestamps = stitch([chunkA, chunkB])

    assert timestamps[0] == 0
    assert timestamps[-1] == 189
    assert timeline.shape[0] == 190
    assert not np.isnan(timeline).any()
    assert np.all(timeline[90:100] == 2.0)  
    assert np.all(timeline[0:90] == 1.0)
    assert np.all(timeline[100:190] == 2.0)

def test_stitch_leaves_gap_for_dropped_second():   
    preds = np.full((3, 4), 5.0, dtype=np.float32)
    segments = [FakeSeg(0), FakeSeg(1), FakeSeg(3)]  
    timeline, timestamps = stitch([(preds, segments, 0)])
    assert np.isnan(timeline[2]).all()
    assert np.all(timeline[0] == 5.0)
    assert np.all(timeline[3] == 5.0)