# Daniel Alvarez — 8/16/26
# Tests for backend/storage.py — embedded Qdrant, no server needed.
# Covers all of Part 5: _pid, collection, payload schema, vector guard,
# and the write -> read -> similarity round trip (get_video + search_similar).

import sys, uuid
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
import storage


@pytest.fixture
def db(tmp_path):
    return storage.QdrantVecDB(path=str(tmp_path / "qd"))   # fresh embedded DB per test


def _onehot(i, n=360):
    v = np.zeros(n, dtype=np.float64)
    v[i] = 1.0
    return v.tolist()


# --- helpers / conventions -------------------------------------------------

def test_pid_is_deterministic_uuid():
    a = storage._pid("dQw4w9WgXcq")
    assert a == storage._pid("dQw4w9WgXcq")     # same video -> same id
    assert a != storage._pid("other")
    uuid.UUID(a)                                 # valid UUID (raises otherwise)


def test_clean_vector_zeros_nonfinite():
    out = storage._clean_vector([np.nan, np.inf, -np.inf, 3.0] + [0.0] * 356)
    assert np.isfinite(out).all()
    assert out[0] == 0.0 and out[1] == 0.0 and out[2] == 0.0 and out[3] == 3.0


def test_build_payload_requires_all_fields():
    with pytest.raises(TypeError):
        storage.build_payload(video_id="A")      # missing the rest -> TypeError


def test_build_payload_full():
    p = storage.build_payload(
        video_id="A", title="t", duration=12,
        system_profile=[0.0] * 6, system_names=["x"] * 6,
        system_tiers=["high"] * 6, system_derived=[True] * 6,
        moments=[], timeline_path="/drive/A.npz", transcript="",
    )
    assert set(p.keys()) == set(storage.PAYLOAD_FIELDS)


# --- collection ------------------------------------------------------------

def test_collection_created_size_360(db):
    assert db.client.collection_exists(db.videos_v1)
    info = db.client.get_collection(db.videos_v1)
    assert info.config.params.vectors.size == 360


# --- write -> read ---------------------------------------------------------

def test_upsert_then_get_video_roundtrips_payload(db):
    payload = storage.build_payload(
        video_id="A", title="Rickroll", duration=213,
        system_profile=[0.1] * 6, system_names=["a"] * 6,
        system_tiers=["low"] * 6, system_derived=[False] * 6,
        moments=[{"t": 5, "label": "peak"}], timeline_path="/drive/A.npz",
        transcript="never gonna give you up",
    )
    db.upsert_video("A", _onehot(0), payload=payload)
    got = db.get_video("A")
    assert got["video_id"] == "A"
    assert got["title"] == "Rickroll"
    assert got["timeline_path"] == "/drive/A.npz"
    assert got["transcript"] == "never gonna give you up"


def test_get_video_missing_returns_none(db):
    assert db.get_video("does-not-exist") is None


def test_upsert_is_idempotent(db):
    db.upsert_video("A", _onehot(0))
    db.upsert_video("A", _onehot(1))              # same id -> overwrite, not a 2nd point
    assert db.client.count(db.videos_v1).count == 1


def test_upsert_cleans_nan_vector(db):
    v = _onehot(0); v[5] = float("nan"); v[6] = float("inf")
    db.upsert_video("A", v)                        # insurance: does not raise
    assert db.get_video("A")["video_id"] == "A"


# --- similarity ------------------------------------------------------------

def test_search_ranks_nearer_first(db):
    db.upsert_video("A", _onehot(0))              # identical to query -> cos 1
    db.upsert_video("B", _onehot(1))              # orthogonal        -> cos 0
    res = db.search_similar(_onehot(0), limit=2)
    assert res[0]["video_id"] == "A"
    assert res[0]["score"] >= res[1]["score"]


def test_search_exclude_id_drops_query_video(db):
    db.upsert_video("A", _onehot(0))
    db.upsert_video("B", _onehot(1))
    res = db.search_similar(_onehot(0), limit=5, exclude_id="A")
    ids = [r["video_id"] for r in res]
    assert "A" not in ids
    assert ids[0] == "B"