# Daniel Alvarez — 8/16/26
# Tests for backend/storage.py — embedded Qdrant, no server needed.
import sys, uuid
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
import storage


@pytest.fixture
def db(tmp_path):
    # fresh embedded DB per test in a temp dir — no shared state, no server
    return storage.QdrantVecDB(path=str(tmp_path / "qd"))


def _vec(seed):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(360).astype(np.float64).tolist()


def test_pid_is_deterministic_uuid():
    a = storage._pid("dQw4w9WgXcq")
    b = storage._pid("dQw4w9WgXcq")
    assert a == b                       # same video -> same point id
    assert a != storage._pid("other")   # different video -> different id
    uuid.UUID(a)                        # is a valid UUID (raises if not)


def test_collection_created(db):
    assert db.client.collection_exists(db.videos_v1)
    info = db.client.get_collection(db.videos_v1)
    assert info.config.params.vectors.size == 360


def test_upsert_then_retrieve_roundtrips_payload(db):
    db.upsert_video("vidA", _vec(1), payload={"title": "A", "duration": 42})
    got = db.client.retrieve(db.videos_v1, ids=[storage._pid("vidA")], with_payload=True)
    assert len(got) == 1
    assert got[0].payload["video_id"] == "vidA"   # real id injected into payload
    assert got[0].payload["title"] == "A"
    assert got[0].payload["duration"] == 42


def test_upsert_is_idempotent(db):
    db.upsert_video("vidA", _vec(1))
    db.upsert_video("vidA", _vec(2))              # same id, new vector -> overwrite, not duplicate
    assert db.client.count(db.videos_v1).count == 1


def test_two_videos_two_points(db):
    db.upsert_video("vidA", _vec(1))
    db.upsert_video("vidB", _vec(2))
    assert db.client.count(db.videos_v1).count == 2


def test_upsert_reports_completed(db):
    info = db.upsert_video("vidA", _vec(1))
    assert str(info.status) == "completed"        # UpdateStatus.COMPLETED


# add three more tests
# get _vidoe
# serarch 
# exclude_id dropping the query video