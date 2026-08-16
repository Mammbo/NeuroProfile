# Daniel Alvarez — 8/16/26
# Tests for backend/reducer.py
#   1. Analytic: timeline[t, v] = labels[v] + t makes every output closed-form,
#      so the numbers are pinned by math and wrong columns/grouping fail the check.
#   2. Reference regression: reduce a fixed random timeline, compare to a committed
#      .npz. Regenerate on purpose with:  REGEN_REFERENCE=1 pytest tests/reducer_test.py

import sys, os, json, warnings
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
import reducer

ICA = Path(__file__).resolve().parent.parent / "ica"
REFERENCE = Path(__file__).resolve().parent / "reducer_reference.npz"

T = 5
NAN_ROW = 3  # simulated dropped second


def _timeline():
    base = reducer.labels.astype(np.float64)
    tl = np.stack([base + t for t in range(T)])
    tl[NAN_ROW] = np.nan
    return tl


def _empty_rows():
    return [j for j, v in enumerate(reducer.verts_for_region) if len(v) == 0]


def test_region_timeline():
    out = reducer.reduce(_timeline())
    rids = np.array(reducer.region_ids, dtype=np.float64)
    expected = rids[None, :] + np.arange(T, dtype=float)[:, None]
    expected[NAN_ROW] = np.nan
    expected[:, _empty_rows()] = np.nan
    assert out["region_ts"].shape == (T, 360)
    np.testing.assert_allclose(out["region_ts"], expected, equal_nan=True)


def test_summary_vector():
    out = reducer.reduce(_timeline())
    rids = np.array(reducer.region_ids, dtype=np.float64)
    expected = rids + np.mean([t for t in range(T) if t != NAN_ROW])  # + 1.75
    expected[_empty_rows()] = np.nan
    assert out["summary_vec"].shape == (360,)
    np.testing.assert_allclose(out["summary_vec"], expected, equal_nan=True)


def test_system_timeline():
    ids = json.load(open(ICA / "fsaverage5_glasser_ids.json"))
    m = json.load(open(ICA / "region_system_map.json"))
    id2name = ids["id2name"]
    r2s = np.array([m["region_system"][id2name[str(rid)]] for rid in ids["region_ids"]])

    rids = np.array(reducer.region_ids, dtype=np.float64)
    region_ts = rids[None, :] + np.arange(T, dtype=float)[:, None]
    region_ts[NAN_ROW] = np.nan
    region_ts[:, _empty_rows()] = np.nan

    expected = np.empty((T, 6))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for s in range(6):
            expected[:, s] = np.nanmean(region_ts[:, np.where(r2s == s)[0]], axis=1)

    out = reducer.reduce(_timeline())
    assert out["system_ts"].shape == (T, 6)
    np.testing.assert_allclose(out["system_ts"], expected, equal_nan=True)


def test_valid_mask():
    out = reducer.reduce(_timeline())
    expected = np.ones(T, dtype=bool)
    expected[NAN_ROW] = False
    np.testing.assert_array_equal(out["valid"], expected)


def test_lookups():
    assigned = sum(len(v) for v in reducer.verts_for_region)
    assert 0 < assigned < 20484
    assert set(reducer.region_to_system.tolist()) == {0, 1, 2, 3, 4, 5}


def test_system_metadata():
    assert reducer.system_names[5] == "affect_reward"
    assert reducer.system_tiers[5] == "low"
    assert reducer.system_derived[5] is False
    assert reducer.system_tiers[2] == "high"
    assert reducer.system_tiers[3] == "high"


def test_reference_regression():
    rng = np.random.default_rng(0)
    tl = rng.standard_normal((6, 20484))
    tl[2] = np.nan
    out = reducer.reduce(tl)
    keys = ["region_ts", "system_ts", "summary_vec", "system_profile"]

    if os.environ.get("REGEN_REFERENCE") or not REFERENCE.exists():
        np.savez(REFERENCE, valid=out["valid"], **{k: out[k] for k in keys})
        pytest.skip("reference (re)generated — commit reducer_reference.npz, then this regresses")

    ref = np.load(REFERENCE)
    for k in keys:
        np.testing.assert_allclose(out[k], ref[k], equal_nan=True, rtol=1e-6, atol=1e-8)
    np.testing.assert_array_equal(out["valid"], ref["valid"])