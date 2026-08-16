# Daniel Alvarez
#8/16/26

# the purpose of this reducer is to take the 20484 floats that make up the cortical mesh and create 360 named sections and then collapse those into 6 regions.

# load the labels, ids and system map
import json, numpy as np
import warnings
from pathlib import Path


# get timeline from stitcher!
labels = np.load("ica/fsaverage5_glasser_labels.npy")
ids = json.load(open("ica/fsaverage5_glasser_ids.json"))
m = json.load(open("ica/region_system_map.json"))

# precompute lookups
region_ids = ids["region_ids"]
id2name = ids["id2name"]

# For each region: which vertex columns does it belong to
verts_for_region = [np.where(labels == rid)[0] for rid in region_ids]   # list of 360 index arrays

# For each region row (0..359): which system am I in?
region_to_system = np.array([
    m["region_system"][id2name[str(rid)]]   # rid -> name -> system id
    for rid in region_ids
])      

rows_for_system = [np.where(region_to_system == s)[0] for s in range(6)]  

system_names   = [m["systems"][str(s)]["name"]    for s in range(6)]
system_tiers   = [m["systems"][str(s)]["tier"]    for s in range(6)]
system_derived = [m["systems"][str(s)]["derived"] for s in range(6)] 

def reduce(timeline):
        
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN dropped seconds warn; that's fine

        # region timeline (T, 360): average vertices within each region, every second
        region_ts = np.column_stack([
            np.nanmean(timeline[:, idx], axis=1) for idx in verts_for_region
        ])

        # system timeline (T, 6): average regions within each system, every second
        system_ts = np.column_stack([
            np.nanmean(region_ts[:, rows], axis=1) for rows in rows_for_system
        ])

        # collapse time. summary_vec is the Qdrant search key.
        summary_vec    = np.nanmean(region_ts, axis=0)   # (360,)
        system_profile = np.nanmean(system_ts, axis=0)   # (6,)

    # which seconds are real vs. dropped-gap (for the dashboard)
    valid = ~np.isnan(timeline).all(axis=1)              # (T,) bool


    return {
        "region_ts": region_ts, "system_ts": system_ts,
        "summary_vec": summary_vec, "system_profile": system_profile,
        "valid": valid,
        "system_names": system_names, "system_tiers": system_tiers,
        "system_derived": system_derived,
    }

if __name__ == "__main__":
    print("assigned vertices:", sum(len(v) for v in verts_for_region))   # < 20484 (??? wall excluded)
    print("systems present:  ", set(region_to_system.tolist()))          # must be {0,1,2,3,4,5}

    # fake a timeline so we can run with no GPU / no video
    rng = np.random.default_rng(0)
    timeline = rng.standard_normal((5, 20484)).astype(np.float32)
    timeline[2, :] = np.nan                                # simulate one dropped second

    out = reduce(timeline)
    print("region_ts ", out["region_ts"].shape)           # (5, 360)
    print("system_ts ", out["system_ts"].shape)           # (5, 6)
    print("summary   ", out["summary_vec"].shape)         # (360,)
    print("valid     ", out["valid"])   