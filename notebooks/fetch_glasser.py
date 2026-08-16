#pull glasser

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np
import nibabel.freesurfer.io as fsio

N_CORTEX = 20484                     # fsaverage5 cortical vertices (HARD CONSTRAINT #1)
HEMI_OFFSET = 1000                   # keeps L/R region ids distinct after concat

FIGSHARE = {
    "lh": "https://ndownloader.figshare.com/files/5528816",
    "rh": "https://ndownloader.figshare.com/files/5528819",
}
SHA256 = {
    "lh": "d4da634644b4c595dbda23963e01752059f0e7714be70169eb84e25e09ba2b44",
    "rh": "744eff4e57ce8121c43851eea425475baf92d9ba8686299a7435517ab972e9a2",
}

try:
    _REPO = Path(__file__).resolve().parent.parent
except NameError:                    # pasted into a notebook cell — no __file__
    _REPO = Path.cwd()

SEARCH_DIRS = [
    _REPO / "ica" / "atlas",                                 # committed copy
    Path("/content/drive/MyDrive/neuroprofile/atlas"),       # Colab, survives resets
    Path("/content"),                                        # Colab scratch
    Path.cwd(),
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch_annot(hemi: str, search_dirs=None) -> Path:
    """Return a verified path to {hemi}.HCP-MMP1.annot, downloading only if needed."""
    name = f"{hemi}.HCP-MMP1.annot"
    for d in search_dirs or SEARCH_DIRS:
        cand = Path(d) / name
        if not cand.is_file():
            continue
        got = _sha256(cand)
        if got == SHA256[hemi]:
            print(f"  {hemi}: {cand}")
            return cand
        # A failed download can leave an HTML error page here. read_annot would
        # then either throw something cryptic or parse partial garbage into
        # silently wrong labels, corrupting every downstream region mapping.
        print(f"  {hemi}: ignoring {cand} — checksum mismatch ({got[:12]}...)")

    dest_dir = Path("/content") if Path("/content").is_dir() else _REPO / "ica" / "atlas"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / name
    print(f"  {hemi}: not found locally, downloading -> {dest}")
    for attempt in range(1, 5):                  # presigned URL expires in 10s; retry
        try:
            urllib.request.urlretrieve(FIGSHARE[hemi], dest)
            if _sha256(dest) == SHA256[hemi]:
                print(f"  {hemi}: downloaded and verified")
                return dest
            print(f"    attempt {attempt}: checksum mismatch, retrying")
        except Exception as e:
            print(f"    attempt {attempt} failed: {e}")
    dest.unlink(missing_ok=True)
    raise RuntimeError(
        f"Could not fetch {name}. Copy it from the repo's ica/atlas/ into one of:\n  "
        + "\n  ".join(str(d) for d in (search_dirs or SEARCH_DIRS))
    )


def _name(x) -> str:
    return x.decode() if isinstance(x, bytes) else x


def _hemi_name(raw, h: str) -> str:
    nm = _name(raw)
    return nm if nm.startswith(("L_", "R_")) else f"{h}_{nm}"


def build_labels(n_cortex: int = N_CORTEX, search_dirs=None):
    """-> (labels, id2name).

    labels  (n_cortex,) int32, vertex -> region id, LH first then RH (+HEMI_OFFSET)
    id2name {region id: "L_V1_ROI"}; unlabelled medial wall keeps its "???" name
    """
    print("resolving atlas files:")
    lh_lab, _, lh_names = fsio.read_annot(fetch_annot("lh", search_dirs))
    rh_lab, _, rh_names = fsio.read_annot(fetch_annot("rh", search_dirs))
    print(f"full fsaverage per-hemi verts: {lh_lab.shape[0]}, {rh_lab.shape[0]}")

    n5 = n_cortex // 2
    if lh_lab.shape[0] < n5 or rh_lab.shape[0] < n5:
        raise ValueError(f"atlas has fewer than {n5} verts/hemi; cannot downsample")

    lh5, rh5 = lh_lab[:n5], rh_lab[:n5]           # nested-icosahedron downsample
    labels = np.concatenate([lh5, rh5 + HEMI_OFFSET]).astype(np.int32)
    assert labels.shape[0] == n_cortex, labels.shape

    id2name = {i: _hemi_name(lh_names[i], "L") for i in range(len(lh_names))}
    id2name.update({i + HEMI_OFFSET: _hemi_name(rh_names[i], "R")
                    for i in range(len(rh_names))})
    return labels, id2name


def region_ids(labels, id2name):
    """Ordered real-region ids, excluding FreeSurfer's '???' (medial wall)."""
    return [int(i) for i in np.unique(labels) if "???" not in id2name.get(int(i), "???")]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--save", metavar="DIR",
                    help="freeze labels (.npy) + id map (.json) into DIR")
    args = ap.parse_args()

    labels, id2name = build_labels()
    rids = region_ids(labels, id2name)
    unlabelled = int((~np.isin(labels, rids)).sum())

    print(f"\nlabels: {labels.shape} {labels.dtype}")
    print(f"regions: {len(rids)} (expect 360)")
    print(f"unlabelled vertices: {unlabelled} ({100 * unlabelled / labels.size:.1f}% — medial wall)")
    print(f"example names: {id2name.get(1)} / {id2name.get(1 + HEMI_OFFSET)}")

    sizes = np.array([int((labels == i).sum()) for i in rids])
    print(f"vertices per region: min {sizes.min()}, median {int(np.median(sizes))}, max {sizes.max()}")
    if sizes.min() < 5:
        tiny = [id2name[r] for r, s in zip(rids, sizes) if s < 5]
        print(f"WARNING — regions with <5 vertices after downsampling: {tiny}")

    if args.save:
        out = Path(args.save)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "fsaverage5_glasser_labels.npy", labels.astype(np.int16))
        (out / "fsaverage5_glasser_ids.json").write_text(json.dumps(
            {"id2name": {str(k): v for k, v in id2name.items()}, "region_ids": rids},
            indent=2))
        print(f"\nfroze -> {out}/fsaverage5_glasser_labels.npy + fsaverage5_glasser_ids.json")
