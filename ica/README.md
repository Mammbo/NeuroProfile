# Functional systems recovered from the TRIBE v2 readout

**Frozen 2026-08-15.** These files are inputs to the reducer — regenerating them
changes every downstream number, so treat them as fixed.

| file | what it is |
| --- | --- |
| `region_system_map.json` | region → system id, plus system names, reliability tiers, provenance |
| `fsaverage5_glasser_labels.npy` | `(20484,)` int16, vertex → Glasser region id (LH 0:10242, RH 10242:20484, RH ids offset by 1000) |
| `fsaverage5_glasser_ids.json` | region id → name, and `region_ids` in the reducer's row order |
| `atlas/*.annot` | source HCP-MMP1 parcellation on full fsaverage, SHA-256 pinned |

## What was done

The released `facebook/tribev2` predicts cortical activity at 20,484 fsaverage5
vertices. A per-vertex output is not interpretable, so the goal was a small set
of functional systems to reduce it onto — derived from the model itself rather
than picked by hand.

The model's latent→cortex readout (`model._model.predictor.weights`, a single
`(1, 2048, 20484)` subject head, which is the released unseen-subject readout)
was decomposed with `FastICA(n_components=5, random_state=0)`, treating the 2048
latent dimensions as samples and the 20,484 vertices as features. Each component
is therefore a spatial map over cortex. Every Glasser region was assigned to the
component it loads onto most strongly, by mean absolute loading.

## The five recovered systems

| id | system | tier | regions | top-loading areas |
| --- | --- | --- | --- | --- |
| 0 | `audiovisual_integration` | moderate | 39 | A5, STSdp, V4t, V7, MT, LO2/3 |
| 1 | `social_sts_tpj` | moderate | 53 | TPOJ1/2, STV, STSdp, STSvp |
| 2 | `visual_motion` | high | 35 | MT, MST, V4t, FST, LO1/2/3, V3CD |
| 3 | `auditory` | high | 92 | PBelt, LBelt, A4, A1, TA2 |
| 4 | `dmn_scene_medial_parietal` | moderate | 93 | POS1, v23ab, 31pd, 7m, PHA1/2, VMV2/3 |
| 5 | `affect_reward` | low | 48 | anterior insula, ACC, OFC/vmPFC |

Components 0–4 are data-derived. **System 5 is not** — it is a hand-assigned
literature-based set, flagged `"derived": false` in the JSON. TRIBE v2 released
cortical weights only, so there is no amygdala or accumbens; anterior insula,
ACC and OFC/vmPFC are cortical proxies and are tiered Low accordingly.
Posterior insula (`PoI1`/`PoI2`) was deliberately excluded — it is
interoceptive/somatosensory, not affect/reward.

## Validation

- **360/360 regions** assigned; 1,742 medial-wall vertices (8.5%) excluded as `???`.
- **Sign robustness:** ICA component sign is arbitrary, and averaging signed
  loadings inside a region lets a genuinely strong region cancel itself to ~0.
  Recomputing with mean-of-absolute loadings changed only **22/360 regions
  (6.1%)**, so the assignment does not hinge on that choice. The signed variant
  was kept.
- **Anchor check** — regions with known function, used to test both component
  coherence and the assumption that TRIBE's vertex order is standard fsaverage5
  (LH first). 7 of 8 anchors landed on the same component in both hemispheres,
  which confirms the ordering; only `PGi` split (L→1, R→3).

## Known limitations

Systems 3 and 4 hold 92 and 93 of the 360 regions while 0–2 hold 35–53. The
larger two behave as catch-alls: regions that load weakly on every component
fall into whichever component has the largest overall scale. Two anchors show
this directly — **`V1` (primary visual) and `4` (primary motor) both land in
`auditory`**, which is not anatomically meaningful.

This is consistent with the model's known behaviour rather than a bug in the
decomposition. Early/retinotopic visual structure is degraded by V-JEPA2's
spatial averaging, and movie-watching involves no motor task, so neither region
has strong readout structure to recover. **Early visual and motor assignments
should be treated as low-confidence** and not reported as findings.

A refinement worth trying if these systems are revisited: z-score each component
map before the argmax, so component scale cannot dominate the assignment.

## Honest-claims boundary

These are population-average predictions from a model trained on group fMRI, not
a measurement of any viewer's brain. Output should be described as a predicted
average cortical-response profile. Reliability is tiered in the JSON and those
tiers must survive into the UI.
