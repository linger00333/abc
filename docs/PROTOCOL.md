# Protocol and evidence boundaries

## Method identities and input dependencies

| Release name | Historical identifier | Forward inputs | Role |
|---|---|---|---|
| RoadBEV | baseline | RGB and calibration/attitude-derived projection indices | Explicit monocular RSRD reference |
| RoadBEV + Aug | baseline_aug | Same projection inputs; training image augmentation | Strong explicit RSRD reference |
| CST-Elev | exp21 / A_HF_light | RGB | Native implementation without AFS |
| EF-Elev | S_free | RGB | Main method; attention anchors; no projection supervision |
| No augmentation | A2 | RGB | Rotation and color augmentation both removed |
| Plain CE | A3 | RGB | Relief weighting, gradient and deplaned-loss additions removed |
| Projection-supervised variant | original S_refsup / PSA | RGB at inference; K/R supervision during training | Supplementary variant, not the main method |

Inference independence does not imply calibration-free data construction or training augmentation. Geometry is used to construct targets, synthesize rotations and exclude training cells outside the input view. Historical source modules also contain inactive geometric variants; the main S_free configuration does not enable them.

## RSRD-dense

The physical candidate pool contains 2,493 training and 300 test frames. Excluding 25 and 4 candidates without the exact required pose leaves 2,468 and 296. No additional validation split was used for these runs; the formal comparison uses final epoch50 for every seed, rather than a test-selected or training-loss-selected checkpoint.

The BEV grid has 164 longitudinal by 64 lateral cells at 3 cm spacing: x from -1 to 0.92 m, z from 2.16 to 7.08 m. Rows run from far to near. The reference camera height is 1.1 m. Targets follow the frozen gravity-frame reconstruction and original calibration/pose conventions. The valid evaluation set is `ele_mask & (abs(ele_gt) < 20 cm)`; no GT holes are filled. This package preserves the existing targets and does not independently certify their absolute gravity alignment.

The fixed `geometry_fixed` matrices in some configurations have an older training1210 provenance. They must not be described as a fresh mean over all 2,468 frames or confused with per-frame GT construction. The active A-only training path reads per-frame cached K/R for image augmentation and visibility checks.

### Training and test transformations

The augmented explicit baseline rotates the image with `H = K deltaR K^-1` and **updates projection to deltaR @ R during training**. Training GT values remain unchanged, and newly invisible supervision cells are masked. CST-Elev / EF-Elev use the matched rotation and color augmentation family: probability 0.5, pitch and roll sampled independently within +/-1 degree; color jitter probability 0.5 and factors 0.9 to 1.1. Native optimizers, schedules and other implementation differences are retained in configs; not every cross-method comparison is single-factor controlled.

IAS rotates the actual input while holding the target fixed. A stale baseline retains the original projection; an updated-projection control uses the known synthetic rotation. The control is not a theoretical error bound: image support, clipping and sampling still matter. Signed pitch/roll values are -5, -3, -2, -1, +1, +2, +3 and +5 degrees. For magnitude tables, signs are averaged within each seed first.

MB leaves the image unchanged and changes calibration rotation (`calR`) or INS fields. These are distinct metadata pathways. The calibration-free curves are clean-prediction dependency controls, not independent repeated image-robustness measurements.

The native warp resamples the full half-resolution source before cropping to 960x528. It uses bilinear interpolation, zero padding and a strict support mask; zero rotation bypasses interpolation. Black borders can reveal rotation. No reflective-padding control has been established by this release. Formal metrics retain their original valid GT mask; an existing common-visibility sensitivity analysis is not mixed into the primary denominator.

### Runtime correction

Native RSRD baseline evaluation workers use **batch1, FP32**. CST-Elev and EF-Elev canonical attention caches use **batch8, BF16**. A historical metric-export script stamped `inference_batch=8` on every method; the preserved source CSV is unchanged, and this document plus `evidence/provenance/adapter_verification.json` supplies the correction. BF16 attention batch size must not be reduced for canonical replay.

## Self-collected versions

| Version | Train / validation / test | Checkpoint rule | Purpose |
|---|---|---|---|
| `data1_half_bump_v1_20261023` | 2552 / 1205 / 988 | Lowest complete validation MAE over 30 epochs, earliest tie | Latest method-level comparison |
| `selfcollected_v1_1139` | 3506 / 923 / 1139 | Lowest clean validation MAE | Matched no-AFS / AFS supplement |

The latest test compares stereo RoadBEV + Aug, validation epoch20, with monocular EF, epoch11. Training batch sizes are 4 and 3; native augmentations also differ. Its labels are per-frame plane-relative road residuals on a 200x80 grid covering x +/-1.2 m and z 4-10 m. A gravity-referenced RSRD error cannot be pooled with this residual-target error. Both latest methods share identical split metadata and 988 test IDs, and their checkpoint hashes were frozen before test evaluation.

Latest training was reduced from 5,103 to 2,552 frames before the compared runs. It retains the predefined motion-enriched training union and samples remaining background by bag/time-block stratification. Thirty epochs therefore do not represent the same optimizer-update budget as the older 5,103-frame training. No held-out frame was moved into the new training set.

The fixed motion proxy is `peak_xy > 0.3744372541341072 rad/s OR path_xy > 1.1249450202979638 degrees`. Recurrent context uses 20-second blocks with at least 36 frames, at least 85% valid motion records, at least 10% high-motion valid records and high motion in at least three distinct seconds. Test counts are 42 instantaneous-high, 120 recurrent-context and 139 union frames, with 23 overlapping frames. These are motion proxies, not independently annotated pavement roughness.

The older AFS comparison uses validation epochs13 (no AFS) and14 (AFS), seed307, FP32. Its joint_1/2/3 conditions are random-direction pitch/roll magnitudes from the original sweep, not the RSRD single-axis protocol. Older experiment exp22 and the latest half-size EF run must not be substituted for this matched AFS pair.

## Metrics and uncertainty

Each frame contributes equally, regardless of its number of valid cells:

- MAE: average absolute valid-cell error, in cm.
- RMSE: square root of mean squared valid-cell error **within each frame**, then average frames.
- GT0.5: percentage of valid cells with absolute error **strictly greater than 0.5 cm**, then average frames. Lower is better. A historical field named `gt0p5cm` represented the complementary accuracy in one self-collected adapter; the unified records use the error percentage.

Three-seed SD uses `ddof=1`. The 16 IAS conditions are averaged inside each seed; they are not 16 independent training repetitions.

Shape analysis independently fits an ordinary least-squares plane to GT and each prediction on the same valid mask. It reports deplaned MAE, adjacent-cell gradient MAE (cm per cell difference, not slope per meter), and the ratio of prediction to GT 98th-percentile absolute deplaned amplitude. Ratios are averaged across frames and then seeds. Undefined near-zero GT amplitudes are not replaced with arbitrary finite ratios.

RSRD IDs span 2023-03-17, 2023-03-21, 2023-04-06, 2023-04-08 and 2023-04-09, with 23/10/23/119/121 test frames. The available source metadata union has no exact test-ID sequence mapping. `official_test` and `test` are placeholders. Dates do not establish independent sessions or routes, so this package supplies descriptive date sensitivity, not acquisition-session CIs. An earlier audit found 273/296 test timestamps within one second of a training timestamp; this establishes temporal proximity, not proof of label-fusion leakage.

Latest self-collected bag resampling uses 10,000 paired resamples, seed0, within the one test acquisition. Its intervals describe conditional fixed-checkpoint bag sensitivity. They do not establish independent-route, acquisition-session or training-seed uncertainty. Test feedback was available during earlier method development, so these remain exploratory results.
