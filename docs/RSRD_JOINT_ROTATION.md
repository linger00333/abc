# RSRD-Dense Joint Pitch-Roll Perturbation

This release adds the completed joint pitch-roll evaluation on all 296
RSRD-dense test frames. It uses fixed epoch-50 checkpoints from three training
seeds (307, 1101 and 2202). No new training or test-set checkpoint selection
was performed.

## Protocol

For each per-axis magnitude `a` in `{1, 2, 3, 5}` degrees, the evaluation
enumerates all four sign combinations:

```text
(-a, -a), (-a, +a), (+a, -a), (+a, +a)
```

The ordered camera rotation is `Rz(roll) @ Rx(pitch)`. A single homography
warps the image, while the gravity-aligned elevation target remains unchanged.
RoadBEV and RoadBEV+Aug retain their stale projection. EF-Elev reads RGB only.

The reported magnitude is the angle on each axis, not the combined rotation
magnitude. For small angles, the combined magnitude is approximately
`sqrt(2) * a`; therefore, the per-axis 5-degree condition must not be described
as a total 5-degree rotation.

Within each seed, metrics are averaged equally over the four sign
combinations. The table then reports the mean and sample SD across the three
training seeds. MAE and RMSE are in centimeters. GT0.5 is the percentage of
valid cells whose absolute error is greater than 0.5 cm; all metrics are lower
when better.

## Full-mask MAE

| Method | Clean | Per-axis 1 deg | Per-axis 2 deg | Per-axis 3 deg | Per-axis 5 deg |
|---|---:|---:|---:|---:|---:|
| RoadBEV | 1.102 +/- 0.062 | 1.346 +/- 0.066 | 1.661 +/- 0.107 | 1.925 +/- 0.085 | 2.559 +/- 0.158 |
| RoadBEV+Aug | **0.962 +/- 0.021** | **1.108 +/- 0.044** | 1.505 +/- 0.120 | 1.972 +/- 0.169 | 2.499 +/- 0.170 |
| EF-Elev | 1.184 +/- 0.041 | 1.216 +/- 0.050 | **1.331 +/- 0.018** | **1.477 +/- 0.030** | **1.814 +/- 0.010** |

EF-Elev is less accurate on clean input and at the per-axis 1-degree
condition, but has lower MAE than the stale-projection baselines from the
per-axis 2-degree condition onward. This comparison describes the fixed
checkpoints and synthetic pure-rotation protocol; it does not establish a
general real-driving threshold.

## Visibility scopes

Three evaluation scopes are retained in the released tables:

- `full`: the original valid target mask, including cells that leave image
  support under a perturbation;
- `visible`: cells supported in the current perturbed image;
- `common_joint17`: the fixed intersection visible across clean input and all
  16 joint conditions.

The common mask retains 89.2989% of the original valid target cells. The four
sign combinations retain 99.9960%, 99.9625%, 99.7323% and 96.0317% on average
at per-axis magnitudes 1, 2, 3 and 5 degrees, respectively.

## Released records

- [Aggregated metrics](../results/rsrd_joint_rotation/summary_metrics.csv)
  preserve the three scopes, five magnitude levels and three metrics.
- [Per-seed metrics](../results/rsrd_joint_rotation/seed_metrics.csv) preserve
  the values used to compute every reported mean and sample SD.
- [Visibility](../results/rsrd_joint_rotation/visibility.csv) records support
  retention for each of the 16 signed conditions.
- [Relative improvements](../results/rsrd_joint_rotation/relative_improvement.csv)
  compare EF-Elev with both explicit baselines.
- [Frozen protocol](../evidence/protocols/rsrd_joint_rotation/protocol.json)
  records rotations, masks, seeds, models and source hashes.
- [Release provenance](../evidence/provenance/rsrd_joint_rotation/release.json)
  records source-file hashes and the publication time zone.

The release excludes images, predictions, datasets and checkpoints. The
acceptance record confirms 144 prediction groups (nine model-seed pairs by 16
joint conditions), 42,624 input-hash checks and clean-prediction parity for all
nine fixed checkpoints.
