# Results and interpretation

## Main RSRD trade-off

EF-Elev clean MAE is 1.184 +/-0.041 cm, compared with 0.962 +/-0.021 cm for RoadBEV + Aug. Under sign-averaged pitch +/-5 degrees, the corresponding errors are 1.746 and 2.488 cm: degradations of about 0.562 and 1.526 cm from clean. The sign-averaged crossover appears between the measured 1- and 2-degree settings. This is not a Gaussian metadata-noise crossover and does not establish superiority at both individual signs. At roll +/-5 degrees, RoadBEV + Aug remains better, 1.036 versus 1.304 cm.

Signed, sign-averaged, updated-projection and metadata results are rebuilt in `results/`. The official primary table denominator is retained. No result is selected from a better epoch or from the most favorable repeated run.

## AFS and training ablations

| Configuration | Clean MAE (cm) | Mean of 16 IAS conditions (cm) |
|---|---:|---:|
| No AFS, native CST-Elev | 1.241 | 1.419 |
| No rotation or color augmentation | 1.381 | 1.846 |
| Plain CE | 1.176 | 1.336 |
| Full EF-Elev | 1.184 | 1.327 |

AFS improves the complete implementation comparison by 0.056 cm on clean input and 0.092 cm on the 16-condition mean. Core numerical settings and shared code were matched, but runtime entry points, worker counts and additional-module initialization differ. This is an implementation-level ablation; test-time branch removal is not a replacement for a separately trained no-AFS model.

Removing augmentation removes rotation and color together. The experiment supports the combined training recipe, rather than identifying the rotation-only effect. Plain CE has slightly better clean MAE than the composite loss, so the loss is not promoted as an established overall-accuracy improvement.

## Newly unified shape analysis

This package computes the same clean-frame shape metrics for six configurations and all three seeds from the already saved prediction arrays. Each configuration has 296 frames per seed. The estimator is documented in [PROTOCOL.md](PROTOCOL.md).

| Method | Deplaned MAE (cm) | Gradient MAE (cm per adjacent-cell difference) | Mean relief recovery ratio |
|---|---:|---:|---:|
| RoadBEV + Aug | 0.439 | 0.186 | 0.825 |
| CST-Elev, no AFS | 0.550 | 0.191 | 0.927 |
| EF-Elev | 0.511 | 0.191 | 0.775 |
| Plain CE | 0.532 | 0.193 | 0.779 |
| No augmentation | 0.575 | 0.196 | 0.992 |

AFS improves deplaned MAE, but does not improve the amplitude ratio toward one in this unified definition. A near-one ratio alone is also insufficient: the no-augmentation model has the largest deplaned error of the rows shown. The composite objective improves the deplaned error relative to plain CE in these clean caches, but the gradient difference is small and amplitude recovery is similar. These findings do not establish broad RASL robustness or faithful bump-peak reconstruction.

Historical pilot figures used other relief-region definitions. Their ratios must not be numerically compared with this all-valid-cell, per-frame p98 statistic. Selected profile images illustrate behavior; they do not replace the 296-frame shape table.

## Latest self-collected comparison

The latest fixed-988 comparison favors EF at the method level: 1.984 versus 2.250 cm overall, and 1.776 versus 1.968 cm on the 139-frame motion union. All 988 paired frame errors and all-frame improvement rankings are supplied. The 42-frame instantaneous subset, 120-frame recurrent context, and their union are not independent test sets.

The result is promising for the defined residual task, with the modalities, native training differences, single seed, one acquisition session and exploratory label construction stated explicitly. It cannot establish that AFS caused the difference or that the model generalizes to an independent route or camera mount.

## Older self-collected AFS analysis

In the separate 1,139-frame version, adding AFS raises clean MAE from 2.490 to 2.566 cm overall and from 2.020 to 2.089 cm on the 192-frame bump subset. At joint magnitude3, errors are 2.633 and 2.650 cm. The increase from clean is smaller with AFS, about 0.084 versus 0.143 cm, while absolute error remains larger at the measured settings.

The older fixed-checkpoint bag analysis is retained: the overall AFS-minus-noAFS difference is +0.07575 cm, conditional bag-resampling interval [+0.03161,+0.12612] cm. It is not evidence of a clean benefit. This version is not pooled with the latest 988-frame version.

## Anchors, positional encoding and real attitude

Existing inference interventions and separately trained controls have different evidential roles. Shuffling BEV anchor assignments changes clean MAE only slightly, while the recorded freeze and center substitutions increase error. The historical freeze averages heads before copying the mean back to every head, so it changes both image adaptation and head diversity.

The separately trained center/static/random anchor and REO-style position-encoding controls have only seed307. Fixed random anchors achieve clean MAE close to normal AFS, and the alternative position encoding does not collapse in that control. These results do not support universal anchor superiority, exact geometric image-to-cell correspondence, or a claim that the original encoding cannot work.

Projection-supervised PSA uses the original accepted `exp31_srefsup_multiseed_20261016` series. Additional PSA runs for the same seed numbers have different weights and scores; they are retained in a separate control file and are not substituted or counted as extra independent seeds. The nominally more geometric variant also trades clean accuracy against large-pitch performance.

Real INS training-tertile groups use pre-existing signed-angle thresholds: pitch17.02/17.3367 degrees and roll-0.26/0.5 degrees. They are not bins of absolute deviation from a mean. EF-Elev does not beat RoadBEV + Aug in these six groups, and road shape varies between frames, so these groups do not establish a causal effect of natural camera attitude.

## Historical efficiency evidence

The preserved 2026-10-16 record reports EF-Elev at 6.1561 M parameters, 161.37 counted GFLOPs and median23.47 ms, compared with RoadBEV-mono at 26.6463 M, 547.34 GFLOPs and18.60 ms. The measurements use 960x528, batch1, FP32, A100-SXM4-40GB, 20 warm-up iterations and 200 timed iterations. EF-Elev is smaller in this record but slower in latency.

This is a historical partial measurement, not a newly certified deployment benchmark: grid_sample/gather are excluded from FLOP counting, and the baseline uses random projection indices. The reported AFS parameter increment is about0.0643 M. The original script also lists positional-table counts separately; parameter totals require a renewed implementation audit before being used as a headline efficiency claim. No FastRSR end-to-end efficiency result is fabricated.
