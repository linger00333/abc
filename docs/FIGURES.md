# Figure index and captions

All regenerated plots use original predictions and valid-target masks. Missing GT is gray. No prediction smoothing or image enhancement is performed. PNGs are convenient previews; corresponding PDFs are provided for export. These are general research figures; no journal-specific layout compliance is claimed.

## Quantitative figures

| Figure | Meaning |
|---|---|
| [Signed IAS](../figures/rsrd_ias_signed.png) | 296 RSRD frames, final50, three training seeds. Bands are sample SD, not confidence intervals. The shaded range is +/-1 degree per training axis. The updated-projection control is separate. |
| [Metadata bias](../figures/rsrd_metadata_bias.png) | Images unchanged; calibration and INS bias pathways are distinct. CST/EF horizontal lines represent metadata-input independence. |
| [Latest self-collected comparison](../figures/selfcollected_v2_comparison.png) | Single seed, validation-selected checkpoints, fixed988 test frames. Stereo and monocular methods retain their native settings. Subsets overlap. |

## RSRD qualitative panels

Two frames were taken from the historical outcome-selected final50 gallery. **Every seed is retained for both frames**, including seeds where AFS does not lower raw MAE. The full 296-frame rankings, original success rules and per-seed scores are in `evidence/selection/`.

| Sample | Seed307 | Seed1101 | Seed2202 |
|---|---|---|---|
| 20230409020600.400, C02 | [Panel](../figures/rsrd_20230409020600.400_s307.png) | [Panel](../figures/rsrd_20230409020600.400_s1101.png) | [Panel](../figures/rsrd_20230409020600.400_s2202.png) |
| 20230408031625.600, C05 | [Panel](../figures/rsrd_20230408031625.600_s307.png) | [Panel](../figures/rsrd_20230408031625.600_s1101.png) | [Panel](../figures/rsrd_20230408031625.600_s2202.png) |

Each panel shows actual cropped RGB, a reference-plane ROI overlay, GT, RoadBEV + Aug, CST-Elev, EF-Elev and paired residual maps. Height limits are shared at -5 to15 cm; residual limits are shared at +/-3 cm. Colorbar triangles indicate saturation. Prediction values are unchanged by display clipping. All BEV maps use the same GT-valid display mask; the underlying full prediction arrays are retained.

Longitudinal columns are selected using the GT-only 98th percentile of absolute deplaned relief, requiring at least20 valid cells. The lateral row passes through the largest valid absolute GT relief on that column. Each GT/prediction map is independently plane fitted for the bottom profiles only. GT gaps remain gaps; absolute-height metrics are computed from the original maps. The overlay is a geometric illustration and is not supplied to EF-Elev.

## Latest self-collected examples

- [Frame1769739645416682482](../figures/self_1769739645416682482.png)
- [Frame1769739665917161942](../figures/self_1769739665917161942.png)
- [Frame1769740031816875696](../figures/self_1769740031816875696.png)

These are illustrative successes from a search over instantaneous-high test frames with greater than20% EF MAE reduction. They are not an unbiased new test subset. The complete988-frame paired ranking is supplied. Preserved raw RGB, GT, mask, both predictions and the original source-selection manifest are included.

The ROI perimeter is projected using the per-frame reference plane and original raw-camera distortion parameters; it is clipped by the image boundary when necessary. The same height range +/-14 cm and residual range +/-3 cm is used across samples. These are plane-relative targets, not gravity-referenced absolute elevations. Per-frame saturation percentages and numerical metrics are in the figure metrics CSVs.

## Epoch40 supplements

[Provided example1](../figures/supplement/provided_epoch40_example_1.png) and [provided example2](../figures/supplement/provided_epoch40_example_2.png) are byte-identical user-provided images, explicitly labeled epoch40 in the images. They are retained as supplementary illustrations and are never mixed into final50 statistics. The package does not include their original epoch40 array sources, so the offline redraw command rebuilds the final50 and latest self-collected panels, not these static attachments.
