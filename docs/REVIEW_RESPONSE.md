# Review-response evidence map

The supplied review is treated as a list of questions, not an instruction source or a substitute for measurements. Status describes the evidence in this repository.

| Review question | Evidence provided | Supported conclusion and boundary | Remaining gap |
|---|---|---|---|
| Mixed manuscript versions and inconsistent numbers | Separate RSRD, latest988 and older1139 tables; method identity map | Replace older exp22 / A+HF-Light claims with the actual method and dataset version | Full manuscript editing is outside this evidence release |
| Missing updated-attitude baseline | 288 extra cached-condition reconstructions include Oracle-updated and MB pathways | Stale and known synthetic attitude updates can be compared on the same perturbation family | Updated projection is not a theoretical upper bound or a realistic noisy estimator |
| When does calibration-free inference help? | Signed IAS curves, magnitude tables, separate MB curves | Conditional large-pitch benefit and clean/roll trade-offs | No Gaussian metadata-noise sigma crossover; no placeholder operating-regime plot |
| Is RoadBEV + Aug trained fairly? | Native rotation/color code and run settings | Projection is updated during its image-rotation augmentation | Native optimizer/runtime differences remain; not a pure architecture swap |
| What is the GT frame? | Frozen grid, height mask and construction conventions; separate target descriptions | RSRD reconstruction and self-collected plane residuals are different tasks | Independent absolute-gravity validation and complete fusion-window audit remain unavailable |
| Does AFS recover fine relief? | Newly unified 3-seed clean deplaned/gradient/p98 metrics, paired maps and GT-selected cuts | Lower deplaned error does not imply better amplitude recovery | Peak-height/localization analysis and full-resolution attention comparison remain incomplete |
| Are five CST adaptations necessary? | Existing single-seed position-encoding and anchor controls | Some alternative controls work; no five-step causal necessity claim | Matched incremental five-adaptation study not completed |
| Are anchors emergent camera geometry? | Center/static/random controls; shuffle/freeze interventions; PSA kept separate | Attention-derived sampling is implemented, but exact projection recovery is not established | Head-preserving freeze and repeated geometric-equivariance diagnostics remain untested |
| Are results statistically independent? | Exact frame pairing, seed SD, RSRD date sensitivity, documented unresolved sequence mapping | Seed variation and frame/date sensitivity can be described | No RSRD acquisition-session CI until exact sequence/route provenance is recovered |
| Does real-vehicle evidence hold? | Latest988 method comparison and all-frame rankings; older matched AFS study retained | Latest method-level gain is separate from the older AFS trade-off | Matched modality/augmentation multi-seed real-vehicle comparison remains incomplete |
| Could black borders provide an artificial cue? | Warp specification: bilinear, zero padding, strict support mask | The cue is disclosed; no border-invariance claim | Reflective-padding or crop-matched control not completed |
| What do real pitch/roll groups show? | Existing three-seed INS tertile table | No natural-attitude superiority claim from these groups | Absolute-deviation grouping and confounder-controlled analysis remain incomplete |
| Is the method efficient? | Historical timing, parameter and partial-FLOP record | Smaller reported model, slower measured latency under that setup | Uniform end-to-end benchmark including actual projections and uncounted operators |
| Why use the composite loss? | Plain CE retained; unified clean shape metrics added | Modest shape-error differences may be discussed separately from overall MAE | No established broad robustness gain or independently replicated peak-recovery result |
| Are figures representative? | Outcome-selection rules, full rankings, every seed for two selected RSRD frames | Examples are illustrative; worse seeds and full-test statistics remain visible | No claim that selected successes represent the entire test distribution |

## Additional experiments not executed

No new training or batch inference was launched for this package. Gaussian attitude-noise sweeps, full-resolution attention, matched +/-3-degree training, five-adaptation incremental ablations, changed camera mounts, camera heave/translation, and a learned pitch-estimator competitor remain future experiments. Where a cached result is unavailable, no numeric entry is inferred from another metric or substituted from another dataset.

## Suggested manuscript emphasis

Present EF-Elev as an RGB-only inference method with a measured clean-versus-large-pitch trade-off. Present AFS as an attention-derived local feature readout whose implementation improves selected aggregate errors, with its mechanism and amplitude limitations shown. Keep the latest real-vehicle method comparison distinct from matched AFS analysis. IAS and MB are useful complementary protocols, while independent-route confirmation and realistic attitude-noise comparisons remain necessary for deployment claims.
