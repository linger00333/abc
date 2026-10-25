# Acceptance report

Evidence snapshot: 2026-10-25. The machine-readable package result is [verification_report.json](../verification_report.json); the immutable file inventory is [SHA256SUMS.json](../SHA256SUMS.json).

## Completed checks

- Rebuilt all318 formal RSRD cached conditions from preserved frame records. Maximum MAE/RMSE/GT0.5 source difference is below1e-6; the observed maximum across these metrics is approximately9.95e-14.
- Reconstructed288 additional Oracle/metadata conditions from source-bound frame tables.
- Recomputed three clean shape metrics from saved prediction arrays for six configurations, three seeds and296 frames per seed. No model inference was used.
- Verified identical latest988 test IDs, the42/120/139 subsets and23-frame overlap, and the disjoint2552/1205/988 split lists.
- Verified qualitative array MAEs against the full-test frame records, shared scales, retained GT holes and explicit epoch/selection labels.
- Tested strict0.5 cm threshold behavior, invalid predictions, equal-frame weighting, seed-level condition averaging, missing-condition rejection, independent plane removal, exact prediction ID ordering, and preservation of decimal sample IDs in exported filenames.
- Rebuilt the release tables and figures in the isolated pinned offline environment. Source and generated file hashes are preserved; PDF timestamps are omitted to permit deterministic same-environment reconstruction.
- Scanned filenames and text, including decoded JSON strings, PDF text/metadata and PNG text metadata, for non-English scripts. Reviewed representative rendered panels and raw image previews. No credential material or full checkpoint/data archive is included.
- Compared the portable baseline voxel projection with the original grid/index expressions on three frames: zero differing indices. This was CPU-only and made zero model forward calls.

## Explicit validation limits

No full checkpoint replay, new GPU inference, training or new acquisition was performed. Existing checkpoint hashes and original inference reports are provenance, not evidence that every model was rerun during packaging. Portable clean replay is supplied for RSRD methods and self-collected monocular EF; modified self-collected stereo RoadBEV is evaluated through existing prediction caches.

The RSRD test metadata does not resolve true acquisition-session/route membership. Its five dates are not five proven independent routes. Latest self-collected resampling is conditional within one acquisition session. Source comments and historical efficiency records may describe older pilot assumptions; qualified release documentation takes precedence.

GitHub upload and private visibility are checked after the local release passes. Repository identity and the final commit are delivered with the task completion message.
