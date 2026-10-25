# Reproduction guide

## Offline tables, statistics and figures

The offline package was tested in a fresh Python3.12 environment with the pinned `requirements-lock.txt`. All frame-level evidence and compact qualitative arrays needed by this path are included.

```bash
python -m pip install -r requirements-lock.txt
python -m unittest discover -s tests -v
python -m scripts.build_tables
python -m scripts.paired_analysis
python -m scripts.render_figures
python -m scripts.verify_release
```

`build_tables` checks 318 formal cached conditions against the preserved source metrics, reconstructs 288 supplementary Oracle/MB conditions, and keeps self-collected versions separate. `paired_analysis` creates the complete latest988 improvement ranking and conditional bag-resampling sensitivity. `render_figures` never reads checkpoints or uses a network.

For reconstruction in another output directory, use `python -m scripts.build_tables --output build/tables` and `python -m scripts.render_figures --output build/figures`. The figure script reads the checked-in `results/` tables. PNGs and PDFs are both exported; PDF metadata timestamps are not expected to be byte-identical between renders.

## Evaluate your saved prediction cache

This mode supports any method on any of the three supplied dataset versions. It requires no PyTorch. Supply a JSON manifest containing unique IDs, in the exact order of the prediction cache:

```json
{
  "samples": [
    {
      "sample_id": "20230409020600.400",
      "image_path": "test/left_half/20230409020600.400.png",
      "gt_path": "gt/test/20230409020600.400.npz"
    }
  ]
}
```

Paths inside the manifest are resolved against `--data-root`. `image_path` is optional for cache scoring. Each GT NPZ contains `ele_gt` / `ele_mask` or `gt` / `mask`. The prediction NPZ contains `sample_ids` and `predictions`, with no object arrays. The shapes must match exactly. Convert trusted native pickle GT to this non-pickle array format before use.

```bash
python -m scripts.evaluate --dataset rsrd --model S_free \
  --data-root /path/to/data --manifest /path/to/manifest.json \
  --predictions /path/to/predictions_cm.npz --output evaluation_output/cache
```

Use `--dataset selfcollected-v1` or `selfcollected-v2` for the 25 cm height limit. RSRD uses20 cm. Output is `per_frame.csv` plus `summary.json`; every frame has equal weight. The evaluator rejects missing, duplicated or reordered IDs and invalid predictions. `--check-only` checks input manifests/GT without model inference.

## Optional checkpoint replay

Replay is supplied for the RSRD monocular explicit baselines, CST-Elev and EF-Elev variants, and for monocular EF on the self-collected grid. Stereo self-collected RoadBEV is supported through saved-cache evaluation; its modified native training/inference project is not repackaged as a portable stereo deployment system.

Replay loads only a checkpoint whose SHA256 matches the frozen experiment inventory, with strict state-dictionary loading. Preserve the original full checkpoint bytes. The separate weights-only archives have different hashes and are not interchangeable with the inventoried full checkpoints. Large checkpoints are not distributed here.

Use an isolated native inference environment. The inspected A100 environment reports Python3.10, PyTorch2.6.0+cu118, torchvision0.21.0+cu118, NumPy1.26.4 and Pillow12.2.0. These are separate from the offline plotting environment. Historical BF16 cache reproducibility is sensitive to batch size and kernel/runtime choices; the package does not claim new end-to-end bitwise replay validation.

```bash
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements-inference.txt
python -m scripts.evaluate --dataset rsrd --model S_free \
  --data-root /path/to/data --manifest /path/to/manifest.json \
  --checkpoint /path/to/final_epoch50.ckpt --config configs/S_free_s307.json \
  --output evaluation_output/ef
```

For CST-Elev use `--model exp21 --config configs/CST_s307.json`. For RoadBEV + Aug use `--model baseline_aug`; it reconstructs the native float32 voxel projection from each GT cache's K and r_vert2cam, so those two arrays are required. The native projection helper was compared with the original grid/index expressions on three frames, with zero differing indices and no model forward calls.

For the latest monocular EF use `--dataset selfcollected-v2 --model S_free --config configs/EF_s307.json`. The clean preprocessing crops raw RGB at (424,920,2024,1800), resizes to960x528 with the native PIL default, then uses ToTensor/Normalize-equivalent arithmetic. RSRD attention input uses the top960x528 crop and `/127.5-1`; explicit baseline normalization follows its native ToTensor/Normalize path.

RSRD explicit baselines use batch1/FP32; attention models use batch8/BF16; self-collected EF uses batch1/FP32. Set CUDA TF32 flags to the historical settings when investigating parity: the self-collected reference record has matmul TF32 disabled and cuDNN TF32 enabled. The wrapper reproduces clean inputs only. Synthetic IAS/MB source logic is preserved in the geometry module and protocol documents; no new sweep is launched by a package check.

Historical configs are retained verbatim inside provenance wrappers. Their absolute training paths are records, not required paths on the new machine: the evaluation interface uses the supplied data root and manifest, and does not initialize pretrained weights before loading a task checkpoint. The source modules still contain unused historical training helpers; only the documented release interfaces are portable.

## What was validated during packaging

The offline numerical, pairing, export, language and integrity checks were run. Original source/array hashes and checkpoint identifiers are recorded. CPU-only native projection checks were run. No new GPU model forward, training update, or batch inference was performed. A future full replay must establish parity against the supplied frame records in its chosen runtime before its outputs replace canonical results.
