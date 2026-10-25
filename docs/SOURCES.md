# Primary sources and attribution

External references explain background methods. They are not independent confirmations of EF-Elev's experimental results. Access checks were performed on2026-10-25.

| Source | Relevance |
|---|---|
| Zhao et al., [RoadBEV: Road Surface Reconstruction in Bird's Eye View](https://arxiv.org/abs/2404.06605),2024; [official code](https://github.com/ztsrxh/RoadBEV) | Explicit BEV elevation baselines and model components |
| Zhao et al., [RSRD dataset paper](https://doi.org/10.1038/s41597-024-03261-9), Scientific Data,2024; [official development toolkit](https://github.com/ztsrxh/RSRD_dev_toolkit) | Public dataset, calibration and motion metadata |
| Zhuang et al., [Robust 3D Semantic Occupancy Prediction with Calibration-free Spatial Transformation](https://arxiv.org/abs/2411.12177),2024; [official English documentation](https://github.com/ICEORY/REO/blob/master/README_EN.md) | Calibration-free attention transformation; occupancy work adapted to a different elevation task |
| Zhu et al., [Deformable DETR](https://arxiv.org/abs/2010.04159), ICLR2021 | Local sampling around reference points; related architectural mechanism |
| Sun et al., [Integral Human Pose Regression](https://openaccess.thecvf.com/content_ECCV_2018/html/Xiao_Sun_Integral_Human_Pose_ECCV_2018_paper.html), ECCV2018 | Differentiable coordinate expectation |
| Nibali et al., [Numerical Coordinate Regression with Convolutional Neural Networks](https://arxiv.org/abs/1801.07372),2018 | Coordinate expectation and DSNT context |
| Jaderberg et al., [Spatial Transformer Networks](https://arxiv.org/abs/1506.02025),2015 | Differentiable image sampling |
| Dao et al., [FlashAttention](https://arxiv.org/abs/2205.14135),2022 | IO-aware exact attention without materializing full attention matrices |
| [Official PyTorch previous-version installation instructions](https://pytorch.org/get-started/previous-versions/) | CUDA11.8 installation pairing for torch2.6 / torchvision0.21 |

The draft's later journal metadata for REO is not copied as verified bibliographic information. The linked author preprint provides an identifiable primary source. No novelty-priority claim is inferred from an incomplete bibliography.

## Code and assets

RoadBEV-derived model components retain the [RoadBEV MIT notice](../licenses/RoadBEV-MIT.txt). REO's upstream notice is preserved in [REO-MIT.txt](../licenses/REO-MIT.txt). Torchvision-derived EfficientNet/MobileNet components retain the [Torchvision BSD notice](../licenses/Torchvision-BSD.txt). These notices do not grant rights to the datasets or the unpublished method's original material beyond their own scope.

Selected RSRD imagery remains subject to the [dataset access terms](https://thu-rsxd.com/dxhdiefb/), which identify noncommercial CC BY-NC use. Acquire the full data from the dataset owner. The full public dataset, self-collected dataset and model checkpoints are not redistributed in this repository. Compact arrays and selected images document the authorized private research evidence package; do not assume a new blanket dataset license.

The repository preserves the user's original experimental artifacts and uses English prose, filenames, labels and commit messages. Source-import hashes distinguish byte-identical native components from the isolated self-collected adapter. Historical source comments record earlier pilot interpretations; the later controlled results and qualified explanations in this release take precedence over those comments.

## Artifact provenance

`evidence/provenance/import_manifest.json` records copied result/figure assets, their original source identifiers and SHA256 values. `clean_cache_bindings.json`, `dense_seed_metrics.csv` and the supplementary source-binding CSV connect table records with prediction/checkpoint sources. The native checkpoint config wrappers retain their original paths only as provenance.

The selected RSRD arrays were exported in memory from immutable native caches. The export executed CPU-only shape calculations and made no model forward calls or remote file changes. The original metadata files were inspected, but no exact acquisition-sequence field was available for the test IDs; location coordinates were not redistributed.
