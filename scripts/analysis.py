"""Metric definitions shared by reconstruction, figures and acceptance checks."""
from collections import defaultdict
import statistics

import numpy as np


def frame_metrics(gt, pred, mask):
    gt, pred = np.asarray(gt, dtype=float), np.asarray(pred, dtype=float)
    valid = np.asarray(mask, dtype=bool) & np.isfinite(gt)
    if gt.shape != pred.shape or valid.shape != gt.shape:
        raise ValueError('Ground truth, prediction and mask shapes must match.')
    if not valid.any() or not np.isfinite(pred[valid]).all():
        raise ValueError('Evaluation requires valid targets and finite predictions.')
    error = pred[valid] - gt[valid]
    return dict(valid_cells=int(valid.sum()), MAE_cm=float(np.abs(error).mean()),
                RMSE_cm=float(np.sqrt(np.square(error).mean())),
                GT0p5_pct=float(100 * (np.abs(error) > .5).mean()))


def paired_seed_summary(rows, metric):
    grouped = defaultdict(dict)
    for row in rows:
        seed, condition = int(row['seed']), row['condition']
        if condition in grouped[seed]:
            raise ValueError('Duplicate seed and condition.')
        grouped[seed][condition] = float(row[metric])
    if not grouped or len({frozenset(values) for values in grouped.values()}) != 1:
        raise ValueError('Every seed must contain the same conditions.')
    per_seed = {seed: statistics.mean(values.values()) for seed, values in grouped.items()}
    values = list(per_seed.values())
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else None, per_seed


def deplane(values, mask):
    values = np.asarray(values, dtype=float)
    mask = np.asarray(mask, bool) & np.isfinite(values)
    z, x = np.indices(values.shape)
    design = np.stack([x, z, np.ones_like(x)], axis=-1).astype(float)
    if mask.sum() < 3 or np.linalg.matrix_rank(design[mask]) < 3:
        raise ValueError('Plane fitting requires at least three non-collinear cells.')
    coefficients = np.linalg.lstsq(design[mask], values[mask], rcond=None)[0]
    return values - design @ coefficients


def shape_metrics(gt, pred, mask):
    valid = np.asarray(mask, bool) & np.isfinite(gt)
    frame_metrics(gt, pred, valid)
    target, estimate = deplane(gt, valid), deplane(pred, valid)
    gt_amp = float(np.percentile(np.abs(target[valid]), 98))
    pred_amp = float(np.percentile(np.abs(estimate[valid]), 98))
    gradient_errors = []
    for axis in (0, 1):
        adjacent = (valid[1:, :] & valid[:-1, :]) if axis == 0 else (valid[:, 1:] & valid[:, :-1])
        delta = np.diff(np.asarray(pred, float), axis=axis) - np.diff(np.asarray(gt, float), axis=axis)
        gradient_errors.extend(np.abs(delta[adjacent]).tolist())
    return dict(deplaned_MAE_cm=float(np.abs(target[valid] - estimate[valid]).mean()),
                gradient_MAE_cm_per_cell=float(np.mean(gradient_errors)) if gradient_errors else None,
                GT_relief_p98_cm=gt_amp, prediction_relief_p98_cm=pred_amp,
                relief_recovery_ratio=pred_amp / gt_amp if gt_amp > 1e-8 else None)
