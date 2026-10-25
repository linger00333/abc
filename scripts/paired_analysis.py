"""Conditional bag-resampling analysis for the latest fixed-checkpoint comparison."""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from scripts.build_tables import read_csv, write_csv

ROOT = Path(__file__).resolve().parents[1]


def main():
    frames = read_csv(ROOT / 'evidence/per_frame/selfcollected_v2.csv')
    ef = {r['sample_id']: r for r in frames if r['model'] == 'S_free'}
    road = {r['sample_id']: r for r in frames if r['model'] == 'baseline_aug'}
    assert set(ef) == set(road) and len(ef) == 988
    rankings = []
    for sid, row in ef.items():
        a, b = float(road[sid]['MAE_cm']), float(row['MAE_cm'])
        rankings.append(dict(sample_id=sid, bag=row['bag'], instantaneous_high=row['instantaneous_high'],
            recurrent_high_segments=row['recurrent_high_segments'], baseline_MAE_cm=a, EF_MAE_cm=b,
            EF_minus_baseline_cm=b-a, EF_reduction_pct=100*(a-b)/a))
    rankings.sort(key=lambda r: (-r['EF_reduction_pct'], r['sample_id']))
    write_csv(ROOT / 'evidence/selection/selfcollected_v2_all_frame_rankings.csv', rankings)
    bags = sorted({r['bag'] for r in ef.values()})
    results = []
    for subset in ('all', 'instantaneous_high', 'recurrent_high_segments', 'bump_union'):
        sums, counts = np.zeros(len(bags)), np.zeros(len(bags))
        for sid, row in ef.items():
            if subset != 'all' and row[subset] != 'True':
                continue
            i = bags.index(row['bag'])
            sums[i] += float(row['MAE_cm']) - float(road[sid]['MAE_cm'])
            counts[i] += 1
        rng = np.random.default_rng(0)
        indices = rng.integers(0, len(bags), size=(10000, len(bags)))
        denominator = counts[indices].sum(1)
        valid = denominator > 0
        delta = sums[indices][valid].sum(1) / denominator[valid]
        lo, hi = np.quantile(delta, [.025, .975])
        results.append(dict(subset=subset, frames=int(counts.sum()), bags_with_subset=int((counts > 0).sum()),
            delta_EF_minus_RoadBEV_cm=float(sums.sum()/counts.sum()), CI95_low_cm=float(lo), CI95_high_cm=float(hi),
            resamples=10000, valid_resamples=int(valid.sum()), random_seed=0,
            scope='Conditional fixed-checkpoint bag resampling within one test acquisition; not route/session or training-seed uncertainty.'))
    write_csv(ROOT / 'results/selfcollected_v2_paired_bag_sensitivity.csv', results)
    print(json.dumps(dict(status='passed', paired_frames=988, bags=len(bags), bootstrap_scope='within-session sensitivity only')))


if __name__ == '__main__':
    main()
