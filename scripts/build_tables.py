"""Rebuild tables from preserved frame records; no models or network required."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics

from scripts.analysis import paired_seed_summary

ROOT = Path(__file__).resolve().parents[1]
METRICS = ('MAE_cm', 'RMSE_cm', 'GT0p5_pct')
SHAPE = ('deplaned_MAE_cm', 'gradient_MAE_cm_per_cell', 'relief_recovery_ratio')
MAIN = ('baseline', 'baseline_aug', 'exp21', 'S_free')


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def frame_groups(rows, metrics=METRICS, expected_frames=None):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['model'], int(row['seed']), row['condition'])].append(row)
    output = []
    for (model, seed, condition), frames in sorted(grouped.items()):
        ids = [r['sample_id'] for r in frames]
        if len(set(ids)) != len(ids) or (expected_frames is not None and len(ids) != expected_frames):
            raise ValueError(f'Invalid frame list: {model}, {seed}, {condition}')
        item = dict(model=model, seed=seed, condition=condition, frames=len(frames))
        for metric in metrics:
            values = [float(r[metric]) for r in frames if r.get(metric) not in ('', None)]
            item[metric] = statistics.mean(values) if len(values) == len(frames) else None
        output.append(item)
    return output


def seed_groups(rows, metrics=METRICS):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['model'], row['condition'])].append(row)
    output = []
    for (model, condition), entries in sorted(grouped.items()):
        item = dict(model=model, condition=condition, n_seeds=len(entries))
        for metric in metrics:
            if any(r.get(metric) is None for r in entries):
                item[metric + '_mean'] = item[metric + '_SD'] = None
            else:
                mean, sd, _ = paired_seed_summary(entries, metric)
                item[metric + '_mean'], item[metric + '_SD'] = mean, sd
        output.append(item)
    return output


def main(output):
    evidence = ROOT / 'evidence'
    dense = read_csv(evidence / 'per_frame/dense_per_frame_metrics.csv')
    seed_rows = frame_groups(dense, expected_frames=296)
    expected = read_csv(evidence / 'per_frame/dense_seed_metrics.csv')
    lookup = {(r['model'], int(r['seed']), r['condition']): r for r in seed_rows}
    maximum_error = 0.
    for row in expected:
        actual = lookup[row['model'], int(row['seed']), row['condition']]
        for metric in METRICS:
            maximum_error = max(maximum_error, abs(actual[metric] - float(row[metric])))
    if maximum_error > 1e-6:
        raise ValueError(f'Source metric mismatch: {maximum_error}')
    write_csv(output / 'rsrd_seed_metrics.csv', seed_rows)
    summary = seed_groups(seed_rows)
    write_csv(output / 'rsrd_main_signed.csv', [r for r in summary if r['model'] in MAIN])
    extra = frame_groups(read_csv(evidence / 'per_frame/rsrd_oracle_metadata.csv'), metrics=METRICS[:2], expected_frames=296)
    write_csv(output / 'rsrd_oracle_metadata.csv', seed_groups(extra, METRICS[:2]))
    magnitudes = []
    for model in MAIN:
        for axis in ('pitch', 'roll'):
            for magnitude in (1, 2, 3, 5):
                entries = [r for r in seed_rows if r['model'] == model and
                           r['condition'] in (f'rotation_{axis}_{-magnitude:+d}', f'rotation_{axis}_{magnitude:+d}')]
                item = dict(model=model, axis=axis, magnitude_deg=magnitude, n_seeds=3)
                for metric in METRICS:
                    item[metric + '_mean'], item[metric + '_SD'], _ = paired_seed_summary(entries, metric)
                magnitudes.append(item)
    write_csv(output / 'rsrd_ias_magnitude.csv', magnitudes)
    ablation = []
    for model in ('exp21', 'A2', 'A3', 'S_free'):
        for label in ('clean', 'IAS16_mean'):
            entries = [r for r in seed_rows if r['model'] == model and
                       (r['condition'] == 'clean' if label == 'clean' else r['condition'].startswith('rotation_'))]
            item = dict(model=model, condition=label, n_seeds=3,
                        AFS=model != 'exp21', augmentation=model != 'A2', composite_loss=model != 'A3')
            for metric in METRICS:
                item[metric + '_mean'], item[metric + '_SD'], _ = paired_seed_summary(entries, metric)
            ablation.append(item)
    write_csv(output / 'ablation.csv', ablation)
    original_ablation = read_csv(evidence / 'references/ablation.csv')
    for item in ablation:
        reference = next(r for r in original_ablation if r['model'] == item['model'] and r['condition'] == item['condition'])
        for metric in METRICS:
            assert abs(item[metric + '_mean'] - float(reference[metric])) < 1e-6
    new = read_csv(evidence / 'per_frame/selfcollected_v2.csv')
    latest = []
    for model in ('baseline_aug', 'S_free'):
        for subset in ('all', 'instantaneous_high', 'recurrent_high_segments', 'bump_union', 'remaining'):
            rows = [r for r in new if r['model'] == model and (subset == 'all' or
                    (r['bump_union'] == 'False' if subset == 'remaining' else r[subset] == 'True'))]
            item = dict(dataset_version='data1_half_bump_v1_20261023', model=model, seed=307,
                        subset=subset, frames=len(rows), bags=len({r['bag'] for r in rows}))
            item.update({metric: statistics.mean(float(r[metric]) for r in rows) for metric in METRICS})
            latest.append(item)
    reference = json.loads((evidence / 'references/same_data_final_comparison_20261025.json').read_text())
    ref_union = json.loads((evidence / 'references/bump_subset_comparison_20261025.json').read_text())
    for row in latest:
        name = 'RoadBEV_Aug' if row['model'] == 'baseline_aug' else 'EF'
        if row['subset'] in reference['test']:
            expected_metrics = reference['test'][row['subset']][name]
        else:
            expected_metrics = ref_union[row['subset']][name]
        for metric in METRICS:
            assert abs(row[metric] - expected_metrics[metric]) < 1e-6, (row, metric)
    write_csv(output / 'selfcollected_v2.csv', latest)
    old = read_csv(evidence / 'per_frame/selfcollected_per_frame_metrics.csv')
    older = []
    for subset in ('all', 'bump'):
        items = frame_groups([r for r in old if subset == 'all' or r['is_bump'] == 'True'])
        for item in items:
            item = dict(dataset_version='selfcollected_v1_1139', subset=subset, **item)
            older.append(item)
    write_csv(output / 'selfcollected_v1.csv', older)
    shapes = frame_groups(read_csv(evidence / 'per_frame/rsrd_shape_clean.csv'), SHAPE, expected_frames=296)
    write_csv(output / 'rsrd_shape_clean.csv', seed_groups(shapes, SHAPE))
    dates = {r['sample_id']: r['date'] for r in read_csv(evidence / 'protocols/rsrd_test_groups.csv')}
    date_rows = []
    for date in sorted(set(dates.values())):
        for row in seed_groups(frame_groups([r for r in dense if r['condition'] == 'clean' and dates[r['sample_id']] == date])):
            date_rows.append(dict(date=date, **row))
    write_csv(output / 'rsrd_date_sensitivity.csv', date_rows)
    report = dict(status='passed', dense_cached_conditions=len(seed_rows),
                  max_source_metric_difference=maximum_error, extra_cached_conditions=len(extra),
                  new_test_frames_per_model=988, bump_union_frames=139,
                  epoch_selection='RSRD fixed final50; self-collected validation-selected checkpoints',
                  statistics='Frame equal weighting; signed conditions averaged within each seed; sample SD across seeds.')
    (output / 'table_validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results')
    main(parser.parse_args().output)
