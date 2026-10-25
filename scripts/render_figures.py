"""Rebuild honest, shared-scale figures from preserved tables and arrays."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Polygon
import numpy as np
from PIL import Image

from scripts.analysis import deplane, frame_metrics, shape_metrics
from scripts.build_tables import read_csv, write_csv

ROOT = Path(__file__).resolve().parents[1]
NAMES = {'baseline': 'RoadBEV', 'baseline_aug': 'RoadBEV + Aug', 'exp21': 'CST-Elev',
         'S_free': 'EF-Elev', 'A2': 'No augmentation', 'A3': 'Plain CE'}
COLORS = {'baseline': '#999999', 'baseline_aug': '#0072B2', 'exp21': '#D55E00', 'S_free': '#009E73'}
STYLES = {'baseline': ':', 'baseline_aug': '--', 'exp21': '-.', 'S_free': '-'}


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(str(path) + '.png'), dpi=170, facecolor='white')
    fig.savefig(Path(str(path) + '.pdf'), metadata={'Title': path.name, 'Author': 'EF-Elev evidence repository',
                                                   'CreationDate': None, 'ModDate': None})
    plt.close(fig)


def curves(output):
    rows = read_csv(ROOT / 'results/rsrd_main_signed.csv')
    extra = read_csv(ROOT / 'results/rsrd_oracle_metadata.csv')
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), layout='constrained')
    for axis, ax in zip(('pitch', 'roll'), axes):
        for model in COLORS:
            points = []
            for row in rows:
                condition = row['condition']
                if row['model'] == model and (condition == 'clean' or condition.startswith('rotation_' + axis + '_')):
                    angle = 0 if condition == 'clean' else int(condition.rsplit('_', 1)[1])
                    points.append((angle, float(row['MAE_cm_mean']), float(row['MAE_cm_SD'])))
            points.sort(); x, y, sd = np.asarray(points).T
            ax.plot(x, y, STYLES[model], color=COLORS[model], marker='o', ms=3, label=NAMES[model])
            ax.fill_between(x, y - sd, y + sd, color=COLORS[model], alpha=.10)
        oracle = [r for r in extra if r['model'] == 'baseline_aug' and r['condition'].startswith('rotation_' + axis)]
        oracle.sort(key=lambda r: int(r['condition'].split('_')[2]))
        ax.plot([int(r['condition'].split('_')[2]) for r in oracle], [float(r['MAE_cm_mean']) for r in oracle],
                color='#CC79A7', linestyle=':', marker='s', ms=3, label='RoadBEV + Aug, updated')
        ax.axvspan(-1, 1, color='black', alpha=.05, label='Training range per axis')
        ax.set(xlabel=f'Signed {axis} perturbation (degrees)', ylabel='Frame-mean MAE (cm)', title=axis.capitalize())
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('RSRD-dense: final50, 296 frames, 3 seeds; bands = sample SD')
    save(fig, output / 'rsrd_ias_signed')

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), layout='constrained')
    for axis, ax in zip(('pitch', 'roll'), axes):
        for model in ('baseline', 'baseline_aug'):
            for kind, style in [('calR', '-'), ('INS', '--')]:
                points = [r for r in extra if r['model'] == model and r['condition'].startswith(f'{kind}_{axis}_')]
                points.sort(key=lambda r: int(r['condition'].rsplit('_', 1)[1]))
                ax.plot([int(r['condition'].rsplit('_', 1)[1]) for r in points],
                        [float(r['MAE_cm_mean']) for r in points], style, color=COLORS[model], marker='o', ms=3,
                        label=f'{NAMES[model]}, {kind}')
        for model in ('exp21', 'S_free'):
            clean = next(r for r in rows if r['model'] == model and r['condition'] == 'clean')
            ax.axhline(float(clean['MAE_cm_mean']), color=COLORS[model], linestyle=STYLES[model],
                       label=NAMES[model] + ' (metadata-independent)')
        ax.set(xlabel=f'Signed {axis} metadata bias (degrees)', ylabel='Frame-mean MAE (cm)', title=axis.capitalize())
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=7.5)
    fig.suptitle('Metadata bias: image unchanged; horizontal lines are structural controls')
    save(fig, output / 'rsrd_metadata_bias')

    latest = read_csv(ROOT / 'results/selfcollected_v2.csv')
    subsets = ['all', 'instantaneous_high', 'recurrent_high_segments', 'bump_union']
    labels = ['All\n988 frames', 'Instantaneous high\n42 frames', 'Recurrent context\n120 frames', 'Bump union\n139 frames']
    fig, ax = plt.subplots(figsize=(9, 4.4), layout='constrained')
    x = np.arange(4)
    for offset, model in [(-.17, 'baseline_aug'), (.17, 'S_free')]:
        values = [float(next(r for r in latest if r['model'] == model and r['subset'] == s)['MAE_cm']) for s in subsets]
        bars = ax.bar(x + offset, values, width=.32, label=NAMES[model], color=COLORS[model])
        ax.bar_label(bars, fmt='%.3f', fontsize=9, padding=3)
    ax.set(xticks=x, xticklabels=labels, ylabel='Frame-mean MAE (cm)', ylim=(0, 2.7))
    ax.legend(); ax.grid(axis='y', alpha=.2)
    ax.set_title('Latest self-collected comparison: seed307; overlapping subsets\nStereo RoadBEV vs monocular EF; native training settings differ', fontsize=11)
    save(fig, output / 'selfcollected_v2_comparison')


def qualitative(output):
    records = []
    for source in sorted((ROOT / 'evidence/figure_sources').glob('rsrd_*.npz')):
        sid = source.stem.removeprefix('rsrd_')
        with np.load(source, allow_pickle=False) as cache:
            gt, mask, image = cache['gt'], cache['mask'].astype(bool), cache['image']
            for seed in (307, 1101, 2202):
                cst, ef, road = [cache[f'{name}_s{seed}'] for name in ('exp21', 'S_free', 'baseline_aug')]
                fig = plt.figure(figsize=(16, 7.2), layout='constrained')
                gs = fig.add_gridspec(2, 8, height_ratios=[1.65, 1], width_ratios=[2.5, 1, 1, 1, 1, .09, 1, .09])
                img_ax = fig.add_subplot(gs[0, 0]); img_ax.imshow(image); img_ax.set_title('Actual RGB input + reference ROI'); img_ax.axis('off')
                xyz = np.array([[-1, 1.1, 2.16], [.92, 1.1, 2.16], [.92, 1.1, 7.08], [-1, 1.1, 7.08]])
                uvz = xyz @ cache['r_vert2cam'].T @ cache['K'].T
                uv = uvz[:, :2] / uvz[:, 2:]
                img_ax.add_patch(Polygon(uv, fill=False, edgecolor='#D62728', linewidth=1.3))
                hmap = plt.get_cmap('viridis').with_extremes(bad='#dddddd')
                emap = plt.get_cmap('RdBu_r').with_extremes(bad='#dddddd')
                for col, value, title in [(1, gt, 'GT'), (2, road, 'RoadBEV + Aug'), (3, cst, 'CST-Elev'), (4, ef, 'EF-Elev')]:
                    ax = fig.add_subplot(gs[0, col]); h = ax.imshow(np.ma.masked_where(~mask, value), cmap=hmap, vmin=-5, vmax=15, aspect='auto', interpolation='nearest')
                    ax.set_title(title); ax.axis('off')
                fig.colorbar(h, cax=fig.add_subplot(gs[0, 5]), label='Elevation (cm)', extend='both')
                # A shared two-column residual inset retains equal color limits for both methods.
                sub = gs[0, 6].subgridspec(1, 2)
                for i, (pred, title) in enumerate([(cst, 'CST'), (ef, 'EF')]):
                    ax = fig.add_subplot(sub[0, i]); error = ax.imshow(np.ma.masked_where(~mask, gt - pred), cmap=emap, vmin=-3, vmax=3, aspect='auto', interpolation='nearest')
                    ax.set_title('GT - ' + title, fontsize=9); ax.axis('off')
                fig.colorbar(error, cax=fig.add_subplot(gs[0, 7]), label='GT - prediction (cm)', extend='both')
                gd, cd, ed = [deplane(a, mask) for a in (gt, cst, ef)]
                amps = [np.percentile(abs(gd[:, c][mask[:, c]]), 98) if mask[:, c].sum() >= 20 else -1 for c in range(mask.shape[1])]
                column = int(np.argmax(amps))
                row = int(np.argmax(np.where(mask[:, column], abs(gd[:, column]), -1)))
                z = 7.08 - (np.arange(gt.shape[0]) + .5) * .03
                x = -1 + (np.arange(gt.shape[1]) + .5) * .03
                for ax, positions, arrays, valid, title, xlabel in [
                    (fig.add_subplot(gs[1, :4]), z, [a[:, column] for a in (gd, cd, ed)], mask[:, column], f'Longitudinal cut, x={x[column]:.3f} m (GT-selected)', 'Forward distance (m)'),
                    (fig.add_subplot(gs[1, 4:]), x, [a[row] for a in (gd, cd, ed)], mask[row], f'Lateral cut, z={z[row]:.3f} m (GT-selected)', 'Lateral position (m)')]:
                    ax.plot(positions[valid], arrays[0][valid], 'k.', ms=3, label='GT')
                    ax.plot(positions, arrays[1], '--', color=COLORS['exp21'], label='CST-Elev')
                    ax.plot(positions, arrays[2], '-', color=COLORS['S_free'], label='EF-Elev')
                    ax.set(title=title, xlabel=xlabel, ylabel='Deplaned height (cm)'); ax.grid(alpha=.2); ax.legend(fontsize=8)
                cm, em = frame_metrics(gt, cst, mask), frame_metrics(gt, ef, mask)
                fig.suptitle(f'{sid} | seed {seed} | clean | final50\nOutcome-selected frame, all seeds retained. MAE: CST {cm["MAE_cm"]:.3f}, EF {em["MAE_cm"]:.3f} cm', fontsize=12)
                dest = output / f'rsrd_{sid}_s{seed}'
                save(fig, dest)
                for name, pred in [('exp21', cst), ('S_free', ef)]:
                    records.append(dict(sample_id=sid, seed=seed, model=name, epoch=50, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                        residual_saturated_pct=float(100 * (abs(gt[mask] - pred[mask]) > 3).mean()),
                        **frame_metrics(gt, pred, mask), **shape_metrics(gt, pred, mask)))
    return records


def self_examples(output):
    manifest = json.loads((ROOT / 'evidence/selection/selfcollected_v2_candidates.json').read_text())
    candidates = manifest['candidates']
    records = []
    for source in sorted((ROOT / 'evidence/figure_sources').glob('self_data_*.npz')):
        sid = source.stem.removeprefix('self_data_')
        with np.load(source, allow_pickle=False) as cache:
            gt, mask, road, ef = cache['gt'], cache['mask'].astype(bool), cache['road'], cache['ef']
            image = Image.open(ROOT / 'evidence/figure_sources' / f'self_original_{sid}.png')
            fig, axes = plt.subplots(1, 6, figsize=(15.5, 4.8), gridspec_kw={'width_ratios': [2.3, 1, 1, 1, 1, 1]}, layout='constrained')
            axes[0].imshow(image); axes[0].set_title('Original RGB + projected ROI')
            record = next(r for r in candidates if r['sample_id'] == sid)
            # Reference-plane projection is an overlay only; no prediction uses this polygon.
            lo, hi = record['roi_x']; near, far = record['roi_z']
            corners = np.array([[lo, far], [hi, far], [hi, near], [lo, near]])
            perimeter = np.concatenate([corners[i] + np.linspace(0, 1, 60, endpoint=False)[:, None] *
                (corners[(i + 1) % 4] - corners[i]) for i in range(4)])
            a, b, c = record['plane_abc']
            world = np.column_stack([perimeter[:, 0], a * perimeter[:, 0] + b * perimeter[:, 1] + c, perimeter[:, 1]])
            camera = world @ np.asarray(record['r_vert2cam']).T
            x, y = camera[:, 0] / camera[:, 2], camera[:, 1] / camera[:, 2]
            k1, k2, p1, p2 = manifest['raw_camera_D']
            rr = x*x + y*y; radial = 1 + k1*rr + k2*rr*rr
            xd = x*radial + 2*p1*x*y + p2*(rr + 2*x*x)
            yd = y*radial + p1*(rr + 2*y*y) + 2*p2*x*y
            k = np.asarray(manifest['raw_camera_K'])
            uv = np.column_stack([k[0, 0]*xd + k[0, 2], k[1, 1]*yd + k[1, 2]])
            axes[0].add_patch(Polygon(uv, fill=False, edgecolor='#D62728', linewidth=1.2))
            axes[0].set_xlim(-.5, image.width-.5); axes[0].set_ylim(image.height-.5, -.5)
            hmap = plt.get_cmap('viridis').with_extremes(bad='#dddddd')
            emap = plt.get_cmap('RdBu_r').with_extremes(bad='#dddddd')
            for ax, value, title in zip(axes[1:], [gt, road, ef, gt-road, gt-ef], ['GT', 'RoadBEV + Aug', 'EF-Elev', 'GT - RoadBEV', 'GT - EF']):
                is_error = title.startswith('GT -')
                h = ax.imshow(np.ma.masked_where(~mask, value), cmap=emap if is_error else hmap, vmin=-3 if is_error else -14, vmax=3 if is_error else 14, aspect='auto', interpolation='nearest')
                ax.set_title(title, fontsize=10)
                if title in ('EF-Elev', 'GT - EF'):
                    fig.colorbar(h, ax=ax, fraction=.045, pad=.02, extend='both', label='Error (cm)' if is_error else 'Elevation (cm)')
            for ax in axes: ax.axis('off')
            rm, em = frame_metrics(gt, road, mask), frame_metrics(gt, ef, mask)
            fig.suptitle(f'Frame {sid} | validation-selected checkpoints | seed307\nIllustrative success selected by outcome; MAE RoadBEV {rm["MAE_cm"]:.3f}, EF {em["MAE_cm"]:.3f} cm\nStereo RoadBEV vs monocular EF. Gray: missing GT. Error colors saturate outside +/-3 cm.', fontsize=10)
            save(fig, output / f'self_{sid}')
            for name, pred in [('baseline_aug', road), ('S_free', ef)]:
                records.append(dict(sample_id=sid, seed=307, model=name, residual_saturated_pct=float(100 * (abs(gt[mask]-pred[mask]) > 3).mean()),
                                    **frame_metrics(gt, pred, mask), **shape_metrics(gt, pred, mask)))
    return records


def main(output):
    output.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({'font.size': 9, 'pdf.fonttype': 42, 'svg.fonttype': 'none'}):
        curves(output)
        rsrd = qualitative(output)
        latest = self_examples(output)
    write_csv(output / 'rsrd_selected_frame_metrics.csv', rsrd)
    write_csv(output / 'self_selected_frame_metrics.csv', latest)
    print(f'Rebuilt {len(list(output.glob("*.png")))} figures from preserved evidence.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'figures')
    main(parser.parse_args().output)
