"""Evaluate saved predictions or replay native RSRD / monocular EF checkpoints."""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from scripts.analysis import frame_metrics
from scripts.build_tables import write_csv

ROOT = Path(__file__).resolve().parents[1]


def load_manifest(path, data_root):
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    rows = payload['samples'] if isinstance(payload, dict) else payload
    if not rows or len({str(r['sample_id']) for r in rows}) != len(rows):
        raise ValueError('Manifest must contain unique, nonempty sample IDs.')
    for row in rows:
        row['sample_id'] = str(row['sample_id'])
        row['gt_path'] = str(data_root / row['gt_path'])
        if 'image_path' in row:
            row['image_path'] = str(data_root / row['image_path'])
    return rows


def load_gt(path, height_limit):
    with np.load(path, allow_pickle=False) as data:
        gt = data['ele_gt'] if 'ele_gt' in data else data['gt']
        mask = data['ele_mask'] if 'ele_mask' in data else data['mask']
        return gt, mask.astype(bool) & (np.abs(gt) < height_limit)


def score_cache(predictions, samples, height_limit):
    with np.load(predictions, allow_pickle=False) as cache:
        ids = [str(x) for x in cache['sample_ids'].tolist()]
        values = cache['predictions']
        if ids != [s['sample_id'] for s in samples] or len(values) != len(ids):
            raise ValueError('Prediction IDs must exactly match manifest order.')
        rows = []
        for sample, pred in zip(samples, values):
            gt, mask = load_gt(sample['gt_path'], height_limit)
            rows.append(dict(sample_id=sample['sample_id'], **frame_metrics(gt, pred, mask)))
    return rows


def checkpoint_is_known(path):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    bindings = json.loads((ROOT / 'evidence/provenance/clean_cache_bindings.json').read_text())
    known = {r['checkpoint_sha256'] for r in bindings}
    latest = json.loads((ROOT / 'evidence/references/same_data_final_comparison_20261025.json').read_text())
    known.update(latest['checkpoint_sha256'].values())
    return digest, digest in known


def rsrd_indices(torch, k, rotation):
    # Preserve the native float32 arithmetic and z/x/y flattening order.
    grid_res = torch.tensor([.03, .01, .03])
    roi_x, roi_z = torch.tensor([-1., .92]), torch.tensor([2.16, 7.08])
    horizontal = torch.zeros((164, 64, 2), dtype=torch.float32)
    horizontal[:, :, 0] = (torch.arange(64) * grid_res[0] + roi_x[0] + grid_res[0]/2)[None].repeat(164, 1)
    horizontal[:, :, 1] = (-torch.arange(164) * grid_res[2] + roi_z[1] - grid_res[2]/2)[:, None].repeat(1, 64)
    voxels = torch.zeros((164, 64, 40, 3), dtype=torch.float32)
    voxels[:, :, :, [0, 2]] = horizontal.unsqueeze(2).repeat(1, 1, 40, 1)
    voxels[:, :, :, 1] = (torch.arange(40) * grid_res[1] + 1.1 - .2 + grid_res[1]/2)[None, None].repeat(164, 64, 1)
    camera = torch.as_tensor(rotation, dtype=torch.float32) @ voxels.reshape(-1, 3).T
    feature_k = torch.as_tensor(k, dtype=torch.float32) / 4
    feature_k[2, 2] = 1
    uvz = feature_k @ camera
    indices = torch.floor(uvz[:2] / uvz[2:]).long()
    indices[0].clamp_(0, 239); indices[1].clamp_(0, 131)
    return indices


def replay(args, samples, output):
    if args.dataset != 'rsrd' and args.model != 'S_free':
        raise ValueError('Self-collected checkpoint replay supports monocular EF only. All methods support --predictions.')
    if not all('image_path' in s for s in samples):
        raise ValueError('Checkpoint replay requires image_path for every manifest entry.')
    digest, known = checkpoint_is_known(args.checkpoint)
    if not known:
        raise ValueError('Checkpoint SHA256 is outside the frozen experiment inventory; replay is restricted to known checkpoints.')
    import torch
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError('Canonical checkpoint replay requires CUDA; offline cache evaluation does not.')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    baseline = args.model.startswith('baseline')
    if baseline:
        sys.path.insert(0, str(ROOT / 'src/roadbev'))
        from models.model import Elevation
        model = Elevation(False, [64, 40, 164], .2, .5).cuda().eval()
    else:
        sys.path.insert(0, str(ROOT / 'src/ef_elev'))
        from sampler import Exp29
        from improved import Improved
        if args.config is None:
            raise ValueError('Monocular checkpoint replay requires --config.')
        content = json.loads(args.config.read_text())
        config = content.get('config', content)
        if args.dataset == 'rsrd':
            model = (Improved if args.model == 'exp21' else Exp29)(config, initialize=False).cuda().eval()
        else:
            from self_inference import SelfSfree
            model = SelfSfree(config, initialize=False).cuda().eval()
    # Deserialization is allowed only after matching an inventoried experiment SHA256.
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if not baseline and checkpoint.get('config') != config:
        raise ValueError('Supplied config differs from the frozen checkpoint configuration.')
    model.load_state_dict(checkpoint['model'], strict=True)
    epoch = int(checkpoint['epoch']); del checkpoint
    if args.dataset == 'rsrd' and epoch != 50:
        raise ValueError('Formal RSRD replay requires final50 weights.')
    batch_size = 1 if baseline or args.dataset != 'rsrd' else 8
    predictions = []
    for start in range(0, len(samples), batch_size):
        chunk = samples[start:start + batch_size]
        images, indices = [], []
        for sample in chunk:
            with Image.open(sample['image_path']) as raw:
                if args.dataset == 'rsrd':
                    image = raw.convert('RGB').crop((0, 0, 960, 528))
                else:
                    image = raw.convert('RGB').crop((424, 920, 2024, 1800)).resize((960, 528))
                array = np.asarray(image).copy()
            value = torch.from_numpy(array).permute(2, 0, 1).float()
            # Baseline and self-collected loaders use ToTensor then Normalize.
            value = (value / 255. - .5) / .5 if baseline or args.dataset != 'rsrd' else value / 127.5 - 1
            images.append(value)
            if baseline:
                with np.load(sample['gt_path'], allow_pickle=False) as data:
                    indices.append(rsrd_indices(torch, data['K'], data['r_vert2cam']))
        autocast = torch.autocast('cuda', dtype=torch.bfloat16) if args.dataset == 'rsrd' and not baseline else nullcontext()
        with torch.inference_mode(), autocast:
            x = torch.stack(images).cuda()
            pred = model(x, torch.stack(indices).cuda()) if baseline else model.components(x)['prediction']
            predictions.append(pred.float().cpu().numpy())
    path = output / 'predictions_cm.npz'
    np.savez_compressed(path, sample_ids=np.array([s['sample_id'] for s in samples]),
                        predictions=np.concatenate(predictions), checkpoint_sha256=digest)
    return path, dict(checkpoint_sha256=digest, epoch=epoch, batch_size=batch_size,
                      precision='BF16' if args.dataset == 'rsrd' and not baseline else 'FP32',
                      condition='clean only; use archived protocol for synthetic perturbations')


def main(args):
    samples = load_manifest(args.manifest, args.data_root)
    if args.check_only:
        for sample in samples:
            load_gt(sample['gt_path'], 20 if args.dataset == 'rsrd' else 25)
            if 'image_path' in sample and not Path(sample['image_path']).is_file():
                raise FileNotFoundError(sample['image_path'])
        print(json.dumps(dict(status='passed', frames=len(samples), inference_performed=False)))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    info = dict(source='saved prediction evaluation', inference_performed=False)
    path = args.predictions
    if args.checkpoint:
        path, replay_info = replay(args, samples, args.output)
        info.update(replay_info, inference_performed=True)
    rows = score_cache(path, samples, 20 if args.dataset == 'rsrd' else 25)
    write_csv(args.output / 'per_frame.csv', rows)
    summary = dict(dataset=args.dataset, model=args.model, frames=len(rows), **info,
                   **{metric: float(np.mean([r[metric] for r in rows])) for metric in ('MAE_cm', 'RMSE_cm', 'GT0p5_pct')})
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['rsrd', 'selfcollected-v1', 'selfcollected-v2'], required=True)
    parser.add_argument('--model', choices=['baseline', 'baseline_aug', 'exp21', 'S_free', 'A2', 'A3'], default='S_free')
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--predictions', type=Path)
    source.add_argument('--checkpoint', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / 'evaluation_output')
    args = parser.parse_args()
    if not args.check_only and not (args.predictions or args.checkpoint):
        parser.error('Specify --predictions, --checkpoint or --check-only.')
    main(args)
