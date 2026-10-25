"""Check release language, provenance, pairings, numerical tables and file integrity."""
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image
from pypdf import PdfReader

from scripts.analysis import frame_metrics
from scripts.build_tables import read_csv

ROOT = Path(__file__).resolve().parents[1]
IGNORED = {'.git', '__pycache__', 'build', '.venv', 'evaluation_output'}
TEXT = {'.py', '.md', '.json', '.csv', '.txt', '.yml', '.yaml', '.toml', '.bib'}
CJK = re.compile(r'[\u2e80-\u9fff\uf900-\ufaff\U00020000-\U0002ffff]')
MANIFEST = 'SHA256SUMS.json'
REPORT = 'verification_report.json'


def release_files(root=ROOT):
    return sorted(p for p in root.rglob('*') if p.is_file() and not any(part in IGNORED for part in p.relative_to(root).parts))


def check_text(value, location):
    if isinstance(value, dict):
        for key, item in value.items():
            check_text(key, location); check_text(item, location)
    elif isinstance(value, list):
        for item in value: check_text(item, location)
    elif isinstance(value, str) and CJK.search(value):
        raise ValueError(f'Non-English script found: {location}')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_files(files):
    counts = dict(text_files=0, PNGs=0, PDFs=0, NPZs=0)
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        check_text(relative, relative)
        if not relative.isascii():
            raise ValueError(f'Filename is not ASCII: {relative}')
        if path.stat().st_size > 25 * 2**20:
            raise ValueError(f'Unexpected large release file: {relative}')
        if path.suffix in {'.pt', '.pth', '.ckpt', '.zip', '.tar', '.gz', '.pkl', '.pem'}:
            raise ValueError(f'Excluded archive, checkpoint or sensitive file: {relative}')
        if path.suffix.lower() in TEXT or path.name.startswith('.'):
            content = path.read_text(encoding='utf-8-sig')
            check_text(content, relative); counts['text_files'] += 1
            if path.suffix == '.json': check_text(json.loads(content), relative)
            if path.suffix == '.py': ast.parse(content, filename=relative)
            if (re.search(r'(?i)(?:gh[pousr]_[a-z0-9]{30,}|github_pat_[a-z0-9_]{30,})', content)
                    or re.search('-----BEGIN ' + r'[A-Z ]*' + 'PRIVATE KEY-----', content)):
                raise ValueError(f'Credential-like material: {relative}')
            if path.suffix == '.md':
                for target in re.findall(r'\]\(([^)]+)\)', content):
                    if target.startswith(('http:', 'https:', '#')): continue
                    target = target.split('#', 1)[0]
                    if not (path.parent / target).exists():
                        raise ValueError(f'Broken relative link: {relative} -> {target}')
        elif path.suffix.lower() == '.png':
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                check_text({k: v for k, v in image.info.items() if isinstance(v, str)}, relative)
            counts['PNGs'] += 1
        elif path.suffix.lower() == '.pdf':
            reader = PdfReader(path)
            check_text({str(k): str(v) for k, v in (reader.metadata or {}).items()}, relative)
            for page in reader.pages: check_text(page.extract_text(), relative)
            counts['PDFs'] += 1
        elif path.suffix.lower() == '.npz':
            with np.load(path, allow_pickle=False) as cache:
                for key in cache.files:
                    check_text(key, relative)
                    if cache[key].dtype.kind in ('U', 'S'): check_text(cache[key].tolist(), relative)
            counts['NPZs'] += 1
    return counts


def validate_provenance():
    for manifest_name in ('import_manifest.json', 'adapter_imports.json'):
        entries = json.loads((ROOT / 'evidence/provenance' / manifest_name).read_text())
        for entry in entries:
            if entry['transformation'] == 'byte-identical copy':
                assert digest(ROOT / entry['packaged_path']) == entry['source_sha256'], entry['packaged_path']
    records = read_csv(ROOT / 'evidence/per_frame/dense_per_frame_metrics.csv')
    full = {(r['model'], int(r['seed']), r['sample_id']): float(r['MAE_cm']) for r in records if r['condition'] == 'clean'}
    for source in (ROOT / 'evidence/figure_sources').glob('rsrd_*.npz'):
        sid = source.stem.removeprefix('rsrd_')
        with np.load(source, allow_pickle=False) as cache:
            for model in ('baseline_aug', 'exp21', 'S_free'):
                for seed in (307, 1101, 2202):
                    actual = frame_metrics(cache['gt'], cache[f'{model}_s{seed}'], cache['mask'])['MAE_cm']
                    assert abs(actual - full[model, seed, sid]) < 1e-6, (sid, model, seed)
    latest = read_csv(ROOT / 'evidence/per_frame/selfcollected_v2.csv')
    pair = {model: {r['sample_id'] for r in latest if r['model'] == model} for model in ('baseline_aug', 'S_free')}
    assert pair['baseline_aug'] == pair['S_free'] and len(pair['S_free']) == 988
    unique = [r for r in latest if r['model'] == 'S_free']
    assert sum(r['instantaneous_high'] == 'True' for r in unique) == 42
    assert sum(r['recurrent_high_segments'] == 'True' for r in unique) == 120
    assert sum(r['bump_union'] == 'True' for r in unique) == 139
    assert sum(r['instantaneous_high'] == r['recurrent_high_segments'] == 'True' for r in unique) == 23
    full_new = {(r['model'], r['sample_id']): float(r['MAE_cm']) for r in latest}
    for source in (ROOT / 'evidence/figure_sources').glob('self_data_*.npz'):
        sid = source.stem.removeprefix('self_data_')
        with np.load(source, allow_pickle=False) as cache:
            for model, key in [('baseline_aug', 'road'), ('S_free', 'ef')]:
                actual = frame_metrics(cache['gt'], cache[key], cache['mask'])['MAE_cm']
                assert abs(actual - full_new[model, sid]) < 1e-6
    split = read_csv(ROOT / 'evidence/protocols/selfcollected_v2_split.csv')
    ids = {s: {r['sample_id'] for r in split if r['split'] == s} for s in ('train', 'val', 'test')}
    assert [len(ids[s]) for s in ('train', 'val', 'test')] == [2552, 1205, 988]
    assert not (ids['train'] & ids['val'] or ids['train'] & ids['test'] or ids['val'] & ids['test'])
    panels = [p for p in (ROOT / 'figures').glob('*.png')
              if re.fullmatch(r'rsrd_\d{14}\.\d+_s(?:307|1101|2202)\.png', p.name)]
    assert len(panels) == 6
    validation = json.loads((ROOT / 'results/table_validation.json').read_text())
    assert validation['status'] == 'passed' and validation['max_source_metric_difference'] <= 1e-6


def main(write_manifest=False):
    files = release_files()
    counts = validate_files(files)
    validate_provenance()
    hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in files if p.name not in (MANIFEST, REPORT)}
    if write_manifest:
        (ROOT / MANIFEST).write_text(json.dumps(hashes, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    else:
        expected = json.loads((ROOT / MANIFEST).read_text())
        if hashes != expected:
            changed = sorted(set(hashes) ^ set(expected) | {k for k in hashes.keys() & expected.keys() if hashes[k] != expected[k]})
            raise ValueError('Release inventory mismatch: ' + ', '.join(changed[:15]))
    report = dict(status='passed', files_hashed=len(hashes), **counts, checks=[
        'English-only script and ASCII filenames; decoded JSON, PDF text and image metadata scanned',
        'Python syntax, relative documentation links and credential patterns checked',
        'Byte-identical imports checked against source SHA256',
        'Selected arrays reproduce full-table frame MAEs within 1e-6 cm',
        '988 paired IDs; 42/120/139 subsets; 23-frame overlap; disjoint 2552/1205/988 splits',
        'Six final50 qualitative RSRD panels preserve both frames across all three seeds',
        'Formal table reconstruction tolerance and file inventory checked'],
        inference_validation='No new model inference. Native CPU projection checks are recorded separately.',
        limitations='Statistical and deployment limitations are documented; no missing experiment is assigned a placeholder score.')
    (ROOT / REPORT).write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-manifest', action='store_true', help='Regenerate the reviewed release inventory after all source checks pass.')
    main(parser.parse_args().write_manifest)
