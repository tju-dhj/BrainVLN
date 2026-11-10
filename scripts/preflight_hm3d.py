#!/usr/bin/env python3
"""Audit HM3D experiment inputs without requiring torch or Habitat.

Writes a new project-local run directory; never installs packages or commits.
Exit 2 means training prerequisites are missing. This is not a simulator test.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone, timedelta

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def command(args, env=None):
    try:
        result = subprocess.run(args, cwd=ROOT, env=env, capture_output=True,
                                text=True, timeout=60)
        return {'returncode': result.returncode, 'stdout': result.stdout.strip(),
                'stderr': result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'returncode': -1, 'error': str(exc)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', default=sys.executable, help='Intended Habitat interpreter')
    parser.add_argument('--habitat-lab', default=os.environ.get('HABITAT_LAB', ''))
    parser.add_argument('--output', default='', help='New directory under project storage/')
    args = parser.parse_args()
    now = datetime.now(timezone(timedelta(hours=8)))
    output = (ROOT / (args.output or ('storage/hm3d_preflight_' + now.strftime('%Y%m%d_%H%M%S_%f')))).resolve()
    if not output.is_relative_to(ROOT / 'storage'):
        parser.error('--output must resolve inside project storage/')
    output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    for key, name in [('TMPDIR', 'tmp'), ('PIP_CACHE_DIR', 'pip'), ('TORCH_HOME', 'torch'),
                      ('XDG_CACHE_HOME', 'xdg'), ('CUDA_CACHE_PATH', 'cuda')]:
        path = output / 'runtime' / name
        path.mkdir(parents=True)
        env[key] = str(path)
    env['PYTHONPATH'] = str(ROOT) + (os.pathsep + args.habitat_lab if args.habitat_lab else '')
    git = {name: command(['git'] + argv) for name, argv in {
        'head': ['rev-parse', 'HEAD'], 'branch': ['branch', '--show-current'],
        'status': ['status', '--short'], 'diff': ['diff', 'HEAD', '--', '.', ':!SpikingNav.pdf'],
    }.items()}
    (output / 'working_tree.patch').write_text(git.pop('diff').get('stdout', '') + '\n')
    sources = {}
    for directory in ['scripts', 'spikingnav', 'tests', 'docs', 'projects']:
        for path in sorted((ROOT / directory).rglob('*')):
            if path.is_file() and not path.is_symlink() and path.suffix in {'.py', '.sh', '.md', '.json', '.yaml', '.yml'}:
                relative = path.relative_to(ROOT)
                sources[str(relative)] = sha256(path)
                target = output / 'source' / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
    for name in ['AGENTS.md', 'README.md', '.gitignore', 'pyproject.toml', 'requirements.txt', 'environment.yml', 'SpikingNav.pdf']:
        sources[name] = sha256(ROOT / name)
        if name != 'SpikingNav.pdf':
            shutil.copyfile(ROOT / name, output / 'source' / name)
    blockers = []
    probe = """import importlib.util, json, sys
modules = ['torch', 'torchvision', 'numpy', 'habitat', 'habitat_sim', 'tensorboard']
print(json.dumps({'python':sys.version, 'modules':{m:importlib.util.find_spec(m) is not None for m in modules}}))
"""
    runtime = command([args.python, '-B', '-c', probe], env)
    if runtime['returncode'] != 0:
        blockers.append('Cannot inspect intended interpreter')
    else:
        runtime['probe'] = json.loads(runtime['stdout'])
        missing = [m for m, ok in runtime['probe']['modules'].items() if not ok]
        if missing:
            blockers.append('Missing Python modules: ' + ', '.join(missing))
    splits = {}
    for split in ['train', 'val', 'val_mini']:
        base = ROOT / 'datasets/objectnav/hm3d/v1' / split
        files = sorted(base.glob('content/*.json.gz'))
        record = {'content_files': len(files), 'episodes': 0, 'scenes': [], 'missing_meshes': [], 'categories': {}}
        inputs = ([base / (split + '.json.gz')] if (base / (split + '.json.gz')).is_file() else []) + files
        record['annotation_sha256'] = {}
        for path in inputs:
            record['annotation_sha256'][str(path.relative_to(ROOT))] = sha256(path)
            with gzip.open(path, 'rt') as handle:
                data = json.load(handle)
            episodes = data.get('episodes', [])
            record['episodes'] += len(episodes)
            for episode in episodes:
                category = episode.get('object_category', 'unknown')
                record['categories'][category] = record['categories'].get(category, 0) + 1
            for scene in sorted({e['scene_id'] for e in episodes}):
                if scene not in record['scenes']:
                    record['scenes'].append(scene)
                    # Habitat removes this legacy dataset prefix before joining scenes_dir.
                    relative = scene.removeprefix('data/scene_datasets/')
                    mesh = ROOT / 'datasets/scene_datasets' / relative
                    if not mesh.is_file():
                        record['missing_meshes'].append(scene)
        if not record['episodes']:
            blockers.append(split + ': no episodes')
        if record['missing_meshes']:
            blockers.append(f"{split}: {len(record['missing_meshes'])}/{len(record['scenes'])} scene meshes missing")
        splits[split] = record
    prior = {}
    for path in sorted((ROOT / 'storage').glob('habitat-objectnav-hm3d-*/eval_results.tsv')):
        with path.open() as handle:
            rows = list(csv.DictReader(handle, delimiter='\t'))
        prior[str(path.relative_to(ROOT))] = {
            'rows': len(rows), 'best': max(rows, key=lambda r: float(r['sr'])) if rows else None,
            'limitation': 'Historical aggregate only; episode IDs, split and protocol are not recorded in TSV.',
        }
    checkpoints = {str(p.relative_to(ROOT)): {'bytes': p.stat().st_size, 'sha256': sha256(p)}
                   for p in sorted((ROOT / 'storage').glob('habitat-objectnav-hm3d-*/*.pt'))}
    report = {'created_at': now.isoformat(), 'root': str(ROOT), 'git': git,
              'source_sha256': sources, 'runtime': runtime, 'gpu': command(['nvidia-smi']),
              'disk_free_bytes': shutil.disk_usage(ROOT).free, 'datasets': splits,
              'historical_results': prior, 'checkpoints': checkpoints,
              'status': 'blocked' if blockers else 'ready_for_simulator_smoke',
              'blockers': blockers,
              'limits': ['No training or evaluation performed.',
                         'Module discovery does not prove binary/CUDA compatibility.',
                         'No Git commit, staging or checkpoint deletion performed.']}
    (output / 'manifest.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'report': str(output / 'manifest.json'), 'status': report['status'],
                      'blockers': blockers}, indent=2, ensure_ascii=False))
    return 2 if blockers else 0


if __name__ == '__main__':
    sys.exit(main())
