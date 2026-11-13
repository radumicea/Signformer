#!/usr/bin/env python3
"""Per-broadcast statistics of the BSL-5k features, for per-broadcast normalisation (README.md, group norm):

    stats.py RSL_NEWS OUT/bsl5k_stats.npz

Over every window of every segment of a broadcast (labelled or not): the mean and standard deviation of each of the
1,024 dimensions. Writes keys (<Channel>/<PrimaTV id or date>), windows, mean, std (float32)."""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from rsl_news.files import load

CHANNELS = {'digi24': 'Digi24', 'prima': 'PrimaTV', 'protv': 'ProTV'}


def stats(job):
    key, files = job
    n, s, ss = 0, np.zeros(1024), np.zeros(1024)
    for f in files:
        if f.exists():
            x = load(f)['features'].astype(np.float64)
            n += len(x); s += x.sum(0); ss += (x * x).sum(0)
    mean = s / max(n, 1)
    return key, n, mean, np.sqrt(np.maximum(ss / max(n, 1) - mean * mean, 0))


if __name__ == '__main__':
    root = Path(sys.argv[1])
    jobs = []
    for m, channel in CHANNELS.items():
        for item in json.loads((root / 'manifests' / f'{m}_manifest.json').read_text())['items']:
            key = item['id'] if m == 'prima' else item['date']
            folder = root / 'dataset' / channel / (key if m == 'prima' else key.replace('-', '/'))
            jobs.append((f'{channel}/{key}', [folder / f'segment_{s["index"]}.bsl5k.npz' for s in item['segments']]))
    with Pool(8) as pool:
        res = [r for r in pool.map(stats, jobs, chunksize=4) if r[1] > 0]
    np.savez(sys.argv[2], keys=np.array([r[0] for r in res]), windows=np.array([r[1] for r in res]),
             mean=np.stack([r[2] for r in res]).astype(np.float32), std=np.stack([r[3] for r in res]).astype(np.float32))
    print(f'{len(res)} broadcasts')
