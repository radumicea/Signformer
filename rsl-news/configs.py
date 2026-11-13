#!/usr/bin/env python3
"""Every run of the RSL-News experiments (README.md) and its config.

    configs.py OUT                  # writes OUT/configs/<run>.yaml for every run, from base.yaml
    configs.py OUT --list GROUP...  # the runs of some groups (or run names), one per line

A run is <name>-s<seed>: a split variant (splits.py: OUT/data-<variant>), a time window and a seed."""
import argparse
import copy
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent

# time windows, seconds around a sentence's timestamps [s, e]: (train, default test). before is subtracted from s;
# a list is a range drawn at random, anew every epoch.
WINDOWS = {
    'none': (None, None),
    'fixed': ({'before': -0.5, 'after': 2}, {'before': -0.5, 'after': 2}),
    'flat5': ({'before': 0, 'after': 5}, {'before': 0, 'after': 5}),
    'jit-notrim': ({'before': 0, 'after': [1, 3]}, {'before': 0, 'after': 2}),
    'jit-long': ({'before': 0, 'after': [1, 5]}, {'before': 0, 'after': 3}),
    'jit-xlong': ({'before': 0, 'after': [1, 7]}, {'before': 0, 'after': 4}),
    'jit': ({'before': [-1, 0], 'after': [1, 3]}, {'before': -0.5, 'after': 2}),
    'jit-long-trim': ({'before': [-1, 0], 'after': [1, 5]}, {'before': 0, 'after': 3}),
}
WINDOW = 'jit-long'                     # the window chosen by the window runs, used by every other run

RUNS = {}                               # run: (variant, window, seed, options)
GROUPS = {}


def add(group, name, variant, window, seeds, **options):
    for s in seeds:
        RUNS[f'{name}-s{s}'] = (variant, window, s, options)
        GROUPS.setdefault(group, []).append(f'{name}-s{s}')


# the time window: 12.5% of train, each run also decoded on 32 test windows
for w in WINDOWS:
    add('window', f'win-{w}', 'lc12', w, (1, 2, 3) if w in ('jit', 'jit-long') else (1, 2), window_grid=True)
# the main model and the learning curve (whose 12.5% point is win-jit-long)
add('main', 'main', 'std', WINDOW, (1, 2, 3))
add('scale', 'lc25', 'lc25', WINDOW, (1,))
add('scale', 'lc50', 'lc50', WINDOW, (1,))
# generalisation: unseen signers, time, unseen channels
add('signer', 'hold-i6i7', 'hold-i6i7', WINDOW, (1,))
add('signer', 'hold-i2', 'hold-i2', WINDOW, (1,))
add('signer', 'hold-i2-ctrl', 'hold-i2-ctrl', WINDOW, (1,))
add('time', 'chrono', 'chrono', WINDOW, (1,))
for c in ('protv', 'digi24', 'primatv'):
    add('channel', f'hold-{c}', f'hold-{c}', WINDOW, (1,))
# per-broadcast feature normalisation (stats.py)
add('norm', 'hold-i2-norm', 'hold-i2', WINDOW, (1,), norm=True)
add('norm', 'hold-i2-ctrl-norm', 'hold-i2-ctrl', WINDOW, (1,), norm=True)


def config(out, run):
    variant, window, seed, options = RUNS[run]
    cfg = copy.deepcopy(yaml.safe_load((HERE / 'base.yaml').read_text()))
    cfg['name'] = run
    cfg['data']['data_path'] = str(out / f'data-{variant}')
    cfg['data']['train_timing'], cfg['data']['eval_timing'] = WINDOWS[window]
    cfg['training']['random_seed'] = seed
    cfg['training']['model_dir'] = str(out / 'runs' / run)
    if options.get('norm'):
        cfg['data']['feature_norm'] = str(out / 'bsl5k_stats.npz')
        cfg['data']['feature_norm_floor'] = 0.01
    if options.get('window_grid'):
        cfg['testing']['window_grid'] = True
    return cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('out', type=Path)
    ap.add_argument('--list', nargs='+', metavar='GROUP')
    a = ap.parse_args()
    out = a.out.resolve()
    if a.list:
        for g in a.list:
            print('\n'.join(GROUPS[g] if g in GROUPS else [g] if g in RUNS else exit(f'no group or run {g}')))
        return
    (out / 'configs').mkdir(parents=True, exist_ok=True)
    for run in RUNS:
        (out / 'configs' / f'{run}.yaml').write_text(yaml.safe_dump(config(out, run), sort_keys=False))
    print(f'{len(RUNS)} configs in {out / "configs"}; groups: ' + ', '.join(f'{g} ({len(r)})' for g, r in GROUPS.items()))


if __name__ == '__main__':
    main()
