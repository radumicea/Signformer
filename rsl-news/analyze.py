#!/usr/bin/env python3
"""Every table of the experiments (README.md) from the evaluated runs: writes OUT/analysis.md and OUT/analysis.json.
From the Signformer folder:

    python rsl-news/analyze.py OUT

Sections whose runs are missing are skipped. Paired comparisons: B - A on the same test sentences, 95% interval from
resampling whole broadcasts (1,000 draws, seed 0). "Expected": the learning curve's score on the same sentences at a
number of train hours, linear in log2(hours) between lc50-s1 and the main model (mean of its seeds)."""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from configs import GROUPS, WINDOW, WINDOWS  # noqa: E402
from evaluate import broadcast_of, sentence_stats  # noqa: E402

OUT = Path(sys.argv[1]).resolve()
RUNS = OUT / 'runs'
HOURS = json.loads((OUT / 'splits' / 'hours.json').read_text())
V1 = json.loads((OUT / 'splits' / 'subsets.json').read_text())['std']       # v1.0 test broadcast -> channel, interpreter
CHANNELS = ('Digi24', 'PrimaTV', 'ProTV')
md, js, _hyps = [], {}, {}


def have(run):
    return (RUNS / run / 'eval' / 'results.json').exists()


def res(run):
    return json.loads((RUNS / run / 'eval' / 'results.json').read_text())


def read(path):
    return dict(line.split('|', 1) for line in path.read_text(encoding='utf-8').splitlines())


def hyps(run):
    if run not in _hyps:
        _hyps[run] = read(RUNS / run / 'eval' / 'test.hyp.txt')
    return _hyps[run]


mean = lambda v: sum(v) / len(v)
sd = lambda v: math.sqrt(sum((x - mean(v)) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else float('nan')
ci = lambda r, m='bleu': f'[{r[m + "_ci95"][0]:.2f}, {r[m + "_ci95"][1]:.2f}]'
REF = read(RUNS / 'main-s1' / 'eval' / 'test.ref.txt') if have('main-s1') else \
    read(RUNS / 'win-none-s1' / 'eval' / 'test.ref.txt')                     # every v1.0 test has these sentences


def names(**want):
    return sorted(n for n in REF if all(V1[broadcast_of(n)][k] == v for k, v in want.items()))


def score(run, ns):
    h = hyps(run)
    bs, cs, bm, cm = sentence_stats([REF[n] for n in ns], [h[n] for n in ns])
    return bm._compute_score_from_stats(list(bs.sum(0))).score


def paired(a, b, ns):
    """B - A on the sentences ns, BLEU-4 and chrF, with 95% intervals."""
    refs = [REF[n] for n in ns]
    sa, sb = sentence_stats(refs, [hyps(a)[n] for n in ns]), sentence_stats(refs, [hyps(b)[n] for n in ns])
    groups = defaultdict(list)
    for i, n in enumerate(ns):
        groups[broadcast_of(n)].append(i)
    keys = sorted(groups)
    r = {'sentences': len(ns), 'broadcasts': len(keys)}
    for metric, k in (('bleu', 0), ('chrf', 1)):
        f = lambda st: sa[2 + k]._compute_score_from_stats(list(st)).score
        ga = np.stack([sa[k][groups[g]].sum(0) for g in keys]); gb = np.stack([sb[k][groups[g]].sum(0) for g in keys])
        rng = np.random.default_rng(0)
        d = [f(gb[p].sum(0)) - f(ga[p].sum(0)) for p in (rng.integers(0, len(keys), len(keys)) for _ in range(1000))]
        r[metric] = {'a': f(ga.sum(0)), 'b': f(gb.sum(0)), 'diff': f(gb.sum(0)) - f(ga.sum(0)),
                     'ci95': [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
    return r


def fmt(p, m='bleu'):
    return f'{p[m]["diff"]:+.2f} [{p[m]["ci95"][0]:+.2f}, {p[m]["ci95"][1]:+.2f}]'


MAIN = [r for r in GROUPS['main'] if have(r)]


def expected(ns, hours):
    lo, hi = score('lc50-s1', ns), mean([score(r, ns) for r in MAIN])
    return lo + (hi - lo) * math.log2(hours / HOURS['lc50']) / math.log2(HOURS['std'] / HOURS['lc50'])


# ---- the time window (12.5% of train)
arms = {w: [r for r in GROUPS['window'] if r.startswith(f'win-{w}-s') and have(r)] for w in WINDOWS}
if all(len(v) >= 2 for v in arms.values()):
    dev = {w: [res(r)['dev']['all']['bleu'] for r in rs] for w, rs in arms.items()}
    test = {w: [res(r)['test']['all']['bleu'] for r in rs] for w, rs in arms.items()}
    var = sum((x - mean(v)) ** 2 for v in dev.values() for x in v) / sum(len(v) - 1 for v in dev.values())
    rank = {'none': 0, 'fixed': 1, 'flat5': 1, 'jit-notrim': 2, 'jit-long': 2, 'jit-xlong': 2, 'jit': 3, 'jit-long-trim': 3}
    winner = max(dev, key=lambda w: mean(dev[w]))
    close = [w for w in dev if rank[w] < rank[winner] and
             mean(dev[winner]) - mean(dev[w]) < math.sqrt(var * (1 / len(dev[w]) + 1 / len(dev[winner])))]
    chosen = min(close, key=lambda w: (rank[w], -mean(dev[w]))) if close else winner
    part = lambda v, sign: f'{min(sign * x for x in v):+g}..{max(sign * x for x in v):g}' if isinstance(v, list) else f'{sign * v:+g}'
    show = lambda t: '[s, e]' if t is None else f'[s{part(t["before"], -1)}, e{part(t["after"], 1)}]'
    md += ['## The time window (12.5% of train, 169.5 h)', '',
           '| Window | train | test | dev BLEU-4 per seed (mean) | test BLEU-4 (mean) | test chrF (mean) |', '|---|---|---|---|---|---|']
    for w, rs in arms.items():
        md.append(f'| {w} | {show(WINDOWS[w][0])} | {show(WINDOWS[w][1])} | {" / ".join(f"{x:.2f}" for x in dev[w])} ({mean(dev[w]):.2f}) | '
                  f'{mean(test[w]):.2f} | {mean([res(r)["test"]["all"]["chrf"] for r in rs]):.2f} |')
    md += ['', f'Seed SD of dev BLEU-4, pooled: {math.sqrt(var):.3f}. Highest mean dev BLEU-4: {winner}; chosen (a simpler '
               f'window within one standard error is preferred): **{chosen}**.', '']
    grid = {w: defaultdict(list) for w in arms}
    for w, rs in arms.items():
        for r in rs:
            for x in json.loads((RUNS / r / 'eval' / 'window_grid.json').read_text()):
                grid[w][(x['start_plus'], x['end_plus'])].append(x['bleu'])
    cells = sorted(grid['none'])
    md += ['Every window model on the same 32 test windows (dev subset, 5,000 sentences, mean over seeds; bold: its '
           'own test window):', '', '| test window | ' + ' | '.join(arms) + ' |', '|---|' + '---|' * len(arms)]
    for c in cells:
        own = lambda w: WINDOWS[w][1] is None and c == (0, 0) or WINDOWS[w][1] is not None and c == (-WINDOWS[w][1]['before'], WINDOWS[w][1]['after'])
        md.append(f'| [s{c[0]:+g}, e+{c[1]:g}] | ' + ' | '.join(
            (f'**{mean(grid[w][c]):.2f}**' if own(w) else f'{mean(grid[w][c]):.2f}') for w in arms) + ' |')
    runner = max((w for w in dev if w != chosen), key=lambda w: mean(dev[w]))
    vs_none, vs_runner = (paired(f'win-{o}-s1', f'win-{chosen}-s1', sorted(REF)) for o in ('none', runner))
    md += ['', f'- Paired test, seed 1: {chosen} minus none {fmt(vs_none)}, chrF {fmt(vs_none, "chrf")}; '
               f'{chosen} minus {runner} {fmt(vs_runner)}', '']
    js['window'] = {'dev': dev, 'test': test, 'chosen': chosen, 'vs_none': vs_none, 'vs_runner_up': vs_runner}

# ---- the main model
if MAIN:
    md += ['## The main model (all of train, 1,351 h)', '', '| Run | dev BLEU-4 | test BLEU-4 [95%] | test chrF [95%] | '
           'beam, bias | best epoch |', '|---|---|---|---|---|---|']
    for r in MAIN:
        x = res(r)
        md.append(f'| {r} | {x["dev"]["all"]["bleu"]:.2f} | {x["test"]["all"]["bleu"]:.2f} {ci(x["test"]["all"])} | '
                  f'{x["test"]["all"]["chrf"]:.2f} {ci(x["test"]["all"], "chrf")} | {x["chosen"]["beam"]}, '
                  f'{x["chosen"]["eos_bias"]} | {x["checkpoint_epoch"]} |')
    tb, tc = [res(r)['test']['all']['bleu'] for r in MAIN], [res(r)['test']['all']['chrf'] for r in MAIN]
    md += [f'| mean +- SD | | **{mean(tb):.2f} +- {sd(tb):.2f}** | **{mean(tc):.2f} +- {sd(tc):.2f}** | | |', '',
           '| Test subset | sentences | BLEU-4 (mean) | chrF (mean) |', '|---|---|---|---|']
    for g in sorted(k for k in res(MAIN[0])['test'] if k != 'all'):
        xs = [res(r)['test'][g] for r in MAIN]
        md.append(f'| {g} | {xs[0]["sentences"]:,} | {mean([x["bleu"] for x in xs]):.2f} | {mean([x["chrf"] for x in xs]):.2f} |')
    md.append('')
    js['main'] = {'test_bleu': tb, 'test_chrf': tc}

# ---- the learning curve
pts = [(v, [r for r in rs if have(r)]) for v, rs in (('lc12', [f'win-{WINDOW}-s{s}' for s in (1, 2, 3)]),
                                                     ('lc25', ['lc25-s1']), ('lc50', ['lc50-s1']), ('std', MAIN))]
pts = [(v, HOURS[v], mean([res(r)['test']['all']['bleu'] for r in rs]), mean([res(r)['test']['all']['chrf'] for r in rs]), len(rs))
       for v, rs in pts if rs]
if len(pts) > 1:
    md += ['## The learning curve (nested train sets; the same dev and test; 100 epochs each)', '',
           '| Train | hours | runs | test BLEU-4 | test chrF | x per doubling |', '|---|---|---|---|---|---|']
    for i, (v, h, b, c, n) in enumerate(pts):
        md.append(f'| {v} | {h:,.1f} | {n} | {b:.2f} | {c:.2f} | ' + (f'x{(b / pts[i - 1][2]) ** (1 / math.log2(h / pts[i - 1][1])):.3f}' if i else '') + ' |')
    md += ['', f'Over the curve: x{(pts[-1][2] / pts[0][2]) ** (1 / math.log2(pts[-1][1] / pts[0][1])):.3f} per doubling.', '']
    js['learning_curve'] = pts

# ---- unseen signers
if have('hold-i6i7-s1') and MAIN:
    i67 = [n for n in REF if V1[broadcast_of(n)]['i6i7']]
    a, c = paired('main-s1', 'hold-i6i7-s1', i67), paired('main-s1', 'hold-i6i7-s1', [n for n in REF if not V1[broadcast_of(n)]['i6i7']])
    allr = res('hold-i6i7-s1')['test']['i6i7=True']
    md += ['## Unseen signers: I6 and I7 left out', '',
           f'- Their {a["broadcasts"]} broadcasts of v1.0 test ({a["sentences"]:,} sentences): seen (main-s1) {a["bleu"]["a"]:.2f}, '
           f'unseen {a["bleu"]["b"]:.2f}: {fmt(a)}',
           f'- All their broadcasts ({allr["sentences"]:,} sentences), unseen: **{allr["bleu"]:.2f}** {ci(allr)}, chrF {allr["chrf"]:.2f}',
           f'- The rest of v1.0 test (check): {fmt(c)}', '']
    js['hold-i6i7'] = {'seen_vs_unseen': a, 'rest': c, 'all': allr}
if all(have(r) for r in ('hold-i2-s1', 'hold-i2-ctrl-s1')) and MAIN:
    i2, i1 = names(interpreter='I2'), names(interpreter='I1')
    sig, d2, d1 = paired('hold-i2-s1', 'hold-i2-ctrl-s1', i2), paired('hold-i2-ctrl-s1', 'main-s1', i2), paired('hold-i2-ctrl-s1', 'main-s1', i1)
    md += ['## Unseen signer at equal data: I2 left out (hold-i2) vs the same hours of others left out (ctrl)', '',
           '| On | main-s1 | ctrl | hold-i2 |', '|---|---|---|---|']
    md += [f'| {lab} ({len(ns):,}) | ' + ' | '.join(f'{score(r, ns):.2f}' for r in ('main-s1', 'hold-i2-ctrl-s1', 'hold-i2-s1')) + ' |'
           for lab, ns in (('I2', i2), ('I1', i1))]
    md += ['', f'- Cost of an unseen signer (ctrl - hold-i2, on I2): **{fmt(sig)}**, chrF {fmt(sig, "chrf")}',
           f'- 20% less data (main - ctrl): on I2 {fmt(d2)}, on I1 {fmt(d1)}', '']
    js['hold-i2'] = {'unseen_signer': sig, 'data_I2': d2, 'data_I1': d1}

# ---- time
if have('chrono-s1'):
    t = res('chrono-s1')['test']
    md += ['## Time: trained on each channel\'s earliest 90% of hours', '',
           '| Channel | within-period test [95%] | future test [95%] | future - within |', '|---|---|---|---|']
    for ch in CHANNELS + ('all',):
        w, f = (t[f'period={p}' + ('' if ch == 'all' else f',channel={ch}')] for p in ('within', 'future'))
        md.append(f'| {ch} | {w["bleu"]:.2f} {ci(w)} | {f["bleu"]:.2f} {ci(f)} | {f["bleu"] - w["bleu"]:+.2f} |')
    md += ['', 'Different sentences: not paired.', '']
    js['chrono'] = {k: v for k, v in t.items() if k.startswith('period')}

# ---- unseen channels
held = [c for c in CHANNELS if have(f'hold-{c.lower()}-s1')]
if held and MAIN and have('lc50-s1'):
    md += ['## Unseen channels: one channel left out of train and dev', '',
           '| Left out | train hours | on it, main-s1 -> hold (paired) | expected had it been seen | cost of the channel being new | other channels (paired) | expected there |',
           '|---|---|---|---|---|---|---|']
    js['channel'] = {}
    for c in held:
        run, h = f'hold-{c.lower()}-s1', HOURS[f'hold-{c.lower()}']
        on, oth = names(channel=c), sorted(n for n in REF if V1[broadcast_of(n)]['channel'] != c)
        p, q = paired('main-s1', run, on), paired('main-s1', run, oth)
        e, eo = expected(on, h), expected(oth, h) - mean([score(r, oth) for r in MAIN])
        md.append(f'| {c} | {h:,.0f} | {p["bleu"]["a"]:.2f} -> {p["bleu"]["b"]:.2f}, {fmt(p)} | {e:.2f} | **{e - p["bleu"]["b"]:+.2f}** | {fmt(q)} | {eo:+.2f} |')
        js['channel'][c] = {'on': p, 'expected_on': e, 'others': q}
    md += ['', '| Left out | interpreter | sentences | main-s1 -> hold | expected | cost |', '|---|---|---|---|---|---|']
    for c in held:
        for i in sorted({V1[broadcast_of(n)]['interpreter'] for n in names(channel=c)}):
            ns = names(channel=c, interpreter=i)
            p, e = paired('main-s1', f'hold-{c.lower()}-s1', ns), expected(ns, HOURS[f'hold-{c.lower()}'])
            md.append(f'| {c} | {i} | {len(ns):,} | {p["bleu"]["a"]:.2f} -> {p["bleu"]["b"]:.2f} | {e:.2f} | {e - p["bleu"]["b"]:+.2f} |')
    md.append('')
    if 'ProTV' in held and all(have(r) for r in ('hold-i2-s1', 'hold-i2-ctrl-s1')):
        md += ['### ProTV: a new signer vs a new channel (paired, the same sentences)', '',
               '| On | main-s1 | ctrl (I2, ProTV seen) | hold-i2 (I2 new) | hold-protv (I2, ProTV new) |', '|---|---|---|---|---|']
        for lab in ('I2', 'I1'):
            ns = names(interpreter=lab)
            md.append(f'| {lab} | ' + ' | '.join(f'{score(r, ns):.2f}' for r in ('main-s1', 'hold-i2-ctrl-s1', 'hold-i2-s1', 'hold-protv-s1')) + ' |')
            p = paired('hold-protv-s1', 'hold-i2-s1', ns)
            drop = expected(ns, HOURS['hold-i2']) - expected(ns, HOURS['hold-protv'])
            js.setdefault('protv', {})[lab] = {'hold_i2_minus_hold_protv': p, 'data_drop': drop}
        p2, d2 = js['protv']['I2']['hold_i2_minus_hold_protv'], js['protv']['I2']['data_drop']
        md += ['', f'- On I2: a new signer costs {js["hold-i2"]["unseen_signer"]["bleu"]["diff"]:.2f} (ctrl -> hold-i2); the new '
                   f'channel adds {p2["bleu"]["diff"] - d2:.2f} (hold-i2 -> hold-protv {fmt(p2)}, minus {d2:.2f} for 1,078 -> 808 h)', '']

# ---- per-broadcast feature normalisation
if all(have(r) for r in GROUPS['norm']) and have('hold-i2-s1') and have('hold-i2-ctrl-s1'):
    i2, i1 = names(interpreter='I2'), names(interpreter='I1')
    md += ['## Per-broadcast feature normalisation, on the I2 pair', '',
           '| On | ctrl | ctrl-norm | hold-i2 | hold-i2-norm |', '|---|---|---|---|---|']
    md += [f'| {lab} | ' + ' | '.join(f'{score(r, ns):.2f}' for r in ('hold-i2-ctrl-s1', 'hold-i2-ctrl-norm-s1', 'hold-i2-s1', 'hold-i2-norm-s1')) + ' |'
           for lab, ns in (('I2', i2), ('I1', i1), ('all', sorted(REF)))]
    seen, unseen = paired('hold-i2-ctrl-s1', 'hold-i2-ctrl-norm-s1', i2), paired('hold-i2-s1', 'hold-i2-norm-s1', i2)
    gap = paired('hold-i2-norm-s1', 'hold-i2-ctrl-norm-s1', i2)
    md += ['', f'- Seen I2 (ctrl-norm - ctrl): {fmt(seen)}; unseen I2 (hold-i2-norm - hold-i2): **{fmt(unseen)}**',
           f'- Unseen-signer gap with normalisation: {fmt(gap)} (without: {js["hold-i2"]["unseen_signer"]["bleu"]["diff"]:.2f})'
           if 'hold-i2' in js else '', '']
    js['norm'] = {'seen': seen, 'unseen': unseen, 'gap': gap}

(OUT / 'analysis.json').write_text(json.dumps(js, indent=1))
(OUT / 'analysis.md').write_text('# RSL-News experiments: results\n\nBLEU-4 and chrF: sacreBLEU 2.6.0, default settings.\n\n' + '\n'.join(md) + '\n')
print('\n'.join(md))
