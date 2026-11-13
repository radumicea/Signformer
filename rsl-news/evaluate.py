#!/usr/bin/env python3
"""Evaluates a finished run (README.md). From the Signformer folder:

    python rsl-news/evaluate.py OUT/runs/RUN [--workers N]

1. Loads RUN/best.ckpt and RUN/config.yaml.
2. Chooses the decoding on the variant's 5,000-sentence dev subset (OUT/splits/<variant>.devsub.json), at the run's
   test window (data.eval_timing): beam 1 / 4 / 8 (alpha 1.0) x end-of-sentence bias 0..7, the best BLEU-4 (ties: the
   smaller beam, then the smaller bias).
3. Decodes the full dev and test sets with it. Writes RUN/eval/: {dev,test}.{hyp,ref}.txt (name|text), grid.json and
   results.json: BLEU-4, chrF, length ratio, the share of outputs repeating a 3-word sequence, and 95% intervals from
   resampling whole broadcasts (1,000 draws); on test also per subset (channel, interpreter, ...;
   OUT/splits/subsets.json).
4. Runs with testing.window_grid: the dev subset decoded on 32 test windows, start + {-0.5, 0, 0.5, 1} s x end +
   {0, 1, 1.5, 2, 2.5, 3, 4, 5} s (window_grid.json).

Scores: sacreBLEU's BLEU-4 and chrF, default settings (main.metrics.scores)."""
import argparse
import copy
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))      # Signformer
from main.data import load_data, make_data_iter  # noqa: E402
from main.helpers import load_checkpoint, load_config  # noqa: E402
from main.loss import XentLoss  # noqa: E402
from main.metrics import scores as corpus_scores  # noqa: E402
from main.model import build_model  # noqa: E402
from main.prediction import validate_on_data, write_outputs  # noqa: E402
from sacrebleu.metrics import BLEU, CHRF  # noqa: E402

BEAMS, BIASES, ALPHA = (1, 4, 8), (0, 1, 2, 3, 4, 5, 6, 7), 1.0
STARTS, ENDS = (-0.5, 0, 0.5, 1), (0, 1, 1.5, 2, 2.5, 3, 4, 5)


class Subset(torch.utils.data.Dataset):
    """Some sentences of a SignTranslationDataset, as one."""

    def __init__(self, base, idx):
        self.base, self.idx = base, np.asarray(idx)
        self.sgn_lengths = base.sgn_lengths[self.idx]
        self.references = [base.references[i] for i in self.idx]
        self.names = [base.names[i] for i in self.idx]

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        _, sgn, txt = self.base[int(self.idx[i])]
        return i, sgn, txt


def broadcast_of(name):
    """'ProTV/2021/03/16/segment_1:12' -> 'ProTV/2021-03-16'; 'PrimaTV/<id>/segment_0:3' -> 'PrimaTV/<id>'."""
    channel, rest = name.split('/', 1)
    episode = rest.rsplit('/segment_', 1)[0]
    return f'{channel}/{episode}' if channel == 'PrimaTV' else f'{channel}/{episode.replace("/", "-")}'


def repeats_trigram(text):
    w = text.split()
    tri = list(zip(w, w[1:], w[2:]))
    return len(tri) != len(set(tri))


def scores(refs, hyps):
    return {**corpus_scores(refs, hyps), 'sentences': len(hyps),
            'repeated_trigram_share': float(np.mean([repeats_trigram(h) for h in hyps])) if hyps else 0.0}


def sentence_stats(refs, hyps):
    """Per-sentence BLEU and chrF statistics, summed over any set of sentences to score it."""
    bm, cm = BLEU(), CHRF()
    return (np.array(bm._extract_corpus_statistics(list(hyps), [list(refs)]), dtype=np.int64),
            np.array(cm._extract_corpus_statistics(list(hyps), [list(refs)]), dtype=np.int64), bm, cm)


def bootstrap(refs, hyps, names, draws=1000):
    """95% intervals of BLEU-4 and chrF, resampling whole broadcasts (seed 0)."""
    bs, cs, bm, cm = sentence_stats(refs, hyps)
    groups = defaultdict(list)
    for i, n in enumerate(names):
        groups[broadcast_of(n)].append(i)
    keys = sorted(groups)
    gb = np.stack([bs[groups[k]].sum(0) for k in keys]); gc = np.stack([cs[groups[k]].sum(0) for k in keys])
    rng = np.random.default_rng(0)
    b, c = [], []
    for _ in range(draws):
        pick = rng.integers(0, len(gb), len(gb))
        b.append(bm._compute_score_from_stats(list(gb[pick].sum(0))).score)
        c.append(cm._compute_score_from_stats(list(gc[pick].sum(0))).score)
    return {'bleu_ci95': [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
            'chrf_ci95': [float(np.percentile(c, 2.5)), float(np.percentile(c, 97.5))], 'broadcasts': len(gb)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run_dir', type=Path)
    ap.add_argument('--workers', type=int, default=7)
    args = ap.parse_args()
    run = args.run_dir.resolve()
    (run / 'eval').mkdir(exist_ok=True)
    cfg = load_config(str(run / 'config.yaml'))
    data_path = Path(cfg['data']['data_path'])
    splits, variant = data_path.parent / 'splits', data_path.name[len('data-'):]
    devsub = set(json.loads((splits / f'{variant}.devsub.json').read_text()))
    subsets = json.loads((splits / 'subsets.json').read_text())[variant]
    seed = cfg['training']['random_seed']

    data, vocab = load_data(cfg['data'], splits=['val', 'test'], seed=seed)
    model = build_model(cfg=cfg['model'], txt_vocab=vocab, sgn_dim=cfg['data']['feature_size'], multimodal=False)
    ckpt = load_checkpoint(str(run / 'best.ckpt'), use_cuda=True)
    model.load_state_dict(ckpt['model_state'])
    model.cuda()
    loss = XentLoss(pad_index=model.txt_pad_index, smoothing=0.0)

    def decode(dataset, beam, bias):
        it = make_data_iter(dataset, batch_size=cfg['training']['eval_batch_size'], pad_index=model.txt_pad_index,
                            num_workers=args.workers, use_cuda=True)
        return validate_on_data(model=model, data_iter=it, use_cuda=True, translation_loss_function=loss,
                                translation_max_output_length=cfg['training']['translation_max_output_length'],
                                translation_beam_size=beam, translation_beam_alpha=ALPHA if beam > 1 else -1,
                                translation_eos_bias=bias)['txt_hyp']

    def dev_subset(val):
        return Subset(val, [i for i, n in enumerate(val.names) if n in devsub])

    sub = dev_subset(data['val'])
    grid = []
    for beam in BEAMS:
        for bias in BIASES:
            s = scores(sub.references, decode(sub, beam, bias))
            grid.append({'beam': beam, 'alpha': ALPHA if beam > 1 else None, 'eos_bias': bias, **s})
            print(f'beam {beam} bias {bias}: BLEU-4 {s["bleu"]:.2f}, length ratio {s["len_ratio"]:.3f}', flush=True)
    best = max(grid, key=lambda g: (round(g['bleu'], 6), -g['beam'], -g['eos_bias']))
    (run / 'eval' / 'grid.json').write_text(json.dumps({'grid': grid, 'chosen': best}, indent=1))
    beam, bias = best['beam'], best['eos_bias']

    results = {'run': run.name, 'variant': variant, 'checkpoint_epoch': ckpt.get('best_ckpt_epoch'),
               'train_timing': cfg['data'].get('train_timing'), 'eval_timing': cfg['data'].get('eval_timing'),
               'chosen': {'beam': beam, 'alpha': ALPHA if beam > 1 else None, 'eos_bias': bias}}
    for split, name in (('val', 'dev'), ('test', 'test')):
        ds = data[split]
        hyp = decode(ds, beam, bias)
        write_outputs(str(run / 'eval' / f'{name}.hyp.txt'), ds.names, hyp)
        write_outputs(str(run / 'eval' / f'{name}.ref.txt'), ds.names, ds.references)
        r = {'all': {**scores(ds.references, hyp), **bootstrap(ds.references, hyp, ds.names)}}
        if split == 'test':
            groups = defaultdict(list)
            for i, n in enumerate(ds.names):
                info = subsets[broadcast_of(n)]
                keys = [f'channel={info["channel"]}', f'interpreter={info["interpreter"]}']
                if variant == 'hold-i6i7':
                    keys += [f'i6i7={info["i6i7"]}', f'i6i7={info["i6i7"]},in_v1_test={info["in_v1_test"]}']
                if variant == 'chrono':
                    keys += [f'period={info["period"]}', f'period={info["period"]},channel={info["channel"]}']
                for k in keys:
                    groups[k].append(i)
            for g, idx in sorted(groups.items()):
                rr, hh, nn = [ds.references[i] for i in idx], [hyp[i] for i in idx], [ds.names[i] for i in idx]
                r[g] = {**scores(rr, hh), **bootstrap(rr, hh, nn)}
        results[name] = r
        results['signatures'] = {'bleu': r['all']['bleu_signature'], 'chrf': r['all']['chrf_signature']}
        print(f'{name}: BLEU-4 {r["all"]["bleu"]:.2f} {r["all"]["bleu_ci95"]}, chrF {r["all"]["chrf"]:.2f}', flush=True)

    if cfg['testing'].get('window_grid'):
        wg = []
        for start in STARTS:
            for end in ENDS:
                dcfg = copy.deepcopy(cfg['data'])
                dcfg['eval_timing'] = {'before': -start, 'after': end}
                val = dev_subset(load_data(dcfg, splits=['val'], seed=seed)[0]['val'])
                wg.append({'start_plus': start, 'end_plus': end, **scores(val.references, decode(val, beam, bias))})
                print(f'window start+{start} end+{end}: BLEU-4 {wg[-1]["bleu"]:.2f}', flush=True)
        (run / 'eval' / 'window_grid.json').write_text(json.dumps(wg, indent=1))
    (run / 'eval' / 'results.json').write_text(json.dumps(results, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
