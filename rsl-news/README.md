# RSL-News experiments

Every baseline experiment on RSL-News: Signformer on the BSL-5k (I3D) features. The scripts below reproduce every
number, from the released dataset to the tables.

| File | Does |
|---|---|
| `splits.py` | the split variants each experiment trains and tests on |
| `configs.py`, `base.yaml` | every run (name, split variant, time window, seed) and its config |
| `stats.py` | per-broadcast feature statistics (only for the `norm` runs) |
| `run.sh` | trains and evaluates runs, K at a time |
| `evaluate.py` | chooses the decoding on dev, then scores dev and test (BLEU-4, chrF, 95% intervals, per subset) |
| `analyze.py` | every results table: `OUT/analysis.md` |

## Setup

- A GPU with at least 24 GB. Each run used ~18 GB at batch 256; three ran side by side on one H200.
- `pip install -r requirements.txt` in the Signformer folder (Python 3.10). The results were made with torch
  2.11.0+cu130, sacreBLEU 2.6.0 and rsl-news-loader 1.2.0.
- The RSL-News release in a folder `$RSL_NEWS`, holding `manifests/`, `dataset/` (labels and `.bsl5k.npz` features)
  and `tokenizer/`. `splits.py` checks that the manifests are the version used here.

## Run

From the Signformer folder, with `$OUT` an empty folder for everything the runs write:

```bash
python rsl-news/splits.py $RSL_NEWS $OUT                    # split variants: seconds
python rsl-news/configs.py $OUT                             # the 32 run configs
python rsl-news/stats.py $RSL_NEWS $OUT/bsl5k_stats.npz     # only for the norm runs: minutes
rsl-news/run.sh $OUT 3 window main scale signer time channel norm
python rsl-news/analyze.py $OUT                             # -> $OUT/analysis.md
```

- `run.sh` takes groups or run names, and skips runs that are already evaluated.
- A run is trained by `python -m main train $OUT/configs/<run>.yaml` and evaluated by
  `python rsl-news/evaluate.py $OUT/runs/<run>`; its log is in `$OUT/logs/`.
- An interrupted training continues from its last epoch with `python -m main train $OUT/configs/<run>.yaml --resume`.

All 32 runs take about 5 days on one H200, three at a time. A run on 12.5% of the training data trains in ~3.5 h; one
on all of it in ~16 h. Evaluation takes 1-5 h; the window runs take longest, since they are decoded on 32 test
windows.

## The experiments

`s`, `e`: a sentence's start and end timestamps. Every run trains for 100 epochs and keeps the checkpoint with the
best dev BLEU-4 (greedy decoding). The test set is the release's test split (42,047 sentences), except where named.

| Group | Runs | Question | Result |
|---|---|---|---|
| `window` | 8 time windows x 2 seeds (3 for `jit`, `jit-long`), on 12.5% of train | Which stretch of video around a sentence to translate | `jit-long`: train on [s, e + random 1-5 s], test on [s, e + 3 s]: 8.08 BLEU-4 vs 5.65 for [s, e] |
| `main` | 3 seeds, all of train | The main model | **13.85 +- 0.03 BLEU-4, 35.83 +- 0.10 chrF** |
| `scale` | 25%, 50% of train (12.5%: `win-jit-long`) | The learning curve | x1.32, x1.17, x1.11 BLEU-4 per doubling of the data |
| `signer` | I6 and I7 left out; I2 left out; the same hours of other signers left out | Unseen signers | I2 unseen: 13.64 -> 5.79 BLEU-4 at equal data; I6 and I7 unseen: 5.05 |
| `time` | each channel's latest 10% of hours left out (5% dropped as a buffer) | Drift over time | little: a signer scores the same or better later on (except I2, -1.9) |
| `channel` | each channel left out of train and dev | Unseen channels | 5-6 BLEU-4 on the unseen channel; on ProTV, ~3/4 of the drop is the new signer, ~1/4 the channel |
| `norm` | the I2 pair, features normalised per broadcast | Does per-broadcast normalisation help unseen signers? | (running) |

All other runs use the `jit-long` window. `analyze.py` chooses it from the `window` runs: the highest mean dev BLEU-4,
unless a simpler window is within one standard error of the seed variance pooled over all windows.

## Reproducibility

- Each run's data order and random time windows depend only on its seed, the epoch and the sentence. GPU kernels
  are not deterministic, so a rerun matches to within run-to-run noise, not bit for bit. The main model's three
  seeds differ by 0.03 BLEU-4 on test; on 12.5% of the data, seeds differ by ~0.2.
- One difference from the original runs: the first 13 `window` runs (seeds 1 and 2 of none, fixed, jit,
  jit-notrim and jit-long; seed 1 of flat5, jit-xlong and jit-long-trim) validated every epoch. Here, every run
  validates every 5th epoch up to epoch 50. Validation uses no randomness, so the training is the same; only the
  choice of `best.ckpt` could differ, and the earliest best epoch of any run was 55.
- Decoding choices (beam, end-of-sentence bias) are made on a fixed 5,000-sentence dev subset per split variant; nothing
  is tuned on test.
