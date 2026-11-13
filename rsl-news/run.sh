#!/bin/bash
# Trains and evaluates runs, K at a time (README.md). From the Signformer folder:
#
#     rsl-news/run.sh OUT K GROUP|RUN ...
#
# A run whose OUT/runs/<run>/eval/results.json exists is skipped. Logs: OUT/logs/<run>.log.
set -u
OUT=$(realpath "$1"); K=$2; shift 2
mkdir -p "$OUT/logs"
export OUT PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python rsl-news/configs.py "$OUT" --list "$@" | xargs -P "$K" -I{} bash -c '
  [ -f "$OUT/runs/{}/eval/results.json" ] && exit 0
  { python -m main train "$OUT/configs/{}.yaml" && python rsl-news/evaluate.py "$OUT/runs/{}"; } > "$OUT/logs/{}.log" 2>&1 \
    && echo "{}: done" || echo "{}: FAILED (see $OUT/logs/{}.log)"'
