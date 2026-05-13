# coding: utf-8
"""
Dataset module — PyTorch Dataset for sign language translation
with confidence ramps and optional previous-sentence visual context.

Port of the OLD RSL_News_Dataset logic, cleaned up.
"""
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

FEATURE_FPS = 12.5  # 25 fps / stride 2

# Post-delay: extend past annotated end (no pre-extension)
POST_DELAY_RANGE = (2.0, 5.0)  # train: uniform in seconds
FIXED_POST_DELAY = 3.5         # eval: fixed seconds

# Context gap: max gap (sec) to previous sentence for providing context
CONTEXT_GAP_MAX = 5.0
CONTEXT_PROB = 0.5  # train: probability of including prev-sentence context


def _sec_to_feat(sec):
    """Convert seconds to feature-frame index."""
    return round(sec * FEATURE_FPS)


def _build_conf_ramp(T, ann_end_local, post_ext_len, start_fade_len, rng, is_train):
    """Build a per-frame confidence ramp of shape (T,).

    - Start ramp: fade-in over start_fade_len frames (boundary uncertainty
      shared with context's end ramp — see _build_ctx_ramp).
    - Post-extension ramp: fade-out after annotated end.
    """
    conf = np.ones(T, dtype=np.float32)

    # --- Start ramp (fade-in) ---
    if start_fade_len > 0 and start_fade_len <= T:
        conf[:start_fade_len] = np.linspace(0.0, 1.0, start_fade_len, endpoint=False)

    # --- Post-extension ramp (fade-out after annotated end) ---
    if post_ext_len > 0 and ann_end_local < T:
        post_confident_frac = rng.uniform(0.3, 0.7) if is_train else 0.5
        confident_frames = round(post_ext_len * post_confident_frac)
        ramp_start = ann_end_local + confident_frames
        if ramp_start < T:
            decay_len = T - ramp_start
            conf[ramp_start:T] = np.linspace(1.0, 0.0, decay_len, endpoint=True)

    return conf


def _build_ctx_ramp(T, end_fade_len):
    """Build per-frame confidence ramp for previous-sentence visual context.

    - End: gap-dependent fade-out that mirrors current sentence's start ramp,
      modeling the same boundary uncertainty from the opposite side.

    The end fade uses the identical duration as the current sentence's start
    fade (computed from the same random draw), so the two confidence profiles
    are complementary at the shared boundary.

    No start fade — the context's own beginning has no special relationship
    to the current sentence, and the features get compressed via attention
    pooling anyway.
    """
    conf = np.ones(T, dtype=np.float32)
    if T <= 1:
        return conf
    end_fade_len = min(end_fade_len, T)
    if end_fade_len > 0:
        conf[-end_fade_len:] = np.linspace(1.0, 0.0, end_fade_len, endpoint=False)
    return conf


class SignTranslationDataset(Dataset):
    """Episode-level dataset with confidence ramps and optional previous-sentence
    visual context.

    Each item = one sentence, features sliced from the episode NPY at runtime.
    Splits are at episode level (deterministic) to prevent data leakage.

    Augmentations:
    - No pre-extension: start always at annotated boundary.
    - Post-extension past annotated end with configurable delay.
    - Confidence ramp: fade-in at start (gap-dependent), fade-out after end.
    - Previous sentence visual context: optionally included (stochastic in train),
      compressed by the model via attention pooling.
    """

    def __init__(
        self,
        file_pairs,
        bos_id,
        eos_id,
        fps=12.5,
        max_sgn_len=256,
        max_txt_len=128,
        train=False,
        seed=42,
        context_prob=CONTEXT_PROB,
        context_gap_max=CONTEXT_GAP_MAX,
    ):
        super().__init__()
        self.fps = fps
        self.max_sgn_len = max_sgn_len
        self.max_txt_len = max_txt_len
        self.train = train
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.seed = seed
        self.epoch = 0
        self.context_prob = context_prob
        self.context_gap_max = context_gap_max

        # Build flat sample list with previous-sentence boundaries pre-extracted.
        # Sorted by start time per episode so prev_start/prev_end are correct.
        self.sentences = []
        for npy_path, json_path in file_pairs:
            with open(json_path, "r", encoding="utf-8") as f:
                entries = json.load(f)
            entries.sort(key=lambda e: e["start"])
            for i, entry in enumerate(entries):
                prev_start = entries[i - 1]["start"] if i > 0 else None
                prev_end = entries[i - 1]["end"] if i > 0 else None
                self.sentences.append({
                    "npy_path": npy_path,
                    "start": entry["start"],
                    "end": entry["end"],
                    "tokens": entry["tokens_lower"],
                    "text": entry["text_lower"],
                    "prev_start": prev_start,
                    "prev_end": prev_end,
                })

    def set_epoch(self, epoch: int):
        """Set epoch for deterministic per-sample RNG."""
        self.epoch = epoch

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        s = self.sentences[idx]

        # Deterministic RNG per (seed, epoch, sample_index)
        rng = np.random.default_rng(self.seed + self.epoch * len(self.sentences) + idx)
        is_train = self.train

        # --- Frame range: start at annotation, extend past end ---
        start_frame = _sec_to_feat(s["start"])
        ann_end_frame = _sec_to_feat(s["end"])

        if is_train:
            post_delay_sec = rng.uniform(*POST_DELAY_RANGE)
        else:
            post_delay_sec = FIXED_POST_DELAY
        post_delay_frames = _sec_to_feat(post_delay_sec)

        # Load episode features (mmap)
        X_ref = np.load(s["npy_path"], mmap_mode="r")
        N = X_ref.shape[0]

        end_frame = min(N, ann_end_frame + post_delay_frames)
        start_frame = max(0, min(N, start_frame))

        if end_frame <= start_frame:
            if N > 0:
                safe_start = max(0, min(N - 1, start_frame))
                sgn = X_ref[safe_start:safe_start + 1].astype(np.float32)
                start_frame = safe_start
                end_frame = safe_start + 1
            else:
                feat_dim = X_ref.shape[1] if X_ref.ndim == 2 and X_ref.shape[1] > 0 else 1
                sgn = np.zeros((1, feat_dim), dtype=np.float32)
                start_frame = 0
                end_frame = 1
                ann_end_frame = 0
        else:
            # Copy slice as float32
            sgn = X_ref[start_frame:end_frame].astype(np.float32)

        # --- Confidence ramp ---
        ann_end_local = ann_end_frame - start_frame
        post_ext_len = max(0, end_frame - ann_end_frame)

        # Boundary ramp parameters: shared between current sentence's start
        # fade-in and context's end fade-out (same uncertainty, mirrored).
        if s["prev_end"] is not None:
            gap_sec = s["start"] - s["prev_end"]
        else:
            gap_sec = float("inf")

        ramp_max_sec = np.clip(1.5 - 0.26 * gap_sec, 0.2, 1.5)
        if is_train:
            boundary_ramp_sec = rng.uniform(0, ramp_max_sec)
            boundary_confident_frac = rng.uniform(0, 0.5)
        else:
            boundary_ramp_sec = ramp_max_sec * 0.5
            boundary_confident_frac = 0.25

        boundary_ramp_frames = _sec_to_feat(boundary_ramp_sec)
        boundary_confident_at = round(boundary_ramp_frames * boundary_confident_frac)
        boundary_fade_len = boundary_ramp_frames - boundary_confident_at

        T = sgn.shape[0]
        conf_ramp = _build_conf_ramp(
            T, ann_end_local, post_ext_len, boundary_fade_len, rng, is_train
        )
        sgn *= conf_ramp[:, np.newaxis]

        # --- Subsample if too long ---
        if sgn.shape[0] > self.max_sgn_len:
            indices = np.sort(
                rng.choice(sgn.shape[0], size=self.max_sgn_len, replace=False)
            )
            sgn = sgn[indices]

        # --- Tokens ---
        tokens = list(s["tokens"])
        if len(tokens) == 0 or tokens[0] != self.bos_id:
            tokens = [self.bos_id] + tokens
        if tokens[-1] != self.eos_id:
            tokens.append(self.eos_id)
        if len(tokens) > self.max_txt_len:
            tokens = tokens[: self.max_txt_len - 1] + [self.eos_id]

        # --- Previous sentence visual context ---
        prev_sgn = None
        if s["prev_start"] is not None and self.context_prob > 0:
            gap = s["start"] - s["prev_end"]
            if gap <= self.context_gap_max:
                include = (not is_train) or (rng.random() < self.context_prob)
                if include:
                    ctx_start = _sec_to_feat(s["prev_start"])
                    # Clip to not overlap with current sentence
                    ctx_end = _sec_to_feat(min(s["prev_end"], s["start"]))
                    ctx_start = max(0, min(N, ctx_start))
                    ctx_end = min(N, ctx_end)
                    if ctx_end > ctx_start:
                        ctx_X = X_ref[ctx_start:ctx_end].astype(np.float32)
                        # Mirrored boundary ramp
                        ctx_X *= _build_ctx_ramp(ctx_X.shape[0], boundary_fade_len)[
                            :, np.newaxis
                        ]
                        # Subsample if too long
                        if ctx_X.shape[0] > self.max_sgn_len:
                            ctx_idx = np.sort(
                                rng.choice(
                                    ctx_X.shape[0],
                                    size=self.max_sgn_len,
                                    replace=False,
                                )
                            )
                            ctx_X = ctx_X[ctx_idx]
                        prev_sgn = ctx_X

        return {
            "sgn": sgn,
            "txt": tokens,
            "prev_sgn": prev_sgn,
            "text_raw": s["text"],
            "name": s["npy_path"].replace(".npy", ""),
        }

    @property
    def txt_references(self):
        return [s["text"] for s in self.sentences]
