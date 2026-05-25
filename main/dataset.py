# coding: utf-8
"""
Dataset module - PyTorch Dataset for sign language translation
with confidence ramps.
"""
import json
import numpy as np
import torch
from torch.utils.data import Dataset

# Post-delay: extend past annotated end (no pre-extension)
POST_DELAY_RANGE = (1.0, 5.0)  # train: uniform in seconds
FIXED_POST_DELAY = 4.0         # eval: fixed seconds


def _build_conf_ramp(T, ann_end_local, post_ext_len, start_fade_len, rng, is_train):
    """Build a per-frame confidence ramp of shape (T,).

    - Start ramp: fade-in over start_fade_len frames (boundary uncertainty).
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


class SignTranslationDataset(Dataset):
    """Dataset that loads sign features, phoneme CTC logits, and text tokens,
    with confidence ramps and randomised post-extension."""

    def __init__(self, file_triplets, bos_id, eos_id, fps=12.5,
                 max_sgn_len=256, max_txt_len=128, train=False, seed=42):
        self.fps = fps
        self.max_sgn_len = max_sgn_len
        self.max_txt_len = max_txt_len
        self.train = train
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.seed = seed
        self.epoch = 0

        # Build sentence-level index, sorted by start time per episode
        # so that prev_end is correct.
        self.sentences = []
        for npy_path, phoneme_path, json_path in file_triplets:
            with open(json_path, "r", encoding="utf-8") as f:
                entries = json.load(f)
            # Sort keeping original indices for phoneme npz keys
            indexed = sorted(enumerate(entries), key=lambda x: x[1]["start"])
            for pos, (seg_idx, entry) in enumerate(indexed):
                prev_end = indexed[pos - 1][1]["end"] if pos > 0 else None
                self.sentences.append({
                    "npy_path": npy_path,
                    "phoneme_path": phoneme_path,
                    "seg_idx": seg_idx,
                    "start": entry["start"],
                    "end": entry["end"],
                    "tokens": entry["tokens_lower"],
                    "text": entry["text_lower"],
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
        ann_end_frame = round(s["end"] * self.fps)

        if is_train:
            post_delay_sec = rng.uniform(*POST_DELAY_RANGE)
        else:
            post_delay_sec = FIXED_POST_DELAY
        post_delay_frames = round(post_delay_sec * self.fps)

        # Load features via mmap and extract the segment
        features = np.load(s["npy_path"], mmap_mode="r")
        total_frames = features.shape[0]

        start_frame = max(0, min(total_frames, round(s["start"] * self.fps)))
        end_frame = min(total_frames, ann_end_frame + post_delay_frames)

        if end_frame <= start_frame:
            sgn = np.zeros((1, features.shape[1]), dtype=np.float32)
            start_frame = 0
            end_frame = 1
            ann_end_frame = 0
        else:
            sgn = features[start_frame:end_frame].astype(np.float32)

        del features

        # Load phoneme CTC logits from npz (per-segment)
        npz = np.load(s["phoneme_path"], mmap_mode="r")
        seg_logits = npz[f"logits_{s['seg_idx']}"][:, :-1]  # (padded_T, 38) — drop bos/eos
        seg_end = int(npz["ends"][s["seg_idx"]])     # actual end frame
        del npz

        # Phoneme range mirrors sign feature post-extension
        phn_end = min(round(seg_end + post_delay_sec * self.fps), seg_logits.shape[0])
        phonemes = seg_logits[:phn_end].astype(np.float32)
        if phonemes.shape[0] == 0:
            phonemes = np.zeros((1, seg_logits.shape[1]), dtype=np.float32)

        # Match phoneme length to sgn length (they come from different extractors)
        if phonemes.shape[0] != sgn.shape[0]:
            p_indices = np.linspace(0, phonemes.shape[0] - 1, sgn.shape[0], dtype=int)
            phonemes = phonemes[p_indices]

        # --- Confidence ramp ---
        ann_end_local = ann_end_frame - start_frame
        post_ext_len = max(0, end_frame - ann_end_frame)

        # Gap to previous sentence for boundary ramp
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

        boundary_ramp_frames = round(boundary_ramp_sec * self.fps)
        boundary_confident_at = round(boundary_ramp_frames * boundary_confident_frac)
        boundary_fade_len = boundary_ramp_frames - boundary_confident_at

        T = sgn.shape[0]
        conf_ramp = _build_conf_ramp(
            T, ann_end_local, post_ext_len, boundary_fade_len, rng, is_train
        )
        sgn *= conf_ramp[:, np.newaxis]

        # Subsample if longer than max_sgn_len
        if sgn.shape[0] > self.max_sgn_len:
            if is_train:
                indices = np.sort(
                    rng.choice(sgn.shape[0], size=self.max_sgn_len, replace=False)
                )
            else:
                indices = np.linspace(0, sgn.shape[0] - 1, self.max_sgn_len, dtype=int)
            sgn = sgn[indices]
            phonemes = phonemes[indices]

        # Build token sequence: [BOS] + tokens + [EOS], truncate to max_txt_len
        tokens = [self.bos_id] + s["tokens"] + [self.eos_id]
        if len(tokens) > self.max_txt_len:
            tokens = tokens[: self.max_txt_len - 1] + [self.eos_id]

        sgn_tensor = torch.from_numpy(sgn)
        txt_tensor = torch.tensor(tokens, dtype=torch.long)
        phn_tensor = torch.from_numpy(phonemes)
        return sgn_tensor, txt_tensor, phn_tensor

    @property
    def txt_references(self):
        return [s["text"] for s in self.sentences]
