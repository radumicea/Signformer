# coding: utf-8
"""
Dataset module - PyTorch Dataset for sign language translation.
"""
import json
import numpy as np
import torch
from torch.utils.data import Dataset


class SignTranslationDataset(Dataset):
    """Dataset that loads sign features, phoneme CTC logits, and text tokens."""

    def __init__(self, file_triplets, bos_id, eos_id, fps=12.5,
                 max_sgn_len=256, max_txt_len=128, train=False):
        self.fps = fps
        self.max_sgn_len = max_sgn_len
        self.max_txt_len = max_txt_len
        self.train = train
        self.bos_id = bos_id
        self.eos_id = eos_id

        # Build sentence-level index
        self.sentences = []
        for npy_path, phoneme_path, json_path in file_triplets:
            with open(json_path, "r", encoding="utf-8") as f:
                entries = json.load(f)
            for seg_idx, entry in enumerate(entries):
                self.sentences.append({
                    "npy_path": npy_path,
                    "phoneme_path": phoneme_path,
                    "seg_idx": seg_idx,
                    "start": entry["start"],
                    "end": entry["end"],
                    "tokens": entry["tokens_lower"],
                    "text": entry["text_lower"],
                })

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        s = self.sentences[idx]

        # Load features via mmap and extract the segment
        features = np.load(s["npy_path"], mmap_mode="r")
        total_frames = features.shape[0]

        if self.train:
            skip_start = np.random.uniform(0.0, 1.5)
            extra_end = np.random.uniform(2.0, 4.0)
        else:
            skip_start = 0.0
            extra_end = 3.0

        start_frame = min(round((s["start"] + skip_start) * self.fps), total_frames)
        end_frame = min(round((s["end"] + extra_end) * self.fps), total_frames)
        sgn = features[start_frame:end_frame].astype(np.float32)

        if sgn.shape[0] == 0:
            sgn = np.zeros((1, features.shape[1]), dtype=np.float32)

        # Load phoneme CTC logits from npz (per-segment)
        npz = np.load(s["phoneme_path"])
        seg_logits = npz[f"logits_{s['seg_idx']}"][:, :-1]  # (padded_T, 38) — drop bos/eos
        seg_end = int(npz["ends"][s["seg_idx"]])     # actual end frame

        # The logits are padded beyond seg_end (up to ~5s extra).
        # Use skip_start / extra_end to mirror the sign feature jitter.
        phn_start = min(round(skip_start * self.fps), seg_end)
        phn_end = min(round(seg_end + extra_end * self.fps), seg_logits.shape[0])
        phonemes = seg_logits[phn_start:phn_end].astype(np.float32)
        if phonemes.shape[0] == 0:
            phonemes = np.zeros((1, seg_logits.shape[1]), dtype=np.float32)

        # Match phoneme length to sgn length (they come from different extractors)
        if phonemes.shape[0] != sgn.shape[0]:
            p_indices = np.linspace(0, phonemes.shape[0] - 1, sgn.shape[0], dtype=int)
            phonemes = phonemes[p_indices]

        # Subsample if longer than max_sgn_len
        if sgn.shape[0] > self.max_sgn_len:
            if self.train:
                indices = np.sort(
                    np.random.choice(sgn.shape[0], size=self.max_sgn_len, replace=False)
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
