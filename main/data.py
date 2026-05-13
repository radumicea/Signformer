# coding: utf-8
"""
Data module — loads and splits dataset, creates data loaders with collation.
"""
import glob
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from main.dataset import SignTranslationDataset, CONTEXT_PROB, CONTEXT_GAP_MAX
from main.vocabulary import Vocabulary, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN


def _find_file_pairs(data_path):
    """Find all matching .npy / .json file pairs recursively."""
    json_files = sorted(
        glob.glob(os.path.join(data_path, "**", "*.json"), recursive=True)
    )
    pairs = []
    for json_path in json_files:
        npy_path = json_path.replace(".json", ".npy")
        if os.path.exists(npy_path):
            pairs.append((npy_path, json_path))
    return pairs


def _pad_2d(seqs, max_T, feat_size):
    """Pad a list of 2-D arrays/tensors to (max_T, feat_size) and stack."""
    out = []
    for x in seqs:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        T = x.shape[0]
        if T >= max_T:
            out.append(x[:max_T])
        else:
            pad = torch.zeros((max_T - T, feat_size), dtype=x.dtype)
            out.append(torch.cat([x, pad], dim=0))
    return torch.stack(out, dim=0)


class Collator:
    """Collate dict-based samples into a padded batch dict."""

    def __init__(self, pad_id, feature_size):
        self.pad_id = pad_id
        self.feature_size = feature_size

    def __call__(self, batch):
        batch = [x for x in batch if x is not None]
        if not batch:
            return None

        # --- Target text ---
        txt_ids_list = [ex["txt"] for ex in batch]
        max_txt = max(len(x) for x in txt_ids_list)
        txt_padded = torch.tensor(
            [x + [self.pad_id] * (max_txt - len(x)) for x in txt_ids_list],
            dtype=torch.long,
        )
        txt_len = torch.tensor([len(x) for x in txt_ids_list], dtype=torch.long)

        # --- Sign features ---
        sgn_list = [ex["sgn"] for ex in batch]
        max_T = max(x.shape[0] for x in sgn_list)
        sgn_padded = _pad_2d(sgn_list, max_T, self.feature_size)
        sgn_len = torch.tensor([x.shape[0] for x in sgn_list], dtype=torch.long)

        # --- Previous sentence visual context ---
        prev_sgn_list = [ex["prev_sgn"] for ex in batch]
        has_any_prev = any(p is not None for p in prev_sgn_list)

        if has_any_prev:
            max_ctx_T = max(x.shape[0] for x in prev_sgn_list if x is not None)
            padded_list, len_list = [], []
            for p in prev_sgn_list:
                if p is not None:
                    padded_list.append(p)
                    len_list.append(p.shape[0])
                else:
                    # Dummy single-frame (will be masked out)
                    padded_list.append(
                        np.zeros((1, self.feature_size), dtype=np.float32)
                    )
                    len_list.append(0)
            prev_sgn_padded = _pad_2d(padded_list, max_ctx_T, self.feature_size)
            prev_sgn_len = torch.tensor(len_list, dtype=torch.long)
        else:
            prev_sgn_padded = None
            prev_sgn_len = None

        return {
            "name": [ex["name"] for ex in batch],
            "sgn": sgn_padded,
            "sgn_len": sgn_len,
            "txt": txt_padded,
            "txt_len": txt_len,
            "prev_sgn": prev_sgn_padded,
            "prev_sgn_len": prev_sgn_len,
            "text_raw": [ex["text_raw"] for ex in batch],
        }


def load_data(data_cfg: dict):
    """
    Load vocabulary and create train/dev/test datasets.

    File pairs are discovered, shuffled with a seed, and split 90/5/5.
    """
    data_path = data_cfg["data_path"]
    vocab_file = data_cfg["vocab_file"]
    fps = data_cfg.get("fps", 12.5)
    max_sgn_len = data_cfg.get("max_sgn_len", 256)
    max_txt_len = data_cfg.get("max_txt_len", 128)
    split_seed = data_cfg.get("split_seed", 42)
    context_prob = data_cfg.get("context_prob", CONTEXT_PROB)
    context_gap_max = data_cfg.get("context_gap_max", CONTEXT_GAP_MAX)

    vocab = Vocabulary(file=vocab_file)
    bos_id = vocab.stoi[BOS_TOKEN]
    eos_id = vocab.stoi[EOS_TOKEN]

    # Find all file pairs
    pairs = _find_file_pairs(data_path)
    assert len(pairs) > 0, f"No file pairs found in {data_path}"

    # Shuffle and split 90/5/5
    rng = np.random.default_rng(split_seed)
    indices = rng.permutation(len(pairs))
    n = len(pairs)
    n_train = int(n * 0.9)
    n_dev = int(n * 0.05)

    train_pairs = [pairs[i] for i in indices[:n_train]]
    dev_pairs = [pairs[i] for i in indices[n_train : n_train + n_dev]]
    test_pairs = [pairs[i] for i in indices[n_train + n_dev :]]

    common = dict(
        bos_id=bos_id, eos_id=eos_id, fps=fps,
        max_sgn_len=max_sgn_len, max_txt_len=max_txt_len, seed=split_seed,
        context_gap_max=context_gap_max,
    )

    train_data = SignTranslationDataset(
        train_pairs, train=True,
        context_prob=context_prob, **common,
    )
    dev_data = SignTranslationDataset(
        dev_pairs, train=False,
        context_prob=0.0, **common,
    )
    test_data = SignTranslationDataset(
        test_pairs, train=False,
        context_prob=0.0, **common,
    )

    return train_data, dev_data, test_data, vocab


def make_data_iter(dataset, batch_size, pad_id, sgn_dim, train=False, shuffle=False):
    """Create a DataLoader for the dataset with proper collation."""
    collator = Collator(pad_id=pad_id, feature_size=sgn_dim)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and train,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        drop_last=False,
    )
