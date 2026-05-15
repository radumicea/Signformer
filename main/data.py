# coding: utf-8
"""
Data module - loads and splits dataset, creates data loaders.
"""
import glob
import os
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader

from main.dataset import SignTranslationDataset
from main.vocabulary import Vocabulary, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN


def _find_file_triplets(data_path):
    """Find all matching .npy / .phonemes.npy / .json file triplets recursively."""
    json_files = sorted(
        glob.glob(os.path.join(data_path, "**", "*.json"), recursive=True)
    )
    triplets = []
    for json_path in json_files:
        npy_path = json_path.replace(".json", ".npy")
        phoneme_path = json_path.replace(".json", ".phonemes.npz")
        if os.path.exists(npy_path) and os.path.exists(phoneme_path):
            triplets.append((npy_path, phoneme_path, json_path))
    return triplets


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

    vocab = Vocabulary(file=vocab_file)
    bos_id = vocab.stoi[BOS_TOKEN]
    eos_id = vocab.stoi[EOS_TOKEN]

    # Find all file triplets
    triplets = _find_file_triplets(data_path)
    assert len(triplets) > 0, f"No file triplets found in {data_path}"

    # Shuffle and split 90/5/5
    rng = np.random.default_rng(split_seed)
    indices = rng.permutation(len(triplets))
    n = len(triplets)
    n_train = int(n * 0.9)
    n_dev = int(n * 0.05)

    train_triplets = [triplets[i] for i in indices[:n_train]]
    dev_triplets = [triplets[i] for i in indices[n_train : n_train + n_dev]]
    test_triplets = [triplets[i] for i in indices[n_train + n_dev :]]

    phoneme_dim = data_cfg["phoneme_dim"]

    train_data = SignTranslationDataset(
        train_triplets, bos_id, eos_id, fps, max_sgn_len, max_txt_len, train=True,
    )
    dev_data = SignTranslationDataset(
        dev_triplets, bos_id, eos_id, fps, max_sgn_len, max_txt_len, train=False,
    )
    test_data = SignTranslationDataset(
        test_triplets, bos_id, eos_id, fps, max_sgn_len, max_txt_len, train=False,
    )

    return train_data, dev_data, test_data, vocab


def _collate_fn(samples, pad_id, sgn_dim, phoneme_dim=0):
    """Collate function for DataLoader - pads sgn, txt, and phonemes to batch max."""
    sgn_list = [s[0] for s in samples]
    txt_list = [s[1] for s in samples]
    phn_list = [s[2] for s in samples]

    sgn_lengths = torch.tensor([s.shape[0] for s in sgn_list])
    txt_lengths = torch.tensor([t.shape[0] for t in txt_list])

    max_sgn = sgn_lengths.max().item()
    max_txt = txt_lengths.max().item()
    batch_size = len(samples)

    sgn = torch.zeros(batch_size, max_sgn, sgn_dim)
    txt = torch.full((batch_size, max_txt), pad_id, dtype=torch.long)

    for i, (s, t) in enumerate(zip(sgn_list, txt_list)):
        sgn[i, : s.shape[0]] = s
        txt[i, : t.shape[0]] = t

    phonemes = torch.zeros(batch_size, max_sgn, phoneme_dim)
    for i, p in enumerate(phn_list):
        phonemes[i, : p.shape[0]] = p

    return sgn, sgn_lengths, txt, txt_lengths, phonemes


def make_data_iter(dataset, batch_size, pad_id, sgn_dim, phoneme_dim, train=False, shuffle=False):
    """Create a DataLoader for the dataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and train,
        collate_fn=partial(_collate_fn, pad_id=pad_id, sgn_dim=sgn_dim, phoneme_dim=phoneme_dim),
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        drop_last=False,
    )
