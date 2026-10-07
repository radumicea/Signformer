# coding: utf-8
"""
Data module - finds the segments of each split, creates data loaders.

RSL-News layout (local, or the same files in a Hugging Face dataset repo):

    <data_path>/
        manifests/<channel>_manifest.json
            {"items": [{..., "split": "train" | "val" | "test", "segments": [{"index": i, ...}]}]}
        dataset/<Channel>/<episode>/segment_<i>.json
            [{"start": s, "end": s, "text_lower": ..., "tokens_lower": [...]}, ...]
        dataset/<Channel>/<episode>/segment_<i>.<features>.npy
            [num_windows, feature_size]
"""
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Callable, Dict, List

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Sampler

from main.dataset import SignTranslationDataset
from main.vocabulary import Vocabulary, BOS_TOKEN, EOS_TOKEN

# manifest file -> channel directory in dataset/
MANIFEST_CHANNELS = {
    "digi24_manifest.json": "Digi24",
    "prima_manifest.json": "PrimaTV",
    "protv_manifest.json": "ProTV",
}


def _episode_dir(item: dict) -> str:
    """Same layout as the download scripts: <id> (PrimaTV) or <YYYY>/<MM>/<DD>."""
    return item["id"] if "id" in item else item["date"].replace("-", "/")


def _split_segments(data_path: str, split: str) -> List[str]:
    """Segments (relative to dataset/, without extension) of the episodes in `split`."""
    segments = []
    for manifest, channel in MANIFEST_CHANNELS.items():
        with open(os.path.join(data_path, "manifests", manifest), encoding="utf-8") as f:
            items = json.load(f)["items"]
        for item in items:
            if item.get("split") == split:
                episode = "{}/{}".format(channel, _episode_dir(item))
                segments += [
                    "{}/segment_{}".format(episode, s["index"]) for s in item["segments"]
                ]
    return segments


def _download(repo_id: str, data_path: str, keep: Callable[[str], bool]) -> None:
    """
    Download the files of Hugging Face dataset `repo_id` for which `keep(path)`
    holds into `data_path`. Files already downloaded are not fetched again.
    """
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    revision = api.dataset_info(repo_id).sha
    files = [
        f
        for f in api.list_repo_files(repo_id, repo_type="dataset", revision=revision)
        if keep(f)
    ]
    fetch = partial(
        hf_hub_download,
        repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=data_path,
    )
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(fetch, files))


def load_data(
    data_cfg: dict, splits: List[str]
) -> (Dict[str, SignTranslationDataset], Vocabulary):
    """
    Load the vocabulary and the datasets of the given splits ("train", "val",
    "test"). With `hf_repo` set, only the files these splits need are
    downloaded (once) into `data_path`.

    Training sentences longer than `max_sgn_len` windows or `max_txt_len`
    tokens are dropped, like in the original code.

    :param data_cfg: configuration dictionary for data
        ("data" part of configuration file)
    :param splits: splits to load
    :return: datasets by split, text vocabulary
    """
    data_path = data_cfg.get("data_path", "../RSL-News")
    hf_repo = data_cfg.get("hf_repo")
    features = data_cfg.get("features", "bsl5k")
    vocab_file = data_cfg["vocab_file"]
    if not os.path.isabs(vocab_file):
        vocab_file = os.path.join(data_path, vocab_file)

    if hf_repo:
        vocab_in_repo = os.path.relpath(vocab_file, data_path)
        _download(
            hf_repo, data_path, lambda f: f.startswith("manifests/") or f == vocab_in_repo
        )

    segments = {split: _split_segments(data_path, split) for split in splits}

    def files(segment: str) -> (str, str):
        path = os.path.join(data_path, "dataset", segment)
        return path + ".json", "{}.{}.npy".format(path, features)

    if hf_repo:
        wanted = {
            os.path.relpath(f, data_path)
            for split in splits
            for segment in segments[split]
            for f in files(segment)
        }
        _download(hf_repo, data_path, wanted.__contains__)

    txt_vocab = Vocabulary(file=vocab_file)

    datasets = {}
    for split in splits:
        datasets[split] = SignTranslationDataset(
            segments=[(segment, *files(segment)) for segment in segments[split]],
            bos_index=txt_vocab.stoi[BOS_TOKEN],
            eos_index=txt_vocab.stoi[EOS_TOKEN],
            fps=data_cfg.get("fps", 25),
            window_size=data_cfg.get("window_size", 8),
            window_stride=data_cfg.get("window_stride", 2),
            max_sgn_len=data_cfg.get("max_sgn_len") if split == "train" else None,
            max_txt_len=data_cfg.get("max_txt_len") if split == "train" else None,
        )

    return datasets, txt_vocab


class BucketBatchSampler(Sampler):
    """
    Batches of sentences with similar numbers of windows, like the torchtext
    BucketIterator of the original code. When shuffling, the data is shuffled
    and cut into pools of `pool_size` batches; each pool is sorted by length
    and split into batches, and the batches are shuffled. Without shuffling,
    all data is sorted by length.

    The order depends only on (seed, epoch), so a resumed run sees the same
    batches as an uninterrupted one.
    """

    def __init__(
        self,
        lengths: np.ndarray,
        batch_size: int,
        shuffle: bool = False,
        seed: int = 0,
        pool_size: int = 100,
    ):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.pool_size = pool_size
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        indices = np.arange(len(self.lengths))
        if self.shuffle:
            rng = np.random.default_rng([self.seed, self.epoch])
            indices = rng.permutation(indices)
            pool = self.batch_size * self.pool_size
            pools = [indices[i : i + pool] for i in range(0, len(indices), pool)]
        else:
            pools = [indices]

        batches = []
        for p in pools:
            # longest first: when evaluating, the largest batch comes first, so
            # running out of memory shows up at once
            p = p[np.argsort(-self.lengths[p], kind="stable")]
            batches += [p[i : i + self.batch_size] for i in range(0, len(p), self.batch_size)]
        if self.shuffle:
            batches = [batches[i] for i in rng.permutation(len(batches))]

        for batch in batches:
            yield batch.tolist()

    def __len__(self) -> int:
        n = len(self.lengths)
        if not self.shuffle:
            return math.ceil(n / self.batch_size)
        pool = self.batch_size * self.pool_size
        return (n // pool) * self.pool_size + math.ceil((n % pool) / self.batch_size)


def collate_fn(samples, pad_index: int):
    """Pad windows with zeros and tokens with <pad> to the longest in the batch."""
    ids, sgn, txt = zip(*samples)
    return (
        torch.tensor(ids),
        pad_sequence(sgn, batch_first=True),
        torch.tensor([s.shape[0] for s in sgn]),
        pad_sequence(txt, batch_first=True, padding_value=pad_index),
    )


def make_data_iter(
    dataset: SignTranslationDataset,
    batch_size: int,
    pad_index: int,
    train: bool = False,
    shuffle: bool = False,
    seed: int = 0,
    num_workers: int = 0,
    use_cuda: bool = False,
) -> DataLoader:
    """
    Returns a data loader yielding (ids, sgn, sgn_lengths, txt) batches.

    :param dataset: dataset
    :param batch_size: sentences per batch
    :param pad_index: txt padding token index
    :param train: whether it's training time, when turned off,
        shuffling is disabled
    :param shuffle: whether to shuffle the data before each epoch; call
        `loader.batch_sampler.set_epoch(epoch)` before iterating
    :param seed: seed of the shuffling
    :param num_workers: data loading processes (reading a slice of a
        memory-mapped file is cheap, a few are enough)
    :param use_cuda: pin memory for faster transfers to the GPU
    :return: data loader
    """
    return DataLoader(
        dataset,
        batch_sampler=BucketBatchSampler(
            dataset.sgn_lengths, batch_size, shuffle=train and shuffle, seed=seed
        ),
        collate_fn=partial(collate_fn, pad_index=pad_index),
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=num_workers > 0,
        # own generator: creating the loader does not consume the global RNG
        generator=torch.Generator().manual_seed(seed),
    )
