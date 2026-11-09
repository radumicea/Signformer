# coding: utf-8
"""
Dataset module
"""
import numpy as np
import torch
from rsl_news.dataset import Sentences


class SignTranslationDataset(Sentences):
    """
    Sentences of RSL-News segments: the feature windows between the sentence
    timestamps, and the sentence token ids.

    The features of a segment are stored as `[num_windows, feature_size]`,
    where window k covers the frames [k * stride, k * stride + size). A
    sentence gets the windows that lie entirely between its start and end:
    the windows of the video cropped to the sentence, up to the grid of the
    feature file (less than `stride` frames). Segments without features are
    skipped.
    """

    def __init__(
        self,
        data,
        split: str,
        features,
        tokenizer,
        bos_index: int,
        eos_index: int,
        max_sgn_len: int = None,
        max_txt_len: int = None,
        timing=None,
        seed: int = 0,
    ):
        """
        :param data: the dataset (rsl_news.RSLNews)
        :param split: "train", "val" or "test"
        :param features: the feature files (rsl_news.Windows)
        :param tokenizer: texts -> token ids
        :param bos_index: index of <s>, prepended to every sentence
        :param eos_index: index of </s>, appended to every sentence
        :param max_sgn_len: drop sentences with more windows than this
        :param max_txt_len: drop sentences with more tokens (with <s> and </s>) than this
        :param timing: the stretch of the segment a sample covers (rsl_news.Extend), None: the
            sentence's own timestamps
        :param seed: seed of random timings (drawn from seed, epoch and sentence; see set_epoch)
        """

        def keep(sentence, windows, tokens):
            return (
                windows > 0
                and len(tokens) > 0
                and (max_sgn_len is None or windows <= max_sgn_len)
                and (max_txt_len is None or len(tokens) + 2 <= max_txt_len)
            )

        super().__init__(data, split, features, tokenizer, timing=timing, keep=keep, seed=seed)
        self.bos_index = bos_index
        self.eos_index = eos_index
        self.sgn_lengths = self.lengths
        self.references = self.texts  # the text as released
        self.segment_names = [segment.name for segment in self.segments]
        self.feature_files = [str(data.path(features.files(s)[0])) for s in self.segments]

    def __getitem__(self, idx: int):
        sample = super().__getitem__(idx)
        sgn = sample["features"].astype(np.float32)
        txt = np.concatenate(([self.bos_index], sample["tokens"], [self.eos_index])).astype(np.int64)
        return idx, torch.from_numpy(sgn), torch.from_numpy(txt)
