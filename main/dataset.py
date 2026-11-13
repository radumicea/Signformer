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
    sentence gets the windows that lie entirely within its time window: by
    default its start and end, or a window around them (`timing`). Segments
    without features are skipped.
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
        norm: dict = None,
    ):
        """
        :param data: the dataset (rsl_news.RSLNews)
        :param split: "train", "val" or "test"
        :param features: the feature files (rsl_news.Windows)
        :param tokenizer: texts -> token ids
        :param bos_index: index of <s>, prepended to every sentence
        :param eos_index: index of </s>, appended to every sentence
        :param max_sgn_len: drop sentences with more windows than this (between their own
            timestamps, as every filter here: the same sentences are kept whatever the timing)
        :param max_txt_len: drop sentences with more tokens (with <s> and </s>) than this
        :param timing: the stretch of the segment a sample covers (rsl_news.Extend), None: the
            sentence's own timestamps
        :param seed: seed of random timings (drawn from seed, epoch and sentence; see set_epoch)
        :param norm: per-broadcast feature statistics ("keys": <Channel>/<broadcast>, "mean", "std",
            "floor"): a sample's features become (x - mean) / max(std, floor) of its broadcast, over all
            of the broadcast's video (labelled or not); None: the features as stored
        """

        lengths = {}

        def own_windows(sentence):
            segment = sentence.segment
            if segment not in lengths:
                lengths[segment] = features.length(data, segment)
            first, stop = features.steps(data, segment, sentence.start, sentence.end, lengths[segment])
            return stop - first

        def keep(sentence, windows, tokens):
            windows = own_windows(sentence)
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
        # windows drawn at random: they change with set_epoch, which must reach the loader's workers
        self.random_timing = timing is not None and any(
            isinstance(v, (tuple, list)) for v in (timing.before, timing.after)
        )
        self.feature_files = [str(data.path(features.files(s)[0])) for s in self.segments]
        self.norm = None
        if norm is not None:
            row = {k: i for i, k in enumerate(norm["keys"])}
            self.norm_row = np.array([row[f"{s.channel}/{s.broadcast}"] for s in self.segments])
            self.norm = (norm["mean"], np.maximum(norm["std"], norm["floor"]))

    def __getitem__(self, idx: int):
        sample = super().__getitem__(idx)
        sgn = sample["features"].astype(np.float32)
        if self.norm is not None:
            r = self.norm_row[self.segment[idx]]
            sgn = (sgn - self.norm[0][r]) / self.norm[1][r]
        txt = np.concatenate(([self.bos_index], sample["tokens"], [self.eos_index])).astype(np.int64)
        return idx, torch.from_numpy(sgn), torch.from_numpy(txt)
