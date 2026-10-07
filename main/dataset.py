# coding: utf-8
"""
Dataset module
"""
import json
import math
import os
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class SignTranslationDataset(Dataset):
    """
    Sentences of RSL-News segments: the feature windows between the sentence
    timestamps, and the sentence token ids.

    The features of a segment are stored as `[num_windows, feature_size]`,
    where window k covers the frames [k * stride, k * stride + size). A
    sentence gets the windows that lie entirely between its start and end:
    the windows of the video cropped to the sentence, up to the grid of the
    feature file (less than `stride` frames).
    """

    def __init__(
        self,
        segments: List[Tuple[str, str, str]],
        bos_index: int,
        eos_index: int,
        fps: float,
        window_size: int,
        window_stride: int,
        max_sgn_len: int = None,
        max_txt_len: int = None,
    ):
        """
        :param segments: (name, annotation .json path, feature .npy path) per
            segment; segments without both files are skipped
        :param bos_index: index of <s>, prepended to every sentence
        :param eos_index: index of </s>, appended to every sentence
        :param fps: video frame rate the timestamps refer to
        :param window_size: frames per feature window
        :param window_stride: frames between the starts of consecutive windows
        :param max_sgn_len: drop sentences with more windows than this
        :param max_txt_len: drop sentences with more tokens (with <s> and </s>) than this
        """
        self.bos_index = bos_index
        self.eos_index = eos_index

        self.segment_names = []
        self.feature_files = []
        segment, sentence, first, last, lengths, tokens = [], [], [], [], [], []
        self.references = []
        self.num_dropped = 0
        self.num_missing = 0
        for name, json_path, npy_path in segments:
            if not (os.path.isfile(json_path) and os.path.isfile(npy_path)):
                self.num_missing += 1
                continue
            num_windows = np.load(npy_path, mmap_mode="r").shape[0]
            with open(json_path, "r", encoding="utf-8") as f:
                entries = json.load(f)
            for i, entry in enumerate(entries):
                # the sentence's frames are [ceil(start * fps), ceil(end * fps)); keep the
                # windows [k * stride, k * stride + size) inside them (round: float noise)
                first_frame = math.ceil(round(entry["start"] * fps, 6))
                end_frame = math.ceil(round(entry["end"] * fps, 6))
                begin = math.ceil(first_frame / window_stride)
                end = (end_frame - window_size) // window_stride + 1
                begin, end = max(begin, 0), min(end, num_windows)
                entry_tokens = entry["tokens_lower"]
                if (
                    end <= begin
                    or not entry_tokens
                    or (max_sgn_len is not None and end - begin > max_sgn_len)
                    or (max_txt_len is not None and len(entry_tokens) + 2 > max_txt_len)
                ):
                    self.num_dropped += 1
                    continue
                segment.append(len(self.feature_files))
                sentence.append(i)
                first.append(begin)
                last.append(end)
                lengths.append(len(entry_tokens))
                tokens += entry_tokens
                self.references.append(entry["text_lower"])
            self.segment_names.append(name)
            self.feature_files.append(npy_path)

        # flat arrays instead of per-sentence objects: they are shared by the
        # data loader workers without being copied
        self.segment = np.array(segment, dtype=np.int32)
        self.sentence = np.array(sentence, dtype=np.int32)
        self.first = np.array(first, dtype=np.int64)
        self.sgn_lengths = np.array(last, dtype=np.int64) - self.first
        self.token_offsets = np.cumsum([0] + lengths, dtype=np.int64)
        self.tokens = np.array(tokens, dtype=np.int32)

    def __len__(self) -> int:
        return len(self.segment)

    def __getitem__(self, idx: int):
        features = np.load(self.feature_files[self.segment[idx]], mmap_mode="r")
        first = self.first[idx]
        sgn = features[first : first + self.sgn_lengths[idx]].astype(np.float32)

        tokens = self.tokens[self.token_offsets[idx] : self.token_offsets[idx + 1]]
        txt = np.concatenate(([self.bos_index], tokens, [self.eos_index])).astype(np.int64)

        return idx, torch.from_numpy(sgn), torch.from_numpy(txt)

    @property
    def names(self) -> List[str]:
        """Sentence names: <segment>:<index of the sentence in the segment>"""
        return [
            "{}:{}".format(self.segment_names[s], i)
            for s, i in zip(self.segment, self.sentence)
        ]
