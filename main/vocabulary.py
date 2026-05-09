# coding: utf-8
"""
Vocabulary module - loads SPM vocabulary from file.
"""
import numpy as np
from typing import List

UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"


class Vocabulary:
    """Vocabulary from an SPM vocab file (token\\tscore per line)."""

    def __init__(self, file: str):
        self.itos = []
        self.stoi = {}
        with open(file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                token = line.rstrip("\n").split("\t")[0]
                self.itos.append(token)
                self.stoi[token] = i

        assert self.stoi[UNK_TOKEN] == 0
        assert self.stoi[PAD_TOKEN] == 1
        assert self.stoi[BOS_TOKEN] == 2
        assert self.stoi[EOS_TOKEN] == 3

    def __len__(self) -> int:
        return len(self.itos)

    def to_file(self, file: str):
        with open(file, "w", encoding="utf-8") as f:
            for t in self.itos:
                f.write(f"{t}\n")

    def array_to_sentence(self, array: np.ndarray, cut_at_eos=True) -> List[str]:
        sentence = []
        for i in array:
            s = self.itos[i]
            if cut_at_eos and s == EOS_TOKEN:
                break
            sentence.append(s)
        return sentence

    def arrays_to_sentences(self, arrays: np.ndarray, cut_at_eos=True) -> List[List[str]]:
        return [self.array_to_sentence(a, cut_at_eos) for a in arrays]

    def decode(self, array: np.ndarray, cut_at_eos=True) -> str:
        pieces = self.array_to_sentence(array, cut_at_eos)
        if pieces and pieces[0] == BOS_TOKEN:
            pieces = pieces[1:]
        return "".join(pieces).replace("\u2581", " ").strip()

    def decode_batch(self, arrays: np.ndarray, cut_at_eos=True) -> List[str]:
        return [self.decode(a, cut_at_eos) for a in arrays]
