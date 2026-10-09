# coding: utf-8
"""
Vocabulary module - the token strings of the SentencePiece model.
"""
import numpy as np
from typing import List

UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"


class Vocabulary:
    """Token strings by id, e.g. the pieces of a SentencePiece model."""

    def __init__(self, pieces: List[str]):
        self.itos = list(pieces)
        self.stoi = {token: i for i, token in enumerate(self.itos)}

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

    def decode(self, array: np.ndarray, cut_at_eos=True) -> str:
        pieces = self.array_to_sentence(array, cut_at_eos)
        if pieces and pieces[0] == BOS_TOKEN:
            pieces = pieces[1:]
        return "".join(pieces).replace("\u2581", " ").strip()

    def decode_batch(self, arrays: np.ndarray, cut_at_eos=True) -> List[str]:
        return [self.decode(a, cut_at_eos) for a in arrays]
