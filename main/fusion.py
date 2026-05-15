# coding: utf-8
"""
Gated fusion module for sign features and phoneme CTC logits.

Inspired by SignBind-LLM (§3.4): projects both modalities into a shared
dimension D, then applies a learned sigmoid gate that decides per-frame
and per-dimension how much to rely on each stream.

    FG       = σ(sgn_proj · W_g + b_g)
    H_fused  = FG ⊙ sgn_proj + (1 − FG) ⊙ phn_proj
"""

import torch
import torch.nn as nn
from torch import Tensor


class PhonemeSignFusion(nn.Module):
    """
    Fuses sign-language frame embeddings (B, T, D) with per-frame
    phoneme CTC logits (B, T, num_phonemes) via a sigmoid gate.
    """

    def __init__(self, embedding_dim: int, phoneme_dim: int):
        """
        :param embedding_dim: dimension D of the projected sign embeddings
        :param phoneme_dim:   number of phoneme classes (CTC logit width)
        """
        super().__init__()
        self.phoneme_proj = nn.Linear(phoneme_dim, embedding_dim)
        self.phoneme_norm = nn.LayerNorm(embedding_dim)
        self.gate = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, sgn: Tensor, phonemes: Tensor) -> Tensor:
        """
        :param sgn:      (B, T, D)  — sign embeddings (already projected)
        :param phonemes: (B, T, num_phonemes) — frame-level CTC logits
        :return:         (B, T, D)  — fused representation
        """
        p = self.phoneme_proj(phonemes)
        p = self.phoneme_norm(p)
        fg = torch.sigmoid(self.gate(sgn))
        return fg * sgn + (1 - fg) * p
