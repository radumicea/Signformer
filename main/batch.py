# coding: utf-8
import torch
import numpy as np


class Batch:
    """Object for holding a batch of data with masks during training."""

    def __init__(self, sgn, sgn_lengths, txt, txt_lengths,
                 txt_pad_index, sgn_dim, use_cuda=False):
        self.sgn = sgn
        self.sgn_lengths = sgn_lengths
        self.sgn_dim = sgn_dim
        self.num_seqs = sgn.size(0)

        # Length-based mask: True where valid (not padding)
        self.sgn_mask = (
            torch.arange(sgn.size(1)).unsqueeze(0) < sgn_lengths.unsqueeze(1)
        ).unsqueeze(1)  # (B, 1, T)

        # txt includes [BOS, tokens..., EOS], padded with pad_id
        # txt_input for teacher forcing: all but last token
        self.txt_input = txt[:, :-1]
        self.txt_lengths = txt_lengths
        # txt target: all but first (shifted by one, strips BOS)
        self.txt = txt[:, 1:]
        self.txt_mask = (self.txt_input != txt_pad_index).unsqueeze(1)
        self.num_txt_tokens = (self.txt != txt_pad_index).data.sum().item()

        self.use_cuda = use_cuda
        if use_cuda:
            self._make_cuda()

    def _make_cuda(self):
        self.sgn = self.sgn.cuda()
        self.sgn_mask = self.sgn_mask.cuda()
        self.txt = self.txt.cuda()
        self.txt_mask = self.txt_mask.cuda()
        self.txt_input = self.txt_input.cuda()

    def sort_by_sgn_lengths(self):
        """Sort by sgn length (descending) and return index to revert sort."""
        _, perm_index = self.sgn_lengths.sort(0, descending=True)
        rev_index = [0] * perm_index.size(0)
        for new_pos, old_pos in enumerate(perm_index.cpu().numpy()):
            rev_index[old_pos] = new_pos

        self.sgn = self.sgn[perm_index]
        self.sgn_mask = self.sgn_mask[perm_index]
        self.sgn_lengths = self.sgn_lengths[perm_index]

        self.txt = self.txt[perm_index]
        self.txt_mask = self.txt_mask[perm_index]
        self.txt_input = self.txt_input[perm_index]
        self.txt_lengths = self.txt_lengths[perm_index]

        if self.use_cuda:
            self._make_cuda()

        return rev_index
