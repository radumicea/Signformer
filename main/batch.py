# coding: utf-8
import torch
from torch import Tensor


class Batch:
    """Object for holding a batch of data with masks during training."""

    def __init__(
        self,
        sgn: Tensor,
        sgn_lengths: Tensor,
        txt: Tensor,
        txt_pad_index: int,
        use_cuda: bool = False,
    ):
        """
        :param sgn: feature windows, zero padded (batch, max_sgn_len, feature_size)
        :param sgn_lengths: number of windows of each sentence (batch,)
        :param txt: [<s>, tokens..., </s>], padded with txt_pad_index (batch, max_txt_len)
        :param txt_pad_index: txt padding token index
        :param use_cuda: move the batch to the GPU
        """
        self.num_seqs = sgn.size(0)

        # Sign: mask is True where valid (not padding)
        self.sgn = sgn
        self.sgn_lengths = sgn_lengths
        self.sgn_mask = (
            torch.arange(sgn.size(1)).unsqueeze(0) < sgn_lengths.unsqueeze(1)
        ).unsqueeze(1)

        # txt_input is used for teacher forcing, last one is cut off
        self.txt_input = txt[:, :-1]
        # txt is used for loss computation, shifted by one since BOS
        self.txt = txt[:, 1:]
        # we exclude the padded areas from the loss computation
        self.txt_mask = (self.txt_input != txt_pad_index).unsqueeze(1)
        self.num_txt_tokens = (self.txt != txt_pad_index).sum().item()

        if use_cuda:
            self._make_cuda()

    def _make_cuda(self):
        """Move the batch to GPU"""
        self.sgn = self.sgn.cuda(non_blocking=True)
        self.sgn_mask = self.sgn_mask.cuda(non_blocking=True)
        self.txt = self.txt.cuda(non_blocking=True)
        self.txt_mask = self.txt_mask.cuda(non_blocking=True)
        self.txt_input = self.txt_input.cuda(non_blocking=True)
