# coding: utf-8
"""
Batch module — holds a single batch with masks during training/eval.
Accepts a dict-based batch from the Collator, including optional
previous-sentence visual context.
"""
import torch


class Batch:
    """Object for holding a batch of data with masks."""

    def __init__(self, torch_batch, txt_pad_index, sgn_dim, use_cuda=False):
        self.sgn = torch_batch["sgn"]          # (B, T, D)
        self.sgn_lengths = torch_batch["sgn_len"]  # (B,)
        self.sgn_dim = sgn_dim
        self.num_seqs = self.sgn.size(0)

        # Length-based mask: True where valid (not padding)
        B, T, _ = self.sgn.shape
        device = self.sgn.device
        self.sgn_mask = (
            torch.arange(T, device=device).unsqueeze(0) < self.sgn_lengths.unsqueeze(1)
        ).unsqueeze(1)  # (B, 1, T)

        # Text: [BOS, tokens..., EOS], padded with pad_id
        txt_full = torch_batch["txt"]
        self.txt_input = txt_full[:, :-1]     # decoder input: all but last
        self.txt = txt_full[:, 1:]            # target: all but first (strips BOS)
        self.txt_lengths = torch_batch["txt_len"]
        self.txt_mask = (self.txt_input != txt_pad_index).unsqueeze(1)
        self.num_txt_tokens = (self.txt != txt_pad_index).data.sum().item()

        # Previous sentence visual context (for encoder context pooling)
        self.prev_sgn = torch_batch.get("prev_sgn", None)      # (B, T_ctx, D) or None
        self.prev_sgn_len = torch_batch.get("prev_sgn_len", None)  # (B,) or None

        self.use_cuda = use_cuda
        if use_cuda:
            self._make_cuda()

    def _make_cuda(self):
        """Move tensors to GPU."""
        self.sgn = self.sgn.cuda(non_blocking=True)
        self.sgn_mask = self.sgn_mask.cuda(non_blocking=True)
        self.txt = self.txt.cuda(non_blocking=True)
        self.txt_mask = self.txt_mask.cuda(non_blocking=True)
        self.txt_input = self.txt_input.cuda(non_blocking=True)
        if self.prev_sgn is not None:
            self.prev_sgn = self.prev_sgn.cuda(non_blocking=True)
        if self.prev_sgn_len is not None:
            self.prev_sgn_len = self.prev_sgn_len.cuda(non_blocking=True)

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

        if self.prev_sgn is not None:
            self.prev_sgn = self.prev_sgn[perm_index]
        if self.prev_sgn_len is not None:
            self.prev_sgn_len = self.prev_sgn_len[perm_index]

        if self.use_cuda:
            self._make_cuda()

        return rev_index
