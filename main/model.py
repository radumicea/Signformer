# coding: utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from main.initialization import initialize_model
from main.embeddings import Embeddings, SpatialEmbeddings
from main.encoders import Encoder, RecurrentEncoder, TransformerEncoder
from main.decoders import Decoder, RecurrentDecoder, TransformerDecoder
from main.search import beam_search, greedy
from main.vocabulary import (
    Vocabulary,
    PAD_TOKEN,
    EOS_TOKEN,
    BOS_TOKEN,
)
from main.batch import Batch
from main.helpers import freeze_params
from torch import Tensor
from typing import Union


class ContextPooling(nn.Module):
    """Compress variable-length visual context into K fixed query vectors
    via cross-attention (Perceiver / Q-Former style).
    """

    def __init__(self, hidden_size, num_queries=8, num_heads=8, dropout=0.1):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, num_queries, hidden_size) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, ctx_features, ctx_mask):
        """Pool context features into fixed-size summary.

        Args:
            ctx_features: (B, T_ctx, D) embedded context sign features
            ctx_mask: (B, 1, T_ctx) bool mask (True = valid)
        Returns:
            (B, K, D) compressed context summary
        """
        B = ctx_features.size(0)
        queries = self.queries.expand(B, -1, -1)
        # MHA expects key_padding_mask: (B, T_ctx), True = IGNORE
        key_padding_mask = ~ctx_mask.squeeze(1)
        out, _ = self.cross_attn(
            queries,
            ctx_features,
            ctx_features,
            key_padding_mask=key_padding_mask,
        )
        return self.norm(out)


class SignModel(nn.Module):
    """Sign Language Translation Model with optional prefix and
    previous-sentence visual context injection into the encoder input.

    Encoder input layout (when context is provided):
        [prefix_emb] [sgn_emb] [context_sep] [ctx_pooled(K)]

    When no context:
        [prefix_emb] [sgn_emb]
    """

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        sgn_embed: SpatialEmbeddings,
        txt_embed: Embeddings,
        txt_vocab: Vocabulary,
        prefix_tokens: list = None,
        context_num_queries: int = 8,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.sgn_embed = sgn_embed
        self.txt_embed = txt_embed

        self.txt_vocab = txt_vocab

        self.txt_bos_index = self.txt_vocab.stoi[BOS_TOKEN]
        self.txt_pad_index = self.txt_vocab.stoi[PAD_TOKEN]
        self.txt_eos_index = self.txt_vocab.stoi[EOS_TOKEN]

        hidden_size = encoder._output_size

        # Learnable separator between sign features and context summary
        self.context_sep = nn.Parameter(torch.randn(1, hidden_size) * 0.02)

        # Context pooling: compress previous sentence visual features
        self.context_pool = ContextPooling(
            hidden_size=hidden_size,
            num_queries=context_num_queries,
        )

        # Fixed prefix token IDs (registered as buffer so they move with .cuda())
        if prefix_tokens:
            self.register_buffer(
                "_prefix_ids",
                torch.tensor(prefix_tokens, dtype=torch.long),
            )
        else:
            self._prefix_ids = None

    def _embed_prefix(self, batch_size: int, device) -> Tensor:
        """Embed the fixed prefix tokens → (B, L_prefix, D)."""
        if self._prefix_ids is None:
            return None
        ids = self._prefix_ids.unsqueeze(0).expand(batch_size, -1)
        mask = torch.ones(batch_size, 1, ids.size(1), dtype=torch.bool, device=device)
        return self.txt_embed(ids, mask=mask)

    def _pool_prev_context(self, prev_sgn, prev_sgn_len):
        """Embed and compress previous sentence visual features → (B, K, D).

        Uses sgn_embed (shared with main features) then ContextPooling.
        Samples with prev_sgn_len==0 get zeroed output.
        """
        B, T_ctx = prev_sgn.shape[:2]
        device = prev_sgn.device

        # Clamp to min 1 so cross-attention doesn't get all-masked input
        safe_len = torch.clamp(prev_sgn_len, min=1)
        arange = torch.arange(T_ctx, device=device).unsqueeze(0)
        ctx_mask = (arange < safe_len.unsqueeze(1)).unsqueeze(1)  # (B, 1, T_ctx)

        # Embed through same projection as main sign features
        ctx_emb = self.sgn_embed(x=prev_sgn, mask=ctx_mask)  # (B, T_ctx, D)

        # Compress to fixed K vectors
        ctx_pooled = self.context_pool(ctx_emb, ctx_mask)  # (B, K, D)

        # Zero out for samples with no actual context
        has_ctx = (prev_sgn_len > 0).float().view(B, 1, 1)
        return ctx_pooled * has_ctx

    def _compose_encoder_input(
        self, sgn, sgn_mask, sgn_lengths, prev_sgn=None, prev_sgn_len=None
    ):
        """Build the composite encoder input and mask.

        Returns:
            enc_input: (B, L_total, D)
            enc_mask:  (B, 1, L_total)
            enc_lengths: (B,)
        """
        B = sgn.size(0)
        device = sgn.device

        # 1. Sign embedding
        sgn_emb = self.sgn_embed(x=sgn, mask=sgn_mask)  # (B, T, D)

        parts = []
        mask_parts = []

        # 2. Prefix (if configured)
        prefix_emb = self._embed_prefix(B, device)
        if prefix_emb is not None:
            L_p = prefix_emb.size(1)
            parts.append(prefix_emb)
            mask_parts.append(torch.ones(B, 1, L_p, dtype=torch.bool, device=device))

        # 3. Sign features
        parts.append(sgn_emb)
        mask_parts.append(sgn_mask)  # (B, 1, T)

        # 4. Separator + compressed visual context (if any in this batch)
        if prev_sgn is not None and prev_sgn_len is not None:
            has_ctx = prev_sgn_len > 0  # (B,) bool

            if has_ctx.any():
                # Separator: expand to (B, 1, D)
                sep = self.context_sep.expand(B, -1, -1)
                sep_mask = has_ctx.view(B, 1, 1)
                parts.append(sep)
                mask_parts.append(sep_mask)

                # Pool context visual features → (B, K, D)
                ctx_pooled = self._pool_prev_context(prev_sgn, prev_sgn_len)
                K = ctx_pooled.size(1)
                ctx_pool_mask = has_ctx.unsqueeze(1).unsqueeze(2).expand(B, 1, K)
                parts.append(ctx_pooled)
                mask_parts.append(ctx_pool_mask)

        enc_input = torch.cat(parts, dim=1)   # (B, L_total, D)
        enc_mask = torch.cat(mask_parts, dim=2)  # (B, 1, L_total)
        enc_lengths = enc_mask.squeeze(1).sum(dim=1).long()  # (B,)

        return enc_input, enc_mask, enc_lengths

    def forward(
        self,
        sgn: Tensor,
        sgn_mask: Tensor,
        sgn_lengths: Tensor,
        txt_input: Tensor,
        txt_mask: Tensor = None,
        prev_sgn: Tensor = None,
        prev_sgn_len: Tensor = None,
    ):
        encoder_output, encoder_hidden, enc_mask = self.encode(
            sgn=sgn,
            sgn_mask=sgn_mask,
            sgn_length=sgn_lengths,
            prev_sgn=prev_sgn,
            prev_sgn_len=prev_sgn_len,
        )
        unroll_steps = txt_input.size(1)
        decoder_outputs = self.decode(
            encoder_output=encoder_output,
            encoder_hidden=encoder_hidden,
            sgn_mask=enc_mask,
            txt_input=txt_input,
            unroll_steps=unroll_steps,
            txt_mask=txt_mask,
        )
        return decoder_outputs

    def encode(
        self,
        sgn: Tensor,
        sgn_mask: Tensor,
        sgn_length: Tensor,
        prev_sgn: Tensor = None,
        prev_sgn_len: Tensor = None,
    ):
        """
        Encodes the source sentence with optional context.

        :return: (encoder_output, encoder_hidden, enc_mask)
        """
        enc_input, enc_mask, enc_lengths = self._compose_encoder_input(
            sgn, sgn_mask, sgn_length, prev_sgn, prev_sgn_len,
        )
        output, hidden = self.encoder(
            embed_src=enc_input,
            src_length=enc_lengths,
            mask=enc_mask,
        )
        return output, hidden, enc_mask

    def decode(
        self,
        encoder_output: Tensor,
        encoder_hidden: Tensor,
        sgn_mask: Tensor,
        txt_input: Tensor,
        unroll_steps: int,
        decoder_hidden: Tensor = None,
        txt_mask: Tensor = None,
    ) -> (Tensor, Tensor, Tensor, Tensor):
        """
        Decode, given an encoded source sentence.

        :param encoder_output: encoder states for attention computation
        :param encoder_hidden: last encoder state for decoder initialization
        :param sgn_mask: sign sequence mask, 1 at valid tokens
        :param txt_input: spoken language sentence inputs
        :param unroll_steps: number of steps to unroll the decoder for
        :param decoder_hidden: decoder hidden state (optional)
        :param txt_mask: mask for spoken language words
        :return: decoder outputs (outputs, hidden, att_probs, att_vectors)
        """
        return self.decoder(
            encoder_output=encoder_output,
            encoder_hidden=encoder_hidden,
            src_mask=sgn_mask,
            trg_embed=self.txt_embed(x=txt_input, mask=txt_mask),
            trg_mask=txt_mask,
            unroll_steps=unroll_steps,
            hidden=decoder_hidden,
        )

    def get_loss_for_batch(
        self,
        batch: Batch,
        translation_loss_function: nn.Module,
        translation_loss_weight: float,
    ) -> Tensor:
        decoder_outputs = self.forward(
            sgn=batch.sgn,
            sgn_mask=batch.sgn_mask,
            sgn_lengths=batch.sgn_lengths,
            txt_input=batch.txt_input,
            txt_mask=batch.txt_mask,
            prev_sgn=batch.prev_sgn,
            prev_sgn_len=batch.prev_sgn_len,
        )
        word_outputs, _, _, _ = decoder_outputs
        txt_log_probs = F.log_softmax(word_outputs, dim=-1)
        translation_loss = (
            translation_loss_function(txt_log_probs, batch.txt)
            * translation_loss_weight
        )
        return translation_loss

    def run_batch(
        self,
        batch: Batch,
        translation_beam_size: int = 1,
        translation_beam_alpha: float = -1,
        translation_max_output_length: int = 100,
    ) -> (np.array, np.array):
        encoder_output, encoder_hidden, enc_mask = self.encode(
            sgn=batch.sgn,
            sgn_mask=batch.sgn_mask,
            sgn_length=batch.sgn_lengths,
            prev_sgn=batch.prev_sgn,
            prev_sgn_len=batch.prev_sgn_len,
        )

        if translation_beam_size < 2:
            stacked_txt_output, stacked_attention_scores = greedy(
                encoder_hidden=encoder_hidden,
                encoder_output=encoder_output,
                src_mask=enc_mask,
                embed=self.txt_embed,
                bos_index=self.txt_bos_index,
                eos_index=self.txt_eos_index,
                decoder=self.decoder,
                max_output_length=translation_max_output_length,
            )
        else:
            stacked_txt_output, stacked_attention_scores = beam_search(
                size=translation_beam_size,
                encoder_hidden=encoder_hidden,
                encoder_output=encoder_output,
                src_mask=enc_mask,
                embed=self.txt_embed,
                max_output_length=translation_max_output_length,
                alpha=translation_beam_alpha,
                eos_index=self.txt_eos_index,
                pad_index=self.txt_pad_index,
                bos_index=self.txt_bos_index,
                decoder=self.decoder,
            )

        return stacked_txt_output, stacked_attention_scores

    def __repr__(self) -> str:
        return (
            "%s(\n"
            "\tencoder=%s,\n"
            "\tdecoder=%s,\n"
            "\tsgn_embed=%s,\n"
            "\ttxt_embed=%s)"
            % (
                self.__class__.__name__,
                self.encoder,
                self.decoder,
                self.sgn_embed,
                self.txt_embed,
            )
        )


def build_model(
    cfg: dict,
    sgn_dim: int,
    txt_vocab: Vocabulary,
    multimodal: bool = False,
) -> SignModel:
    txt_padding_idx = txt_vocab.stoi[PAD_TOKEN]

    sgn_embed: SpatialEmbeddings = SpatialEmbeddings(
        **cfg["encoder"]["embeddings"],
        num_heads=cfg["encoder"]["num_heads"],
        input_size=sgn_dim,
        multimodal=multimodal
    )

    # build encoder
    enc_dropout = cfg["encoder"].get("dropout", 0.0)
    enc_emb_dropout = cfg["encoder"]["embeddings"].get("dropout", enc_dropout)
    cope = cfg.get("cope")
    if cfg["encoder"].get("type", "recurrent") == "transformer":
        assert (
            cfg["encoder"]["embeddings"]["embedding_dim"]
            == cfg["encoder"]["hidden_size"]
        ), "for transformer, emb_size must be hidden_size"

        encoder = TransformerEncoder(
            **cfg["encoder"],
            emb_size=sgn_embed.embedding_dim,
            emb_dropout=enc_emb_dropout,
            cope=cope
        )
    else:
        encoder = RecurrentEncoder(
            **cfg["encoder"],
            emb_size=sgn_embed.embedding_dim,
            emb_dropout=enc_emb_dropout,
        )

    txt_embed: Embeddings = Embeddings(
            **cfg["decoder"]["embeddings"],
            num_heads=cfg["decoder"]["num_heads"],
            vocab_size=len(txt_vocab),
            padding_idx=txt_padding_idx,
        )
    dec_dropout = cfg["decoder"].get("dropout", 0.0)
    dec_emb_dropout = cfg["decoder"]["embeddings"].get("dropout", dec_dropout)
    if cfg["decoder"].get("type", "recurrent") == "transformer":
        decoder = TransformerDecoder(
            **cfg["decoder"],
            encoder=encoder,
            vocab_size=len(txt_vocab),
            emb_size=txt_embed.embedding_dim,
            emb_dropout=dec_emb_dropout,
            cope=cope,
        )
    else:
        decoder = RecurrentDecoder(
            **cfg["decoder"],
            encoder=encoder,
            vocab_size=len(txt_vocab),
            emb_size=txt_embed.embedding_dim,
            emb_dropout=dec_emb_dropout,
        )

    model: SignModel = SignModel(
        encoder=encoder,
        decoder=decoder,
        sgn_embed=sgn_embed,
        txt_embed=txt_embed,
        txt_vocab=txt_vocab,
        prefix_tokens=cfg.get("prefix_tokens", None),
        context_num_queries=cfg.get("context_num_queries", 8),
    )
    if cfg.get("tied_softmax", False):
        if txt_embed.lut.weight.shape == model.decoder.output_layer.weight.shape:
            model.decoder.output_layer.weight = txt_embed.lut.weight
        else:
            raise ValueError(
                "For tied_softmax, the decoder embedding_dim and decoder "
                "hidden_size must be the same. "
                "The decoder must be a Transformer."
            )

    # custom initialization of model parameters
    initialize_model(model, cfg, txt_padding_idx)

    return model
