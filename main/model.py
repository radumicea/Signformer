# coding: utf-8
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from main.initialization import initialize_model
from main.embeddings import Embeddings, SpatialEmbeddings
from main.encoders import Encoder, RecurrentEncoder, TransformerEncoder
from main.decoders import Decoder, RecurrentDecoder, TransformerDecoder
from main.fusion import PhonemeSignFusion
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


class SignModel(nn.Module):
    """Sign Language Translation Model"""

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        fusion: PhonemeSignFusion,
        sgn_embed: SpatialEmbeddings,
        txt_embed: Embeddings,
        txt_vocab: Vocabulary,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.fusion = fusion

        self.sgn_embed = sgn_embed
        self.txt_embed = txt_embed

        self.txt_vocab = txt_vocab

        self.txt_bos_index = self.txt_vocab.stoi[BOS_TOKEN]
        self.txt_pad_index = self.txt_vocab.stoi[PAD_TOKEN]
        self.txt_eos_index = self.txt_vocab.stoi[EOS_TOKEN]

    def forward(
        self,
        sgn: Tensor,
        sgn_mask: Tensor,
        sgn_lengths: Tensor,
        txt_input: Tensor,
        txt_mask: Tensor,
        phonemes: Tensor,
    ):
        encoder_output, encoder_hidden = self.encode(
            sgn=sgn, sgn_mask=sgn_mask, sgn_length=sgn_lengths, phonemes=phonemes
        )
        unroll_steps = txt_input.size(1)
        decoder_outputs = self.decode(
            encoder_output=encoder_output,
            encoder_hidden=encoder_hidden,
            sgn_mask=sgn_mask,
            txt_input=txt_input,
            unroll_steps=unroll_steps,
            txt_mask=txt_mask,
        )
        return decoder_outputs

    def encode(
        self, sgn: Tensor, sgn_mask: Tensor, sgn_length: Tensor, phonemes: Tensor,
    ) -> (Tensor, Tensor):
        """
        Encodes the source sentence.

        :param sgn:
        :param sgn_mask:
        :param sgn_length:
        :param phonemes: per-frame CTC logits (B, T, num_phonemes)
        :return: encoder outputs (output, hidden_concat)
        """
        x = self.sgn_embed(x=sgn, mask=sgn_mask)
        x = self.fusion(sgn=x, phonemes=phonemes)
        return self.encoder(
            embed_src=x,
            src_length=sgn_length,
            mask=sgn_mask,
        )

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
            phonemes=batch.phonemes,
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
        encoder_output, encoder_hidden = self.encode(
            sgn=batch.sgn, sgn_mask=batch.sgn_mask, sgn_length=batch.sgn_lengths, phonemes=batch.phonemes,
        )

        if translation_beam_size < 2:
            stacked_txt_output, stacked_attention_scores = greedy(
                encoder_hidden=encoder_hidden,
                encoder_output=encoder_output,
                src_mask=batch.sgn_mask,
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
                src_mask=batch.sgn_mask,
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
            "\tfusion=%s,\n"
            "\tencoder=%s,\n"
            "\tdecoder=%s,\n"
            "\tsgn_embed=%s,\n"
            "\ttxt_embed=%s)"
            % (
                self.__class__.__name__,
                self.fusion,
                self.encoder,
                self.decoder,
                self.sgn_embed,
                self.txt_embed,
            )
        )


def build_model(
    cfg: dict,
    sgn_dim: int,
    phoneme_dim: int,
    txt_vocab: Vocabulary,
    multimodal: bool = False,
) -> SignModel:
    txt_padding_idx = txt_vocab.stoi[PAD_TOKEN]

    sgn_embed: SpatialEmbeddings = SpatialEmbeddings(
        **cfg["encoder"]["embeddings"],
        num_heads=cfg["encoder"]["num_heads"],
        input_size=sgn_dim,
        multimodal=multimodal,
    )

    fusion = PhonemeSignFusion(
        embedding_dim=cfg["encoder"]["embeddings"]["embedding_dim"],
        phoneme_dim=phoneme_dim,
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
        fusion=fusion,
        encoder=encoder,
        decoder=decoder,
        sgn_embed=sgn_embed,
        txt_embed=txt_embed,
        txt_vocab=txt_vocab,
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
