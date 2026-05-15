#!/usr/bin/env python
import torch

torch.backends.cudnn.deterministic = True

import logging
import numpy as np
import pickle as pickle
import time

from typing import List
from main.loss import XentLoss
from main.helpers import (
    load_config,
    get_latest_checkpoint,
    load_checkpoint,
)
from main.metrics import bleu, chrf, rouge
from main.model import build_model, SignModel
from main.batch import Batch
from main.data import load_data, make_data_iter
from main.vocabulary import PAD_TOKEN


# pylint: disable=too-many-arguments,too-many-locals,no-member
def validate_on_data(
    model: SignModel,
    data,
    batch_size: int,
    use_cuda: bool,
    sgn_dim: int,
    phoneme_dim: int,
    translation_loss_function,
    translation_loss_weight: float,
    translation_max_output_length: int,
    txt_pad_index: int,
    translation_beam_size: int = 1,
    translation_beam_alpha: int = -1,
):
    valid_iter = make_data_iter(
        dataset=data,
        batch_size=batch_size,
        pad_id=txt_pad_index,
        sgn_dim=sgn_dim,
        phoneme_dim=phoneme_dim,
        train=False,
        shuffle=False,
    )

    model.eval()
    with torch.no_grad():
        all_txt_outputs = []
        all_attention_scores = []
        total_translation_loss = 0
        total_num_txt_tokens = 0
        total_num_seqs = 0

        for sgn, sgn_lengths, txt, txt_lengths, phonemes in valid_iter:
            batch = Batch(
                sgn, sgn_lengths, txt, txt_lengths,
                phonemes=phonemes,
                txt_pad_index=txt_pad_index,
                sgn_dim=sgn_dim,
                use_cuda=use_cuda,
            )
            sort_reverse_index = batch.sort_by_sgn_lengths()

            batch_translation_loss = model.get_loss_for_batch(
                batch=batch,
                translation_loss_function=translation_loss_function,
                translation_loss_weight=translation_loss_weight,
            )

            total_translation_loss += batch_translation_loss
            total_num_txt_tokens += batch.num_txt_tokens
            total_num_seqs += batch.num_seqs

            batch_txt_predictions, batch_attention_scores = model.run_batch(
                batch=batch,
                translation_beam_size=translation_beam_size,
                translation_beam_alpha=translation_beam_alpha,
                translation_max_output_length=translation_max_output_length,
            )

            all_txt_outputs.extend(batch_txt_predictions[sort_reverse_index])
            all_attention_scores.extend(
                batch_attention_scores[sort_reverse_index]
                if batch_attention_scores is not None
                else []
            )

        assert len(all_txt_outputs) == len(data)

        if (
            translation_loss_function is not None
            and translation_loss_weight != 0
            and total_num_txt_tokens > 0
        ):
            valid_translation_loss = total_translation_loss
            valid_ppl = torch.exp(total_translation_loss / total_num_txt_tokens)
        else:
            valid_translation_loss = -1
            valid_ppl = -1

        # Decode hypotheses via SPM vocab
        txt_hyp = model.txt_vocab.decode_batch(all_txt_outputs)
        # References from dataset
        txt_ref = data.txt_references
        assert len(txt_ref) == len(txt_hyp)

        # TXT Metrics
        txt_bleu = bleu(references=txt_ref, hypotheses=txt_hyp)
        txt_chrf = chrf(references=txt_ref, hypotheses=txt_hyp)
        txt_rouge = rouge(references=txt_ref, hypotheses=txt_hyp)

        valid_scores = {
            "bleu": txt_bleu["bleu4"],
            "bleu_scores": txt_bleu,
            "chrf": txt_chrf,
            "rouge": txt_rouge,
        }

    return {
        "valid_scores": valid_scores,
        "valid_translation_loss": valid_translation_loss,
        "valid_ppl": valid_ppl,
        "txt_ref": txt_ref,
        "txt_hyp": txt_hyp,
        "all_attention_scores": all_attention_scores,
    }


# pylint: disable-msg=logging-too-many-args
def test(
    cfg_file, ckpt: str, output_path: str = None, logger: logging.Logger = None
) -> None:
    if logger is None:
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            FORMAT = "%(asctime)-15s - %(message)s"
            logging.basicConfig(format=FORMAT)
            logger.setLevel(level=logging.DEBUG)

    cfg = load_config(cfg_file)

    if ckpt is None:
        model_dir = cfg["training"]["model_dir"]
        ckpt = get_latest_checkpoint(model_dir)
        if ckpt is None:
            raise FileNotFoundError(
                "No checkpoint found in directory {}.".format(model_dir)
            )

    batch_size = cfg["training"]["batch_size"]
    use_cuda = cfg["training"].get("use_cuda", False)
    translation_max_output_length = cfg["training"].get(
        "translation_max_output_length", None
    )

    _, dev_data, test_data, txt_vocab = load_data(data_cfg=cfg["data"])

    model_checkpoint = load_checkpoint(ckpt, use_cuda=use_cuda)

    multimodal = cfg["data"].get("multimodal", False)
    phoneme_dim = cfg["data"]["phoneme_dim"]
    sgn_dim = (
        sum(cfg["data"]["feature_size"])
        if isinstance(cfg["data"]["feature_size"], list)
        else cfg["data"]["feature_size"]
    )
    model = build_model(
        cfg=cfg["model"],
        txt_vocab=txt_vocab,
        sgn_dim=sgn_dim,
        phoneme_dim=phoneme_dim,
        multimodal=multimodal,
    )
    model.load_state_dict(model_checkpoint["model_state"])

    if use_cuda:
        model.cuda()

    if "testing" in cfg.keys():
        translation_beam_sizes = cfg["testing"].get("translation_beam_sizes", [1])
        translation_beam_alphas = cfg["testing"].get("translation_beam_alphas", [-1])
    else:
        translation_beam_sizes = [1]
        translation_beam_alphas = [-1]

    translation_loss_function = XentLoss(
        pad_index=txt_vocab.stoi[PAD_TOKEN], smoothing=0.0
    )
    if use_cuda:
        translation_loss_function.cuda()

    logger.info("=" * 60)
    dev_translation_results = {}
    dev_best_bleu_score = float("-inf")
    dev_best_translation_beam_size = 1
    dev_best_translation_alpha = 1
    for tbw in translation_beam_sizes:
        dev_translation_results[tbw] = {}
        for ta in translation_beam_alphas:
            dev_translation_results[tbw][ta] = validate_on_data(
                model=model,
                data=dev_data,
                batch_size=batch_size,
                use_cuda=use_cuda,
                sgn_dim=sgn_dim,
                phoneme_dim=phoneme_dim,
                translation_loss_function=translation_loss_function,
                translation_loss_weight=1,
                translation_max_output_length=translation_max_output_length,
                txt_pad_index=txt_vocab.stoi[PAD_TOKEN],
                translation_beam_size=tbw,
                translation_beam_alpha=ta,
            )

            if (
                dev_translation_results[tbw][ta]["valid_scores"]["bleu"]
                > dev_best_bleu_score
            ):
                dev_best_bleu_score = dev_translation_results[tbw][ta][
                    "valid_scores"
                ]["bleu"]
                dev_best_translation_beam_size = tbw
                dev_best_translation_alpha = ta
                dev_best_translation_result = dev_translation_results[tbw][ta]
                logger.info(
                    "[DEV] partition [Translation] results:\n\t"
                    "New Best Translation Beam Size: %d and Alpha: %d\n\t"
                    "BLEU-4 %.2f\t(BLEU-1: %.2f,\tBLEU-2: %.2f,\tBLEU-3: %.2f,\tBLEU-4: %.2f)\n\t"
                    "CHRF %.2f\t"
                    "ROUGE %.2f",
                    dev_best_translation_beam_size,
                    dev_best_translation_alpha,
                    dev_best_translation_result["valid_scores"]["bleu"],
                    dev_best_translation_result["valid_scores"]["bleu_scores"]["bleu1"],
                    dev_best_translation_result["valid_scores"]["bleu_scores"]["bleu2"],
                    dev_best_translation_result["valid_scores"]["bleu_scores"]["bleu3"],
                    dev_best_translation_result["valid_scores"]["bleu_scores"]["bleu4"],
                    dev_best_translation_result["valid_scores"]["chrf"],
                    dev_best_translation_result["valid_scores"]["rouge"],
                )
                logger.info("-" * 60)

    logger.info("*" * 60)

    test_best_result = validate_on_data(
        model=model,
        data=test_data,
        batch_size=batch_size,
        use_cuda=use_cuda,
        sgn_dim=sgn_dim,
        phoneme_dim=phoneme_dim,
        txt_pad_index=txt_vocab.stoi[PAD_TOKEN],
        translation_loss_function=translation_loss_function,
        translation_loss_weight=1,
        translation_max_output_length=translation_max_output_length,
        translation_beam_size=dev_best_translation_beam_size,
        translation_beam_alpha=dev_best_translation_alpha,
    )

    logger.info(
        "[TEST] partition [Translation] results:\n\t"
        "Best Translation Beam Size: %d and Alpha: %d\n\t"
        "BLEU-4 %.2f\t(BLEU-1: %.2f,\tBLEU-2: %.2f,\tBLEU-3: %.2f,\tBLEU-4: %.2f)\n\t"
        "CHRF %.2f\t"
        "ROUGE %.2f",
        dev_best_translation_beam_size,
        dev_best_translation_alpha,
        test_best_result["valid_scores"]["bleu"],
        test_best_result["valid_scores"]["bleu_scores"]["bleu1"],
        test_best_result["valid_scores"]["bleu_scores"]["bleu2"],
        test_best_result["valid_scores"]["bleu_scores"]["bleu3"],
        test_best_result["valid_scores"]["bleu_scores"]["bleu4"],
        test_best_result["valid_scores"]["chrf"],
        test_best_result["valid_scores"]["rouge"],
    )
    logger.info("*" * 60)

    def _write_to_file(file_path: str, hypotheses: List[str]):
        with open(file_path, mode="w", encoding="utf-8") as out_file:
            for i, hyp in enumerate(hypotheses):
                out_file.write(str(i) + "|" + hyp + "\n")

    if output_path is not None:
        dev_txt_output_path_set = "{}.BW_{:02d}.A_{:1d}.{}.txt".format(
            output_path,
            dev_best_translation_beam_size,
            dev_best_translation_alpha,
            "dev",
        )
        test_txt_output_path_set = "{}.BW_{:02d}.A_{:1d}.{}.txt".format(
            output_path,
            dev_best_translation_beam_size,
            dev_best_translation_alpha,
            "test",
        )

        _write_to_file(dev_txt_output_path_set, dev_best_translation_result["txt_hyp"])
        _write_to_file(test_txt_output_path_set, test_best_result["txt_hyp"])

        with open(output_path + ".dev_results.pkl", "wb") as out:
            pickle.dump(
                {"translation_results": dev_translation_results},
                out,
            )
        with open(output_path + ".test_results.pkl", "wb") as out:
            pickle.dump(test_best_result, out)
