#!/usr/bin/env python
import torch

torch.backends.cudnn.deterministic = True

import logging
import os
from typing import List

from torch.utils.data import DataLoader

from main.loss import XentLoss
from main.helpers import load_config, load_checkpoint
from main.metrics import bleu, chrf, rouge
from main.model import build_model, SignModel
from main.batch import Batch
from main.data import load_data, make_data_iter


def format_scores(scores: dict) -> str:
    """One-line summary of the scores returned by `validate_on_data`."""
    return (
        "BLEU-4 {:.2f}\t(BLEU-1: {:.2f},\tBLEU-2: {:.2f},\tBLEU-3: {:.2f},\tBLEU-4: {:.2f})\t"
        "CHRF {:.2f}\tROUGE {:.2f}".format(
            scores["bleu"],
            scores["bleu_scores"]["bleu1"],
            scores["bleu_scores"]["bleu2"],
            scores["bleu_scores"]["bleu3"],
            scores["bleu_scores"]["bleu4"],
            scores["chrf"],
            scores["rouge"],
        )
    )


# pylint: disable=too-many-arguments,too-many-locals,no-member
def validate_on_data(
    model: SignModel,
    data_iter: DataLoader,
    use_cuda: bool,
    translation_loss_function: torch.nn.Module,
    translation_max_output_length: int,
    translation_beam_size: int = 1,
    translation_beam_alpha: float = -1,
) -> dict:
    """
    Generate translations for the given data and compute the loss and scores.

    :param model: model module
    :param data_iter: data loader of the dataset to validate on
    :param use_cuda: if True, use CUDA
    :param translation_loss_function: loss for the validation loss and ppl
    :param translation_max_output_length: maximum length for generated hypotheses
    :param translation_beam_size: beam size for validation.
        If 1 then greedy decoding (default).
    :param translation_beam_alpha: beam search alpha for length penalty,
        disabled if set to -1 (default).
    :return: dict with
        - valid_scores: bleu, bleu_scores, chrf, rouge,
        - valid_translation_loss: summed over all tokens,
        - valid_ppl: perplexity,
        - txt_ref / txt_hyp: references / hypotheses in dataset order
    """
    dataset = data_iter.dataset

    # disable dropout
    model.eval()
    # don't track gradients during validation
    with torch.no_grad():
        txt_outputs = [None] * len(dataset)
        total_translation_loss = 0.0
        total_num_txt_tokens = 0
        for ids, sgn, sgn_lengths, txt in data_iter:
            batch = Batch(
                sgn, sgn_lengths, txt, txt_pad_index=model.txt_pad_index, use_cuda=use_cuda
            )
            batch_translation_loss = model.get_loss_for_batch(
                batch=batch,
                translation_loss_function=translation_loss_function,
                translation_loss_weight=1.0,
            )
            total_translation_loss += batch_translation_loss.item()
            total_num_txt_tokens += batch.num_txt_tokens

            batch_txt_predictions, _ = model.run_batch(
                batch=batch,
                translation_beam_size=translation_beam_size,
                translation_beam_alpha=translation_beam_alpha,
                translation_max_output_length=translation_max_output_length,
            )
            # batches are sorted by length: put outputs back in dataset order
            for i, prediction in zip(ids.tolist(), batch_txt_predictions):
                txt_outputs[i] = prediction

    if total_num_txt_tokens > 0:
        valid_translation_loss = total_translation_loss
        # exponent of token-level negative log prob
        valid_ppl = torch.tensor(total_translation_loss / total_num_txt_tokens).exp().item()
    else:
        valid_translation_loss = -1
        valid_ppl = -1

    txt_hyp = model.txt_vocab.decode_batch(txt_outputs)
    txt_ref = dataset.references
    assert len(txt_ref) == len(txt_hyp)

    txt_bleu = bleu(references=txt_ref, hypotheses=txt_hyp)
    valid_scores = {
        "bleu": txt_bleu["bleu4"],
        "bleu_scores": txt_bleu,
        "chrf": chrf(references=txt_ref, hypotheses=txt_hyp),
        "rouge": rouge(references=txt_ref, hypotheses=txt_hyp),
    }

    return {
        "valid_scores": valid_scores,
        "valid_translation_loss": valid_translation_loss,
        "valid_ppl": valid_ppl,
        "txt_ref": txt_ref,
        "txt_hyp": txt_hyp,
    }


def write_outputs(file_path: str, names: List[str], lines: List[str]) -> None:
    with open(file_path, mode="w", encoding="utf-8") as out_file:
        for name, line in zip(names, lines):
            out_file.write("{}|{}\n".format(name, line))


# pylint: disable-msg=logging-too-many-args
def test(
    cfg_file, ckpt: str = None, output_path: str = None, logger: logging.Logger = None
) -> None:
    """
    Pick the beam size and alpha with the best dev BLEU, then translate test.

    :param cfg_file: path to configuration file
    :param ckpt: checkpoint to load, default: <model_dir>/best.ckpt
    :param output_path: prefix of the files the hypotheses are written to
    :param logger: logger, a new one if None
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            FORMAT = "%(asctime)-15s - %(message)s"
            logging.basicConfig(format=FORMAT)
            logger.setLevel(level=logging.DEBUG)

    cfg = load_config(cfg_file)
    train_cfg = cfg["training"]
    if ckpt is None:
        ckpt = os.path.join(train_cfg["model_dir"], "best.ckpt")

    use_cuda = train_cfg.get("use_cuda", False)
    batch_size = train_cfg.get("eval_batch_size", train_cfg["batch_size"])
    translation_max_output_length = train_cfg.get("translation_max_output_length", 100)

    data, txt_vocab = load_data(data_cfg=cfg["data"], splits=["val", "test"])

    model = build_model(
        cfg=cfg["model"],
        txt_vocab=txt_vocab,
        sgn_dim=cfg["data"]["feature_size"],
        multimodal=cfg["data"].get("multimodal", False),
    )
    model.load_state_dict(load_checkpoint(ckpt, use_cuda=use_cuda)["model_state"])
    if use_cuda:
        model.cuda()

    translation_beam_sizes = cfg.get("testing", {}).get("translation_beam_sizes", [1])
    translation_beam_alphas = cfg.get("testing", {}).get("translation_beam_alphas", [-1])
    translation_loss_function = XentLoss(pad_index=model.txt_pad_index, smoothing=0.0)

    def make_iter(dataset):
        return make_data_iter(
            dataset,
            batch_size=batch_size,
            pad_index=model.txt_pad_index,
            num_workers=cfg["data"].get("num_workers", 4),
            use_cuda=use_cuda,
        )

    def translate(data_iter, beam_size, beam_alpha):
        return validate_on_data(
            model=model,
            data_iter=data_iter,
            use_cuda=use_cuda,
            translation_loss_function=translation_loss_function,
            translation_max_output_length=translation_max_output_length,
            translation_beam_size=beam_size,
            translation_beam_alpha=beam_alpha,
        )

    logger.info("=" * 60)
    dev_iter = make_iter(data["val"])
    dev_best = None
    for beam_size in translation_beam_sizes:
        # greedy decoding has no length penalty
        for beam_alpha in translation_beam_alphas if beam_size > 1 else [-1]:
            dev_result = translate(dev_iter, beam_size, beam_alpha)
            dev_bleu = dev_result["valid_scores"]["bleu"]
            if dev_best is None or dev_bleu > dev_best[2]["valid_scores"]["bleu"]:
                dev_best = (beam_size, beam_alpha, dev_result)
                logger.info(
                    "[DEV] partition [Translation] results:\n\t"
                    "New Best Translation Beam Size: %d and Alpha: %g\n\t%s",
                    beam_size,
                    beam_alpha,
                    format_scores(dev_result["valid_scores"]),
                )
                logger.info("-" * 60)
    del dev_iter
    best_beam_size, best_beam_alpha, dev_best_result = dev_best

    logger.info("*" * 60)
    if len(data["test"]) > 0:
        test_result = translate(make_iter(data["test"]), best_beam_size, best_beam_alpha)
        logger.info(
            "[TEST] partition [Translation] results:\n\t"
            "Best Translation Beam Size: %d and Alpha: %g\n\t%s",
            best_beam_size,
            best_beam_alpha,
            format_scores(test_result["valid_scores"]),
        )
    else:
        test_result = None
        logger.warning("[TEST] partition is empty, skipped.")
    logger.info("*" * 60)

    if output_path is not None:
        prefix = "{}.BW_{:02d}.A_{:g}".format(output_path, best_beam_size, best_beam_alpha)
        write_outputs(prefix + ".dev.txt", data["val"].names, dev_best_result["txt_hyp"])
        if test_result is not None:
            write_outputs(prefix + ".test.txt", data["test"].names, test_result["txt_hyp"])
