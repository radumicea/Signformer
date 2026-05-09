#!/usr/bin/env python
import torch

torch.backends.cudnn.deterministic = True

import argparse
import random
import numpy as np
import os
import shutil
import time

from main.model import build_model
from main.batch import Batch
from main.helpers import (
    log_data_info,
    load_config,
    log_cfg,
    load_checkpoint,
    make_model_dir,
    make_logger,
    set_seed,
    symlink_update,
)
from main.model import SignModel
from main.prediction import validate_on_data
from main.loss import XentLoss
from main.data import load_data, make_data_iter
from main.builders import build_optimizer, build_scheduler, build_gradient_clipper
from main.prediction import test
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from typing import List, Dict

# pylint: disable=too-many-instance-attributes
class TrainManager:
    """ Manages training loop, validations, learning rate scheduling
    and early stopping."""

    def __init__(self, model: SignModel, config: dict) -> None:
        train_config = config["training"]

        # files for logging and storing
        self.model_dir = make_model_dir(
            train_config["model_dir"], overwrite=train_config.get("overwrite", False)
        )
        self.logger = make_logger(model_dir=self.model_dir)
        self.logging_freq = train_config.get("logging_freq", 100)
        self.valid_report_file = "{}/validations.txt".format(self.model_dir)
        self.tb_writer = SummaryWriter(log_dir=self.model_dir + "/tensorboard/")

        # input
        self.feature_size = (
            sum(config["data"]["feature_size"])
            if isinstance(config["data"]["feature_size"], list)
            else config["data"]["feature_size"]
        )

        # model
        self.model = model
        self.txt_pad_index = self.model.txt_pad_index
        self.txt_bos_index = self.model.txt_bos_index
        self._log_parameters_list()

        # Translation parameters
        self.label_smoothing = train_config.get("label_smoothing", 0.0)
        self.translation_loss_function = XentLoss(
            pad_index=self.txt_pad_index, smoothing=self.label_smoothing
        )
        self.translation_normalization_mode = train_config.get(
            "translation_normalization", "batch"
        )
        if self.translation_normalization_mode not in ["batch", "tokens"]:
            raise ValueError(
                "Invalid normalization {}.".format(self.translation_normalization_mode)
            )
        self.translation_loss_weight = train_config.get("translation_loss_weight", 1.0)
        self.eval_translation_beam_size = train_config.get(
            "eval_translation_beam_size", 1
        )
        self.eval_translation_beam_alpha = train_config.get(
            "eval_translation_beam_alpha", -1
        )
        self.translation_max_output_length = train_config.get(
            "translation_max_output_length", None
        )

        # optimization
        self.last_best_lr = train_config.get("learning_rate", -1)
        self.learning_rate_min = train_config.get("learning_rate_min", 1.0e-8)
        self.clip_grad_fun = build_gradient_clipper(config=train_config)

        params = model.parameters()
        self.optimizer = build_optimizer(
            config=train_config, parameters=params
        )
        self.batch_multiplier = train_config.get("batch_multiplier", 1)

        # validation & early stopping
        self.num_valid_log = train_config.get("num_valid_log", 5)
        self.eval_metric = train_config.get("eval_metric", "bleu")
        if self.eval_metric not in ["bleu", "chrf", "rouge"]:
            raise ValueError(
                "Invalid setting for 'eval_metric': {}".format(self.eval_metric)
            )
        self.early_stopping_metric = train_config.get(
            "early_stopping_metric", "eval_metric"
        )

        if self.early_stopping_metric in ["ppl", "translation_loss"]:
            self.minimize_metric = True
        elif self.early_stopping_metric == "eval_metric":
            if self.eval_metric in ["bleu", "chrf", "rouge"]:
                self.minimize_metric = False
            else:
                self.minimize_metric = True
        else:
            raise ValueError(
                "Invalid setting for 'early_stopping_metric': {}".format(
                    self.early_stopping_metric
                )
            )

        # learning rate scheduling
        self.scheduler, self.scheduler_step_at = build_scheduler(
            config=train_config,
            scheduler_mode="min" if self.minimize_metric else "max",
            optimizer=self.optimizer,
            hidden_size=config["model"]["encoder"]["hidden_size"],
        )

        # data & batch handling
        self.shuffle = train_config.get("shuffle", True)
        self.epochs = train_config["epochs"]
        self.batch_size = train_config["batch_size"]
        self.eval_batch_size = train_config.get("eval_batch_size", self.batch_size)

        self.use_cuda = train_config["use_cuda"]
        if self.use_cuda:
            self.model.cuda()
            self.translation_loss_function.cuda()

        # initialize training statistics
        self.steps = 0
        self.stop = False
        self.total_txt_tokens = 0
        self.best_ckpt_iteration = 0
        self.best_ckpt_score = np.inf if self.minimize_metric else -np.inf
        self.best_all_ckpt_scores = {}
        self.current_epoch = 0
        self.is_best = (
            lambda score: score < self.best_ckpt_score
            if self.minimize_metric
            else score > self.best_ckpt_score
        )

        # model parameters
        if "load_model" in train_config.keys():
            model_load_path = train_config["load_model"]
            self.logger.info("Loading model from %s", model_load_path)
            reset_best_ckpt = train_config.get("reset_best_ckpt", False)
            reset_scheduler = train_config.get("reset_scheduler", False)
            reset_optimizer = train_config.get("reset_optimizer", False)
            self.init_from_checkpoint(
                model_load_path,
                reset_best_ckpt=reset_best_ckpt,
                reset_scheduler=reset_scheduler,
                reset_optimizer=reset_optimizer,
            )

    def _save_checkpoint(self, is_best: bool = False) -> None:
        state = {
            "steps": self.steps,
            "epoch": self.current_epoch,
            "total_txt_tokens": self.total_txt_tokens,
            "best_ckpt_score": self.best_ckpt_score,
            "best_all_ckpt_scores": self.best_all_ckpt_scores,
            "best_ckpt_iteration": self.best_ckpt_iteration,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict()
            if self.scheduler is not None
            else None,
            "torch_rng_state": torch.random.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
            "cuda_rng_state": torch.cuda.get_rng_state()
            if self.use_cuda
            else None,
        }
        latest_path = os.path.join(self.model_dir, "latest.ckpt")
        torch.save(state, latest_path)
        if is_best:
            best_path = os.path.join(self.model_dir, "best.ckpt")
            torch.save(state, best_path)
            self.logger.info("Saving new best checkpoint.")

    def init_from_checkpoint(
        self,
        path: str,
        reset_best_ckpt: bool = False,
        reset_scheduler: bool = False,
        reset_optimizer: bool = False,
    ) -> None:
        """
        Initialize the trainer from a given checkpoint file.

        This checkpoint file contains not only model parameters, but also
        scheduler and optimizer states, see `self._save_checkpoint`.

        :param path: path to checkpoint
        :param reset_best_ckpt: reset tracking of the best checkpoint,
                                use for domain adaptation with a new dev
                                set or when using a new metric for fine-tuning.
        :param reset_scheduler: reset the learning rate scheduler, and do not
                                use the one stored in the checkpoint.
        :param reset_optimizer: reset the optimizer, and do not use the one
                                stored in the checkpoint.
        """
        model_checkpoint = load_checkpoint(path=path, use_cuda=self.use_cuda)

        # restore model and optimizer parameters
        self.model.load_state_dict(model_checkpoint["model_state"])

        if not reset_optimizer:
            self.optimizer.load_state_dict(model_checkpoint["optimizer_state"])
        else:
            self.logger.info("Reset optimizer.")

        if not reset_scheduler:
            if (
                model_checkpoint["scheduler_state"] is not None
                and self.scheduler is not None
            ):
                self.scheduler.load_state_dict(model_checkpoint["scheduler_state"])
        else:
            self.logger.info("Reset scheduler.")

        # restore counts
        self.steps = model_checkpoint["steps"]
        self.total_txt_tokens = model_checkpoint["total_txt_tokens"]
        self.current_epoch = model_checkpoint.get("epoch", 0) + 1

        if not reset_best_ckpt:
            self.best_ckpt_score = model_checkpoint["best_ckpt_score"]
            self.best_all_ckpt_scores = model_checkpoint["best_all_ckpt_scores"]
            self.best_ckpt_iteration = model_checkpoint["best_ckpt_iteration"]
        else:
            self.logger.info("Reset tracking of the best checkpoint.")

        # restore rng states for exact resumption
        if "torch_rng_state" in model_checkpoint:
            torch.random.set_rng_state(model_checkpoint["torch_rng_state"])
        if "numpy_rng_state" in model_checkpoint:
            np.random.set_state(model_checkpoint["numpy_rng_state"])
        if "python_rng_state" in model_checkpoint:
            random.setstate(model_checkpoint["python_rng_state"])
        if (
            "cuda_rng_state" in model_checkpoint
            and model_checkpoint["cuda_rng_state"] is not None
        ):
            torch.cuda.set_rng_state(model_checkpoint["cuda_rng_state"])

        # move parameters to cuda
        if self.use_cuda:
            self.model.cuda()

    def train_and_validate(self, train_data, valid_data) -> None:
        train_iter = make_data_iter(
            train_data,
            batch_size=self.batch_size,
            pad_id=self.txt_pad_index,
            sgn_dim=self.feature_size,
            train=True,
            shuffle=self.shuffle,
        )
        epoch_no = None
        for epoch_no in range(self.current_epoch, self.epochs):
            self.current_epoch = epoch_no
            self.logger.info("EPOCH %d", epoch_no + 1)

            if self.scheduler is not None and self.scheduler_step_at == "epoch":
                self.scheduler.step(epoch=epoch_no)

            self.model.train()
            start = time.time()
            count = self.batch_multiplier - 1

            processed_txt_tokens = self.total_txt_tokens
            epoch_translation_loss = 0

            for sgn, sgn_lengths, txt, txt_lengths in train_iter:
                batch = Batch(
                    sgn, sgn_lengths, txt, txt_lengths,
                    txt_pad_index=self.txt_pad_index,
                    sgn_dim=self.feature_size,
                    use_cuda=self.use_cuda,
                )

                update = count == 0

                translation_loss = self._train_batch(batch, update=update)

                self.tb_writer.add_scalar(
                    "train/train_translation_loss", translation_loss, self.steps
                )
                epoch_translation_loss += translation_loss.detach().cpu().numpy()

                count = self.batch_multiplier if update else count
                count -= 1

                if (
                    self.scheduler is not None
                    and self.scheduler_step_at == "step"
                    and update
                ):
                    self.scheduler.step()

                # log learning progress
                if self.steps % self.logging_freq == 0 and update:
                    elapsed = time.time() - start

                    log_out = "[Epoch: {:03d} Step: {:08d}] ".format(
                        epoch_no + 1, self.steps,
                    )

                    elapsed_txt_tokens = (
                        self.total_txt_tokens - processed_txt_tokens
                    )
                    processed_txt_tokens = self.total_txt_tokens
                    log_out += "Batch Translation Loss: {:10.6f} => ".format(
                        translation_loss
                    )
                    log_out += "Txt Tokens per Sec: {:8.0f} || ".format(
                        elapsed_txt_tokens / elapsed
                    )
                    log_out += "Lr: {:.6f}".format(self.optimizer.param_groups[0]["lr"])
                    self.logger.info(log_out)
                    start = time.time()

            self.logger.info(
                "Epoch %3d: Total Training Translation Loss %.2f",
                epoch_no + 1,
                epoch_translation_loss,
            )

            # validate on the entire dev set at end of epoch
            valid_start_time = time.time()
            val_res = validate_on_data(
                model=self.model,
                data=valid_data,
                batch_size=self.eval_batch_size,
                use_cuda=self.use_cuda,
                sgn_dim=self.feature_size,
                txt_pad_index=self.txt_pad_index,
                translation_loss_function=self.translation_loss_function,
                translation_max_output_length=self.translation_max_output_length,
                translation_loss_weight=self.translation_loss_weight,
                translation_beam_size=self.eval_translation_beam_size,
                translation_beam_alpha=self.eval_translation_beam_alpha,
            )
            self.model.train()
            self.tb_writer.add_scalar(
                "learning_rate",
                self.scheduler.optimizer.param_groups[0]["lr"],
                self.steps,
            )
            self.tb_writer.add_scalar(
                "valid/valid_translation_loss",
                val_res["valid_translation_loss"],
                self.steps,
            )
            self.tb_writer.add_scalar(
                "valid/valid_ppl", val_res["valid_ppl"], self.steps
            )
            self.tb_writer.add_scalar(
                "valid/chrf", val_res["valid_scores"]["chrf"], self.steps
            )
            self.tb_writer.add_scalar(
                "valid/rouge", val_res["valid_scores"]["rouge"], self.steps
            )
            self.tb_writer.add_scalar(
                "valid/bleu", val_res["valid_scores"]["bleu"], self.steps
            )
            self.tb_writer.add_scalars(
                "valid/bleu_scores",
                val_res["valid_scores"]["bleu_scores"],
                self.steps,
            )

            if self.early_stopping_metric == "translation_loss":
                ckpt_score = val_res["valid_translation_loss"]
            elif self.early_stopping_metric in ["ppl", "perplexity"]:
                ckpt_score = val_res["valid_ppl"]
            else:
                ckpt_score = val_res["valid_scores"][self.eval_metric]

            new_best = False
            if self.is_best(ckpt_score):
                self.best_ckpt_score = ckpt_score
                self.best_all_ckpt_scores = val_res["valid_scores"]
                self.best_ckpt_iteration = self.steps
                self.logger.info(
                    "Hooray! New best validation result [%s]!",
                    self.early_stopping_metric,
                )
                new_best = True

            self._save_checkpoint(is_best=new_best)

            if (
                self.scheduler is not None
                and self.scheduler_step_at == "validation"
            ):
                prev_lr = self.scheduler.optimizer.param_groups[0]["lr"]
                self.scheduler.step(ckpt_score)
                now_lr = self.scheduler.optimizer.param_groups[0]["lr"]

            # append to validation report
            self._add_report(
                valid_scores=val_res["valid_scores"],
                valid_translation_loss=val_res["valid_translation_loss"],
                valid_ppl=val_res["valid_ppl"],
                eval_metric=self.eval_metric,
                new_best=new_best,
            )
            valid_duration = time.time() - valid_start_time
            self.logger.info(
                "Validation result at epoch %3d, step %8d: duration: %.4fs\n\t"
                "Translation Beam Size: %d\t"
                "Translation Beam Alpha: %d\n\t"
                "Translation Loss: %4.5f\t"
                "PPL: %4.5f\n\t"
                "Eval Metric: %s\n\t"
                "BLEU-4 %.2f\t(BLEU-1: %.2f,\tBLEU-2: %.2f,\tBLEU-3: %.2f,\tBLEU-4: %.2f)\n\t"
                "CHRF %.2f\t"
                "ROUGE %.2f",
                epoch_no + 1,
                self.steps,
                valid_duration,
                self.eval_translation_beam_size,
                self.eval_translation_beam_alpha,
                val_res["valid_translation_loss"],
                val_res["valid_ppl"],
                self.eval_metric.upper(),
                val_res["valid_scores"]["bleu"],
                val_res["valid_scores"]["bleu_scores"]["bleu1"],
                val_res["valid_scores"]["bleu_scores"]["bleu2"],
                val_res["valid_scores"]["bleu_scores"]["bleu3"],
                val_res["valid_scores"]["bleu_scores"]["bleu4"],
                val_res["valid_scores"]["chrf"],
                val_res["valid_scores"]["rouge"],
            )

            self._log_examples(
                txt_references=val_res["txt_ref"],
                txt_hypotheses=val_res["txt_hyp"],
            )

            self._store_outputs(
                "dev.hyp.txt", val_res["txt_hyp"], "txt"
            )
            self._store_outputs(
                "references.dev.txt", val_res["txt_ref"]
            )

            if self.stop:
                if (
                    self.scheduler is not None
                    and self.scheduler_step_at == "validation"
                    and self.last_best_lr != prev_lr
                ):
                    self.logger.info(
                        "Training ended since there were no improvements in"
                        "the last learning rate step: %f",
                        prev_lr,
                    )
                else:
                    self.logger.info(
                        "Training ended since minimum lr %f was reached.",
                        self.learning_rate_min,
                    )
                break

        else:
            self.logger.info("Training ended after %3d epochs.", epoch_no + 1)
        self.logger.info(
            "Best validation result at step %8d: %6.2f %s.",
            self.best_ckpt_iteration,
            self.best_ckpt_score,
            self.early_stopping_metric,
        )

        self.tb_writer.close()

    def _train_batch(self, batch: Batch, update: bool = True) -> Tensor:
        translation_loss = self.model.get_loss_for_batch(
            batch=batch,
            translation_loss_function=self.translation_loss_function,
            translation_loss_weight=self.translation_loss_weight,
        )

        if self.translation_normalization_mode == "batch":
            txt_normalization_factor = batch.num_seqs
        elif self.translation_normalization_mode == "tokens":
            txt_normalization_factor = batch.num_txt_tokens
        else:
            raise NotImplementedError("Only normalize by 'batch' or 'tokens'")

        normalized_translation_loss = translation_loss / (
            txt_normalization_factor * self.batch_multiplier
        )

        normalized_translation_loss.backward()

        if self.clip_grad_fun is not None:
            self.clip_grad_fun(params=self.model.parameters())

        if update:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.steps += 1

        self.total_txt_tokens += batch.num_txt_tokens

        return normalized_translation_loss

    def _add_report(
        self,
        valid_scores: Dict,
        valid_translation_loss: float,
        valid_ppl: float,
        eval_metric: str,
        new_best: bool = False,
    ) -> None:
        current_lr = -1
        for param_group in self.optimizer.param_groups:
            current_lr = param_group["lr"]

        if new_best:
            self.last_best_lr = current_lr

        if current_lr < self.learning_rate_min:
            self.stop = True

        with open(self.valid_report_file, "a", encoding="utf-8") as opened_file:
            opened_file.write(
                "Steps: {}\t"
                "Translation Loss: {:.5f}\t"
                "PPL: {:.5f}\t"
                "Eval Metric: {}\t"
                "BLEU-4 {:.2f}\t(BLEU-1: {:.2f},\tBLEU-2: {:.2f},\tBLEU-3: {:.2f},\tBLEU-4: {:.2f})\t"
                "CHRF {:.2f}\t"
                "ROUGE {:.2f}\t"
                "LR: {:.8f}\t{}\n".format(
                    self.steps,
                    valid_translation_loss,
                    valid_ppl,
                    eval_metric,
                    valid_scores["bleu"],
                    valid_scores["bleu_scores"]["bleu1"],
                    valid_scores["bleu_scores"]["bleu2"],
                    valid_scores["bleu_scores"]["bleu3"],
                    valid_scores["bleu_scores"]["bleu4"],
                    valid_scores["chrf"],
                    valid_scores["rouge"],
                    current_lr,
                    "*" if new_best else "",
                )
            )

    def _log_parameters_list(self) -> None:
        """
        Write all model parameters (name, shape) to the log.
        """
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        n_params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info(f"Total params: {n_params:,}")
        trainable_params = [
            n for (n, p) in self.model.named_parameters() if p.requires_grad
        ]
        self.logger.info("Trainable parameters: %s", sorted(trainable_params))
        assert trainable_params

    def _log_examples(
        self,
        txt_references: List[str],
        txt_hypotheses: List[str],
    ) -> None:
        assert len(txt_references) == len(txt_hypotheses)
        num_sequences = len(txt_hypotheses)
        rand_idx = np.sort(np.random.permutation(num_sequences)[: self.num_valid_log])
        self.logger.info("Logging Translation Outputs")
        self.logger.info("=" * 120)
        for ri in rand_idx:
            self.logger.info("\tReference : %s", txt_references[ri])
            self.logger.info("\tHypothesis: %s", txt_hypotheses[ri])
            self.logger.info("=" * 120)

    def _store_outputs(
        self, tag: str, hypotheses: List[str], sub_folder=None
    ) -> None:
        if sub_folder:
            out_folder = os.path.join(self.model_dir, sub_folder)
            if not os.path.exists(out_folder):
                os.makedirs(out_folder)
            current_valid_output_file = "{}/{}.{}".format(out_folder, self.steps, tag)
        else:
            out_folder = self.model_dir
            current_valid_output_file = "{}/{}".format(out_folder, tag)

        with open(current_valid_output_file, "w", encoding="utf-8") as opened_file:
            for i, hyp in enumerate(hypotheses):
                opened_file.write("{}|{}\n".format(i, hyp))


def train(cfg_file: str) -> None:
    cfg = load_config(cfg_file)
    set_seed(seed=cfg["training"].get("random_seed", 42))

    train_data, dev_data, test_data, txt_vocab = load_data(data_cfg=cfg["data"])

    multimodal = cfg["data"].get("multimodal", False)
    model = build_model(
        cfg=cfg["model"],
        txt_vocab=txt_vocab,
        sgn_dim=sum(cfg["data"]["feature_size"])
        if isinstance(cfg["data"]["feature_size"], list)
        else cfg["data"]["feature_size"],
        multimodal=multimodal,
    )

    trainer = TrainManager(model=model, config=cfg)
    shutil.copy2(cfg_file, trainer.model_dir + "/config.yaml")
    log_cfg(cfg, trainer.logger)
    log_data_info(
        train_data=train_data,
        valid_data=dev_data,
        test_data=test_data,
        txt_vocab=txt_vocab,
        logging_function=trainer.logger.info,
    )
    trainer.logger.info(str(model))

    txt_vocab_file = "{}/txt.vocab".format(cfg["training"]["model_dir"])
    txt_vocab.to_file(txt_vocab_file)

    trainer.train_and_validate(train_data=train_data, valid_data=dev_data)
    del train_data, dev_data, test_data

    ckpt = os.path.join(trainer.model_dir, "best.ckpt")
    output_name = "best.IT_{:08d}".format(trainer.best_ckpt_iteration)
    output_path = os.path.join(trainer.model_dir, output_name)
    logger = trainer.logger
    del trainer
    test(cfg_file, ckpt=ckpt, output_path=output_path, logger=logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Joey-NMT")
    parser.add_argument(
        "config",
        default="configs/default.yaml",
        type=str,
        help="Training configuration file (yaml).",
    )
    parser.add_argument(
        "--gpu_id", type=str, default="0", help="gpu to run your job on"
    )
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    train(cfg_file=args.config)
