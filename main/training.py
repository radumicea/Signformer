#!/usr/bin/env python
import torch

torch.backends.cudnn.deterministic = True

import math
import numpy as np
import os
import random
import shutil
import time

from main.model import build_model, SignModel
from main.batch import Batch
from main.helpers import (
    log_data_info,
    load_config,
    log_cfg,
    load_checkpoint,
    make_model_dir,
    make_logger,
    set_seed,
)
from main.prediction import validate_on_data, format_scores, write_outputs, test
from main.loss import XentLoss
from main.data import load_data, make_data_iter
from main.builders import build_optimizer, build_scheduler, build_gradient_clipper
from main.dataset import SignTranslationDataset
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


# pylint: disable=too-many-instance-attributes
class TrainManager:
    """ Manages training loop, validations, learning rate scheduling
    and early stopping."""

    def __init__(self, model: SignModel, config: dict, resume: bool = False) -> None:
        """
        Creates a new TrainManager for a model, specified as in configuration.

        :param model: torch module defining the model
        :param config: dictionary containing the training configurations
        :param resume: continue the training in model_dir from latest.ckpt
        """
        train_config = config["training"]

        # files for logging and storing
        self.model_dir = make_model_dir(
            train_config["model_dir"],
            overwrite=train_config.get("overwrite", False),
            resume=resume,
        )
        self.logger = make_logger(model_dir=self.model_dir)
        self.logging_freq = train_config.get("logging_freq", 100)
        self.valid_report_file = os.path.join(self.model_dir, "validations.txt")

        # model
        self.model = model
        self.txt_pad_index = self.model.txt_pad_index
        self._log_parameters_list()

        # translation
        self.label_smoothing = train_config.get("label_smoothing", 0.0)
        self.translation_loss_function = XentLoss(
            pad_index=self.txt_pad_index, smoothing=self.label_smoothing
        )
        # validation loss and ppl without label smoothing
        self.valid_loss_function = XentLoss(pad_index=self.txt_pad_index, smoothing=0.0)
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
            "translation_max_output_length", 100
        )

        # optimization
        self.learning_rate = train_config["learning_rate"]
        self.learning_rate_min = train_config.get("learning_rate_min", 1.0e-8)
        self.clip_grad_fun = build_gradient_clipper(config=train_config)
        self.optimizer = build_optimizer(
            config=train_config, parameters=model.parameters()
        )
        # gradient accumulation: update every batch_multiplier batches
        self.batch_multiplier = train_config.get("batch_multiplier", 1)
        # linear warmup from 0 to learning_rate, in epochs (can be fractional);
        # the scheduler only takes over after it
        self.warmup_epochs = train_config.get("warmup_epochs", 0)
        # both set from the number of batches per epoch
        self.warmup_steps = 0
        self.total_steps = 0

        # validation & early stopping
        self.validation_freq = train_config.get("validation_freq", 1)
        self.num_valid_log = train_config.get("num_valid_log", 5)
        self.eval_metric = train_config.get("eval_metric", "bleu")
        if self.eval_metric not in ["bleu", "chrf", "rouge"]:
            raise ValueError(
                "Invalid setting for 'eval_metric': {}".format(self.eval_metric)
            )
        self.early_stopping_metric = train_config.get(
            "early_stopping_metric", "eval_metric"
        )

        # if we schedule after BLEU/chrf/rouge, we want to maximize it, else minimize
        if self.early_stopping_metric in ["ppl", "translation_loss"]:
            self.minimize_metric = True
        elif self.early_stopping_metric == "eval_metric":
            self.minimize_metric = False
        else:
            raise ValueError(
                "Invalid setting for 'early_stopping_metric': {}".format(
                    self.early_stopping_metric
                )
            )

        # learning rate scheduling
        self.scheduling = train_config["scheduling"]
        if self.scheduling in ["cosine", "linear"]:
            # set before every update, see _set_learning_rate
            self.scheduler, self.scheduler_step_at = None, None
        else:
            self.scheduler, self.scheduler_step_at = build_scheduler(
                config=train_config,
                scheduler_mode="min" if self.minimize_metric else "max",
                optimizer=self.optimizer,
                hidden_size=config["model"]["encoder"]["hidden_size"],
            )

        # data & batch handling
        self.shuffle = train_config.get("shuffle", True)
        self.seed = train_config.get("random_seed", 42)
        self.epochs = train_config["epochs"]
        self.batch_size = train_config["batch_size"]
        self.eval_batch_size = train_config.get("eval_batch_size", self.batch_size)
        self.num_workers = config["data"].get("num_workers", 4)

        self.use_cuda = train_config["use_cuda"]
        if self.use_cuda:
            self.model.cuda()

        # initialize training statistics
        self.epoch = 0  # completed epochs
        self.steps = 0  # optimizer updates
        # stop training if this flag is True by reaching learning rate minimum
        self.stop = False
        self.total_txt_tokens = 0
        self.best_ckpt_epoch = 0
        self.best_ckpt_steps = 0
        # initial values for best scores
        self.best_ckpt_score = np.inf if self.minimize_metric else -np.inf
        self.best_all_ckpt_scores = {}

        if resume:
            self._load_checkpoint(os.path.join(self.model_dir, "latest.ckpt"))
            self.logger.info("Resuming after epoch %d, step %d.", self.epoch, self.steps)

        # drop events logged after the checkpoint by an interrupted run
        self.tb_writer = SummaryWriter(
            log_dir=os.path.join(self.model_dir, "tensorboard"),
            purge_step=self.steps if resume else None,
        )

    def is_best(self, score: float) -> bool:
        if self.minimize_metric:
            return score < self.best_ckpt_score
        return score > self.best_ckpt_score

    def _save_checkpoint(self, name: str) -> None:
        """
        Save the model's current parameters and the training state (counters,
        best score, optimizer, scheduler and RNG states) to
        `model_dir/name`, so that training can be resumed exactly.
        """
        state = {
            "epoch": self.epoch,
            "steps": self.steps,
            "total_txt_tokens": self.total_txt_tokens,
            "best_ckpt_score": self.best_ckpt_score,
            "best_all_ckpt_scores": self.best_all_ckpt_scores,
            "best_ckpt_epoch": self.best_ckpt_epoch,
            "best_ckpt_steps": self.best_ckpt_steps,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict()
            if self.scheduler is not None
            else None,
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if self.use_cuda else None,
            },
        }
        path = os.path.join(self.model_dir, name)
        # write to a temporary file first: an interruption can't corrupt the checkpoint
        torch.save(state, path + ".tmp")
        os.replace(path + ".tmp", path)

    def _load_checkpoint(self, path: str) -> None:
        """
        Restore the training state saved by `self._save_checkpoint`.

        :param path: path to checkpoint
        """
        checkpoint = load_checkpoint(path=path, use_cuda=self.use_cuda)

        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        if checkpoint["scheduler_state"] is not None and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])

        self.epoch = checkpoint["epoch"]
        self.steps = checkpoint["steps"]
        self.total_txt_tokens = checkpoint["total_txt_tokens"]
        self.best_ckpt_score = checkpoint["best_ckpt_score"]
        self.best_all_ckpt_scores = checkpoint["best_all_ckpt_scores"]
        self.best_ckpt_epoch = checkpoint["best_ckpt_epoch"]
        self.best_ckpt_steps = checkpoint["best_ckpt_steps"]

        rng_state = checkpoint["rng_state"]
        random.setstate(rng_state["python"])
        np.random.set_state(rng_state["numpy"])
        torch.set_rng_state(rng_state["torch"].cpu())
        if self.use_cuda and rng_state["cuda"] is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in rng_state["cuda"]])

    def train_and_validate(
        self, train_data: SignTranslationDataset, valid_data: SignTranslationDataset
    ) -> None:
        """
        Train the model and validate it every `validation_freq` epochs.

        :param train_data: training data
        :param valid_data: validation data
        """
        train_iter = make_data_iter(
            train_data,
            batch_size=self.batch_size,
            pad_index=self.txt_pad_index,
            train=True,
            shuffle=self.shuffle,
            seed=self.seed,
            num_workers=self.num_workers,
            use_cuda=self.use_cuda,
        )
        valid_iter = make_data_iter(
            valid_data,
            batch_size=self.eval_batch_size,
            pad_index=self.txt_pad_index,
            num_workers=self.num_workers,
            use_cuda=self.use_cuda,
        )
        num_batches = len(train_iter)
        updates_per_epoch = math.ceil(num_batches / self.batch_multiplier)
        self.warmup_steps = round(self.warmup_epochs * updates_per_epoch)
        self.total_steps = self.epochs * updates_per_epoch
        # log the same validation sentences every time
        valid_log_idx = np.sort(
            np.random.default_rng(self.seed).permutation(len(valid_data))[
                : self.num_valid_log
            ]
        )

        for epoch_no in range(self.epoch, self.epochs):
            self.logger.info("EPOCH %d", epoch_no + 1)
            train_iter.batch_sampler.set_epoch(epoch_no)

            self.model.train()
            start = time.time()
            processed_txt_tokens = self.total_txt_tokens
            epoch_translation_loss = 0

            for i, (_, sgn, sgn_lengths, txt) in enumerate(train_iter):
                batch = Batch(
                    sgn, sgn_lengths, txt,
                    txt_pad_index=self.txt_pad_index,
                    use_cuda=self.use_cuda,
                )

                # only update every batch_multiplier batches, and at the end
                # of the epoch, so that epochs are independent of each other
                update = (i + 1) % self.batch_multiplier == 0 or i + 1 == num_batches

                translation_loss = self._train_batch(batch, update=update)
                epoch_translation_loss += translation_loss

                # log learning progress
                if update and self.steps % self.logging_freq == 0:
                    elapsed = time.time() - start
                    elapsed_txt_tokens = self.total_txt_tokens - processed_txt_tokens
                    lr = self.optimizer.param_groups[0]["lr"]
                    self.logger.info(
                        "[Epoch: %03d Step: %08d] Batch Translation Loss: %10.6f => "
                        "Txt Tokens per Sec: %8.0f || Lr: %.6f",
                        epoch_no + 1,
                        self.steps,
                        translation_loss,
                        elapsed_txt_tokens / elapsed,
                        lr,
                    )
                    self.tb_writer.add_scalar(
                        "train/train_translation_loss", translation_loss, self.steps
                    )
                    self.tb_writer.add_scalar("learning_rate", lr, self.steps)
                    start = time.time()
                    processed_txt_tokens = self.total_txt_tokens

            self.epoch = epoch_no + 1
            self.logger.info(
                "Epoch %3d: Total Training Translation Loss %.2f",
                self.epoch,
                epoch_translation_loss,
            )

            # the first epoch after the warmup still uses the full learning rate
            if (
                self.scheduler is not None
                and self.scheduler_step_at == "epoch"
                and self.steps > self.warmup_steps
            ):
                self.scheduler.step()

            if self.epoch % self.validation_freq == 0 or self.epoch == self.epochs:
                self._validate(valid_iter, valid_data, valid_log_idx)

            self._save_checkpoint("latest.ckpt")

            if self.stop:
                self.logger.info(
                    "Training ended since minimum lr %f was reached.",
                    self.learning_rate_min,
                )
                break
        else:
            self.logger.info("Training ended after %3d epochs.", self.epoch)

        self.logger.info(
            "Best validation result at epoch %3d, step %8d: %6.2f %s.",
            self.best_ckpt_epoch,
            self.best_ckpt_steps,
            self.best_ckpt_score,
            self.early_stopping_metric,
        )

        self.tb_writer.close()  # close Tensorboard writer

    def _train_batch(self, batch: Batch, update: bool = True) -> Tensor:
        """
        Train the model on one batch: Compute the loss, make a gradient step.

        :param batch: training batch
        :param update: if False, only store gradient. if True also make update
        :return: normalized translation loss (detached)
        """
        translation_loss = self.model.get_loss_for_batch(
            batch=batch,
            translation_loss_function=self.translation_loss_function,
            translation_loss_weight=self.translation_loss_weight,
        )

        # normalize translation loss
        if self.translation_normalization_mode == "batch":
            txt_normalization_factor = batch.num_seqs
        else:
            txt_normalization_factor = batch.num_txt_tokens

        # division needed since loss.backward sums the gradients until updated
        normalized_translation_loss = translation_loss / (
            txt_normalization_factor * self.batch_multiplier
        )

        # compute gradients
        normalized_translation_loss.backward()

        if update:
            if self.clip_grad_fun is not None:
                # clip gradients (in-place)
                self.clip_grad_fun(params=self.model.parameters())

            self._set_learning_rate()

            # make gradient step
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            # increment step counter
            self.steps += 1

            if (
                self.scheduler is not None
                and self.scheduler_step_at == "step"
                and self.steps > self.warmup_steps
            ):
                self.scheduler.step()

        # increment token counter
        self.total_txt_tokens += batch.num_txt_tokens

        return normalized_translation_loss.detach()

    def _set_learning_rate(self) -> None:
        """
        Set the learning rate of the next update: linear warmup to
        learning_rate, then for "cosine" / "linear" a cosine / linear decay
        that reaches learning_rate_min at the last update of the last epoch.
        Other schedulers take over after the warmup.

        It only depends on the number of updates, so resuming continues it exactly.
        """
        if self.steps < self.warmup_steps:
            lr = self.learning_rate * (self.steps + 1) / self.warmup_steps
        elif self.scheduling in ["cosine", "linear"]:
            progress = min(
                (self.steps - self.warmup_steps)
                / max(self.total_steps - 1 - self.warmup_steps, 1),
                1.0,
            )
            if self.scheduling == "cosine":
                decay = 0.5 * (1 + math.cos(math.pi * progress))
            else:
                decay = 1 - progress
            lr = self.learning_rate_min + (self.learning_rate - self.learning_rate_min) * decay
        else:
            return
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _validate(
        self,
        valid_iter: DataLoader,
        valid_data: SignTranslationDataset,
        valid_log_idx: np.ndarray,
    ) -> None:
        """
        Validate on the entire dev set: log and store the results, keep the
        best checkpoint, step the plateau scheduler.
        """
        valid_start_time = time.time()
        val_res = validate_on_data(
            model=self.model,
            data_iter=valid_iter,
            use_cuda=self.use_cuda,
            translation_loss_function=self.valid_loss_function,
            translation_max_output_length=self.translation_max_output_length,
            translation_beam_size=self.eval_translation_beam_size,
            translation_beam_alpha=self.eval_translation_beam_alpha,
        )
        self.model.train()
        valid_scores = val_res["valid_scores"]

        if self.early_stopping_metric == "translation_loss":
            ckpt_score = val_res["valid_translation_loss"]
        elif self.early_stopping_metric == "ppl":
            ckpt_score = val_res["valid_ppl"]
        else:
            ckpt_score = valid_scores[self.eval_metric]

        new_best = self.is_best(ckpt_score)
        if new_best:
            self.best_ckpt_score = ckpt_score
            self.best_all_ckpt_scores = valid_scores
            self.best_ckpt_epoch = self.epoch
            self.best_ckpt_steps = self.steps
            self.logger.info(
                "Hooray! New best validation result [%s]!", self.early_stopping_metric
            )

        after_warmup = self.steps >= self.warmup_steps
        if (
            self.scheduler is not None
            and self.scheduler_step_at == "validation"
            and after_warmup
        ):
            self.scheduler.step(ckpt_score)

        current_lr = self.optimizer.param_groups[0]["lr"]
        if after_warmup and current_lr < self.learning_rate_min:
            self.stop = True

        if new_best:
            self.logger.info("Saving new best checkpoint.")
            self._save_checkpoint("best.ckpt")

        self.tb_writer.add_scalar(
            "valid/valid_translation_loss", val_res["valid_translation_loss"], self.steps
        )
        self.tb_writer.add_scalar("valid/valid_ppl", val_res["valid_ppl"], self.steps)
        self.tb_writer.add_scalar("valid/chrf", valid_scores["chrf"], self.steps)
        self.tb_writer.add_scalar("valid/rouge", valid_scores["rouge"], self.steps)
        self.tb_writer.add_scalar("valid/bleu", valid_scores["bleu"], self.steps)
        self.tb_writer.add_scalars(
            "valid/bleu_scores", valid_scores["bleu_scores"], self.steps
        )

        # append to validation report
        with open(self.valid_report_file, "a", encoding="utf-8") as opened_file:
            opened_file.write(
                "Epoch: {}\tSteps: {}\tTranslation Loss: {:.5f}\tPPL: {:.5f}\t"
                "Eval Metric: {}\t{}\tLR: {:.8f}\t{}\n".format(
                    self.epoch,
                    self.steps,
                    val_res["valid_translation_loss"],
                    val_res["valid_ppl"],
                    self.eval_metric,
                    format_scores(valid_scores),
                    current_lr,
                    "*" if new_best else "",
                )
            )

        self.logger.info(
            "Validation result at epoch %3d, step %8d: duration: %.4fs\n\t"
            "Translation Beam Size: %d\t"
            "Translation Beam Alpha: %g\n\t"
            "Translation Loss: %4.5f\t"
            "PPL: %4.5f\n\t"
            "Eval Metric: %s\n\t%s",
            self.epoch,
            self.steps,
            time.time() - valid_start_time,
            self.eval_translation_beam_size,
            self.eval_translation_beam_alpha,
            val_res["valid_translation_loss"],
            val_res["valid_ppl"],
            self.eval_metric.upper(),
            format_scores(valid_scores),
        )

        # log some examples
        names = valid_data.names
        self.logger.info("Logging Translation Outputs")
        self.logger.info("=" * 120)
        for i in valid_log_idx:
            self.logger.info("Logging Sequence: %s", names[i])
            self.logger.info("\tReference : %s", val_res["txt_ref"][i])
            self.logger.info("\tHypothesis: %s", val_res["txt_hyp"][i])
            self.logger.info("=" * 120)

        # store validation set outputs and references
        txt_dir = os.path.join(self.model_dir, "txt")
        os.makedirs(txt_dir, exist_ok=True)
        write_outputs(
            os.path.join(txt_dir, "{}.dev.hyp.txt".format(self.epoch)), names, val_res["txt_hyp"]
        )
        write_outputs(
            os.path.join(self.model_dir, "references.dev.txt"), names, val_res["txt_ref"]
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


def train(cfg_file: str, resume: bool = False) -> None:
    """
    Main training function. After training, also test on test data if given.

    :param cfg_file: path to configuration yaml file
    :param resume: continue the training in model_dir from latest.ckpt
    """
    cfg = load_config(cfg_file)

    # set the random seed
    set_seed(seed=cfg["training"].get("random_seed", 42))

    data, txt_vocab = load_data(data_cfg=cfg["data"], splits=["train", "val"])
    for split, dataset in data.items():
        if len(dataset) == 0:
            raise ValueError(
                "No {} sentences found: are there manifest items with "
                "\"split\": \"{}\" and features for them?".format(split, split)
            )

    # build model and load parameters into it
    model = build_model(
        cfg=cfg["model"],
        txt_vocab=txt_vocab,
        sgn_dim=cfg["data"]["feature_size"],
        multimodal=cfg["data"].get("multimodal", False),
    )

    # for training management, e.g. early stopping and model selection
    trainer = TrainManager(model=model, config=cfg, resume=resume)

    # store copy of the training config in model dir
    shutil.copy2(cfg_file, os.path.join(trainer.model_dir, "config.yaml"))

    # log all entries of config
    log_cfg(cfg, trainer.logger)

    log_data_info(data=data, txt_vocab=txt_vocab, logging_function=trainer.logger.info)

    trainer.logger.info(str(model))

    # store the vocab
    txt_vocab.to_file(os.path.join(trainer.model_dir, "txt.vocab"))

    # train the model
    trainer.train_and_validate(train_data=data["train"], valid_data=data["val"])
    # Delete to speed things up as we don't need training data anymore
    del data

    # predict with the best model on validation and test
    ckpt = os.path.join(trainer.model_dir, "best.ckpt")
    output_name = "best.EP_{:04d}".format(trainer.best_ckpt_epoch)
    output_path = os.path.join(trainer.model_dir, output_name)
    logger = trainer.logger
    del trainer
    test(cfg_file, ckpt=ckpt, output_path=output_path, logger=logger)
