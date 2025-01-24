import collections
import logging
import multiprocessing as mp
import pathlib
import tempfile
from typing import Optional

import datasets
import huggingface_hub
import ray
import sklearn
import torch
import torch.nn as nn
import torch.optim as optim
import transformers

import wandb
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.constants import HF_TOKEN
from src.metrics import compute_metrics, compute_metrics_with_vague
from src.utils import generate_id

logger = logging.getLogger(__name__)


class EarlyStopping:
    def __init__(self, patience: int, greater_is_better: bool):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.greater_is_better = greater_is_better

    def __call__(
        self,
        score: float,
    ):
        if self.best_score is None:
            self.best_score = score

        elif self.greater_is_better:
            if score > self.best_score:
                self.counter = 0
                self.best_score = score
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
        else:
            if score < self.best_score:
                self.counter = 0
                self.best_score = score
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True


class Trainer:
    def __init__(
        self,
        config,
        model: nn.Module,
        tokenizer: transformers.PreTrainedTokenizer,
        train_dataset: datasets.Dataset,
        valid_dataset: datasets.Dataset,
        data_collator,
    ):
        self.config = config
        self.output_dir = pathlib.Path(config.output_dir)

        self.accelerator = Accelerator()

        if config.init_bias:
            logger.info("Initializing bias of the last layer")
            lids = train_dataset["label"]
            counter = collections.Counter(lids)
            fqs = [counter[i] / len(train_dataset) for i in range(len(counter))]
            init_bias = torch.log(torch.tensor(fqs))
            init_bias = init_bias.to(model.score.bias.dtype)
            model.score.bias.data = init_bias

        if config.freeze_backbone:
            logger.info(
                "Freezing backbone. That is, only the new embeddings and the score layer are trainable."
            )
            model.model.requires_grad_(False)
            # activate gradient for the embeddings of the special tokens
            model.model.embed_tokens.weight[model.tokens_to_encode_ids].requires_grad_(
                True
            )

        # unique labels sorted by id
        self.unique_labels = sorted(
            list(model.config.id2label.values()),
            key=lambda x: model.config.label2id[x],
        )

        # Define the metrics function
        if "-" in self.unique_labels:
            self._metrics_func = compute_metrics_with_vague
        else:
            self._metrics_func = compute_metrics

        self.model = model

        if self.config.torch_compile:
            logger.info("Compiling model")
            self.model = torch.compile(self.model)

        self.optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)

        self.steps_per_epoch = (
            len(train_dataset) // self.config.per_device_train_batch_size
        )
        self.total_train_steps = self.config.num_train_epochs * self.steps_per_epoch
        self.warmup_steps = int(
            self.total_train_steps * self.config.lr_scheduler.warmup_steps_pct
        )
        logger.info(f"Total train steps: {self.total_train_steps}")
        logger.info(f"Warmup steps: {self.warmup_steps}")
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=self.config.lr_scheduler.warmup_factor,
            total_iters=self.warmup_steps,
        )
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.total_train_steps - self.warmup_steps,
            T_mult=1,
        )
        self.scheduler = optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.warmup_steps],
        )

        self.model, self.optimizer, self.scheduler = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.scheduler,
        )

        self.criterion = nn.CrossEntropyLoss(
            reduction="mean", label_smoothing=config.label_smoothing_factor
        )
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.data_collator = data_collator
        self.features = ["input_ids", "attention_mask", "label"]

        self.run = None  # wandb run
        self.global_step = 0
        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience, greater_is_better=True
        )

        self.log_every = max(self.steps_per_epoch // 10, 1)

    def get_dataloaders(self, dataset: datasets.Dataset, batch_size: int):
        dataset = dataset.select_columns(self.features)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=self.data_collator,
            shuffle=True,
            pin_memory=True,
            num_workers=mp.cpu_count(),
            drop_last=True,
        )
        return self.accelerator.prepare(dataloader)

    def train(self):
        logging.info("Getting dataloaders")
        run_id = generate_id()
        train_loader = self.get_dataloaders(
            self.train_dataset, self.config.per_device_train_batch_size
        )
        valid_loader = self.get_dataloaders(
            self.valid_dataset, self.config.per_device_eval_batch_size
        )

        self.run = wandb.init(
            project="TemporalClassifier", name=f"{self.output_dir.stem}-{run_id}"
        )
        self.run.watch(self.model)
        self.run.config.update(self.config)

        logging.info("Training")
        best_valid_f1 = float("-inf")
        for epoch in range(self.config.num_train_epochs):
            train_metrics = self.train_step(train_loader)
            valid_metrics = self.valid_step(valid_loader)

            self.run.log(
                {
                    "epoch": epoch,
                    "train": train_metrics,
                    "valid": valid_metrics,
                },
                step=self.global_step,
            )

            if valid_metrics["f1-score"] > best_valid_f1:
                best_valid_f1 = valid_metrics["f1-score"]
                logger.info(
                    f"Saving model. New best valid f1-score: {best_valid_f1:.4f}"
                )
                self.save_model(output_dir=self.output_dir / run_id)
                self.save_tokenizer(self.output_dir / run_id)
                if self.config.push_to_hub:
                    self.push_to_hub(
                        repo_id=f"{self.config.hub_model_id}-{run_id}",
                        folder_path=self.output_dir / run_id,
                        commit_message=f"Epoch {epoch} Best valid f1-score: {best_valid_f1:.4f}",
                        blocking=True,
                    )

            self.early_stopping(valid_metrics["f1-score"])
            if self.early_stopping.early_stop:
                logger.info(
                    "Early stopping since no improvement in validation f1-score"
                )
                break

            if self.config.hp_search:
                with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                    temp_checkpoint_dir = pathlib.Path(temp_checkpoint_dir)

                    torch.save(
                        self.model.state_dict(),
                        temp_checkpoint_dir / "model.pth",
                    )
                    checkpoint = ray.train.Checkpoint.from_directory(
                        temp_checkpoint_dir
                    )
                    ray.train.report(
                        metrics=valid_metrics,
                        checkpoint=checkpoint,
                    )
                    ray.train.report(
                        metrics={"f1-score": valid_metrics["f1-score"]},
                        checkpoint=checkpoint,
                    )

            logger.info(
                f"Epoch {epoch}\n"
                f"\tTrain Loss: {train_metrics['loss']:.4f}\n"
                f"\tTrain Acc: {train_metrics['accuracy']:.4f}\n"
                f"\tTrain F1-score: {train_metrics['f1-score']:.4f}\n"
                f"\tValid Loss: {valid_metrics['loss']:.4f}\n"
                f"\tValid Acc: {valid_metrics['accuracy']:.4f}\n"
                f"\tValid F1-score: {valid_metrics['f1-score']:.4f}"
            )

        if self.config.load_best_model_at_end:
            logger.info("Loading best model")
            self.load_model(self.output_dir / run_id)

        if self.config.push_to_hub:
            self.save_tokenizer(self.output_dir / run_id)
            self.push_to_hub(
                repo_id=f"{self.config.hub_model_id}-{run_id}",
                folder_path=self.output_dir / run_id,
                commit_message="End of training",
                blocking=True,
            )

    def train_step(self, train_loader: DataLoader):
        self.model.train()
        pb = tqdm(
            range(len(train_loader)), disable=not self.accelerator.is_local_main_process
        )

        total_correct = 0
        total_examples = 0
        total_loss = 0
        y_preds, y_trues = [], []
        epoch_frac = 1 / len(train_loader)
        for batch in train_loader:
            self.optimizer.zero_grad()
            logits = self.model.forward(**batch).logits
            loss = self.criterion(logits, batch["labels"])

            self.accelerator.backward(loss)

            if self.config.max_grad_norm is not None:
                self.accelerator.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
            self.optimizer.step()
            self.scheduler.step()

            # metrics
            total_loss += loss.item()
            total_correct += (logits.argmax(dim=1) == batch["labels"]).sum().item()
            total_examples += batch["labels"].shape[0]
            y_preds += logits.argmax(dim=1).tolist()
            y_trues += batch["labels"].tolist()

            lr = self.optimizer.param_groups[0]["lr"]
            if (self.global_step + 1) % self.log_every == 0:
                metrics = self.compute_metrics(y_preds, y_trues)
                metrics["loss"] = loss.item()

                # logging
                self.run.log(
                    {
                        "train": metrics,
                        "lr": lr,
                        "epoch": epoch_frac * self.global_step,
                    },
                    step=self.global_step,
                )
            else:
                self.run.log(
                    {
                        "train": {"loss": loss.item()},
                        "lr": lr,
                        "epoch": epoch_frac * self.global_step,
                    },
                    step=self.global_step,
                )
            pb.set_description(
                f"Loss: {loss.item():.4f} - Acc: {total_correct / total_examples:.4f}"
            )
            pb.update(1)
            self.global_step += 1

        metrics = self.compute_metrics(y_preds, y_trues)
        metrics["loss"] = total_loss / len(train_loader)
        return metrics

    def valid_step(self, valid_loader: DataLoader):
        self.model.eval()
        total_loss = 0
        y_preds, y_trues = [], []
        with torch.no_grad():
            for batch in valid_loader:
                logits = self.model.forward(**batch).logits
                loss = self.criterion(logits, batch["labels"])
                total_loss += loss.item()
                y_preds += logits.argmax(dim=1).tolist()
                y_trues += batch["labels"].tolist()

        metrics = self.compute_metrics(y_preds, y_trues)
        metrics = {f"{k}": v for k, v in metrics.items()}
        metrics["loss"] = total_loss / len(valid_loader)
        return metrics

    @staticmethod
    def push_to_hub(
        repo_id: str,
        folder_path: pathlib.Path,
        commit_message: Optional[str] = "End of training",
        blocking: bool = True,
    ) -> str:
        """
        Upload `self.model` and `self.processing_class` to the 🤗 model hub on the repo `self.args.hub_model_id`.

        Parameters:
            commit_message (`str`, *optional*, defaults to `"End of training"`):
                Message to commit while pushing.
            blocking (`bool`, *optional*, defaults to `True`):
                Whether the function should return only when the `git push` has finished.

        Returns:
            The URL of the repository where the model was pushed if `blocking=False`, or a `Future` object tracking the
            progress of the commit if `blocking=True`.
        """

        # Create the repo if it doesn't exist
        if not huggingface_hub.repo_exists(repo_id):
            huggingface_hub.create_repo(repo_id, token=HF_TOKEN)

        return huggingface_hub.upload_folder(
            repo_id=repo_id,
            folder_path=folder_path,
            commit_message=commit_message,
            token=HF_TOKEN,
            run_as_future=not blocking,
            ignore_patterns=[
                "_*",
                f"{transformers.trainer_utils.PREFIX_CHECKPOINT_DIR}-*",
            ],
        )

    def save_model(
        self,
        output_dir: pathlib.Path,
    ):
        """
        Will save the model, so you can reload it using `from_pretrained()`.

        Will only save from the main process.
        """

        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if hasattr(self.model, "save_pretrained"):
            state_dict = self.model.state_dict()
            self.model.save_pretrained(
                output_dir,
                state_dict=state_dict,
                safe_serialization=self.config.save_safetensors,
            )
        else:
            logger.info(
                "Trainer.model is not a `PreTrainedModel`, only saving its state dict."
            )

            torch.save(
                state_dict,
                output_dir / transformers.utils.WEIGHTS_NAME,
            )

    def save_tokenizer(self, output_dir: pathlib.Path):
        self.tokenizer.save_pretrained(output_dir)

    def load_model(self, output_dir: pathlib.Path):
        if hasattr(self.model, "from_pretrained"):
            model = self.model.from_pretrained(output_dir)
        else:
            model.load_state_dict(
                torch.load(output_dir / transformers.utils.WEIGHTS_NAME)
            )
        return model

    def compute_metrics(self, y_pred: list, y_true: list):
        preds = [self.model.config.id2label[int(pred)] for pred in y_pred]
        labels = [self.model.config.id2label[int(label)] for label in y_true]

        metrics = self._metrics_func(
            y_true=labels, y_pred=preds, labels=self.unique_labels
        )

        per_label = sklearn.metrics.classification_report(
            y_true=labels,
            y_pred=preds,
            output_dict=True,
            zero_division=0.0,
            labels=self.unique_labels,
        )
        per_label.pop("accuracy", None)
        per_label.pop("micro avg", None)
        per_label.pop("macro avg", None)
        per_label.pop("weighted avg", None)
        metrics["per_label"] = per_label
        return metrics
