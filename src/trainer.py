import logging
import multiprocessing as mp
import os
from typing import Optional

import datasets
import huggingface_hub
import torch
import torch.nn as nn
import torch.optim as optim
import transformers

import wandb
from torch.utils.data import DataLoader

from src.constants import HF_TOKEN

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        config,
        model: nn.Module,
        tokenizer: transformers.PreTrainedTokenizer,
        train_dataset: datasets.Dataset,
        valid_dataset: datasets.Dataset,
        data_collator,
        callbacks,
    ):
        self.device = "cuda"
        self.config = config
        self.model = model.to(self.device)
        self.optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
        self.criterion = nn.CrossEntropyLoss(reduction="mean")
        self.callbacks = callbacks
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.data_collator = data_collator
        self.features = ["input_ids", "attention_mask", "label"]

        self.run = None  # wandb run
        self.global_step = 0

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
        return dataloader

    def train(self):
        logging.info("Getting dataloaders")
        train_loader = self.get_dataloaders(
            self.train_dataset, self.config.per_device_train_batch_size
        )
        valid_loader = self.get_dataloaders(
            self.valid_dataset, self.config.per_device_eval_batch_size
        )

        self.run = wandb.init(project="TemporalClassifier")
        self.run.watch(self.model)
        self.run.config.update(self.config)

        logging.info("Training")
        for epoch in range(self.config.num_train_epochs):
            train_loss, train_acc = self.train_step(train_loader)
            valid_loss, valid_acc = self.valid_step(valid_loader)
            self.run.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "train/acc": train_acc,
                    "valid/loss": valid_loss,
                    "valid/acc": valid_acc,
                },
                step=self.global_step,
            )
            print(
                f"Epoch {epoch} - Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} - Valid Loss: {valid_loss:.4f} - Valid Acc: {valid_acc:.4f}"
            )

    def train_step(self, train_loader: DataLoader):
        self.model.train()

        total_correct = 0
        total_examples = 0
        total_loss = 0
        n_steps = len(train_loader)
        for batch in train_loader:
            self.optimizer.zero_grad()
            batch = {k: v.to(self.device) for k, v in batch.items()}
            logits = self.model.forward(**batch)
            loss = self.criterion(logits, batch["labels"])
            loss.backward()

            if self.config.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
            self.optimizer.step()

            # metrics
            total_loss += loss.item()
            total_correct += (
                (logits.argmax(dim=1) == batch["labels"].argmax(dim=1)).sum().item()
            )
            total_examples += batch["labels"].shape[0]

            # logging
            self.run.log({"train/loss": loss.item()}, step=self.global_step)
            print(
                f"Step {self.global_step} / {n_steps} - Loss: {loss.item():.4f} - Acc: {total_correct / total_examples:.4f}"
            )
            self.global_step += 1

        loss = total_loss / len(train_loader)
        acc = total_correct / total_examples
        return loss, acc

    def valid_step(self, valid_loader: DataLoader):
        self.model.eval()
        total_correct = 0
        total_examples = 0
        total_loss = 0
        with torch.no_grad():
            for batch in valid_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                logits = self.model.forward(**batch)
                total_correct += (
                    (logits.argmax(dim=1) == batch["labels"].argmax(dim=1)).sum().item()
                )
                loss = self.criterion(logits, batch["labels"])
                total_loss += loss.item()
                total_examples += batch["labels"].shape[0]

        loss = total_loss / len(valid_loader)
        acc = total_correct / total_examples
        return loss, acc

    def push_to_hub(
        self,
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
        if not huggingface_hub.repo_exists(self.config.hub_model_id):
            huggingface_hub.create_repo(self.config.hub_model_id, token=HF_TOKEN)

        return huggingface_hub.upload_folder(
            repo_id=self.config.hub_model_id,
            folder_path=self.config.output_dir,
            commit_message=commit_message,
            token=HF_TOKEN,
            run_as_future=not blocking,
            ignore_patterns=[
                "_*",
                f"{transformers.trainer_utils.PREFIX_CHECKPOINT_DIR}-*",
            ],
        )

    def save_model(self):
        """
        Will save the model, so you can reload it using `from_pretrained()`.

        Will only save from the main process.
        """

        # If we are executing this function, we are the process zero, so we don't check for that.
        os.makedirs(self.config.output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {self.config.output_dir}")

        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if isinstance(self.model, transformers.PreTrainedModel):
            state_dict = self.model.state_dict()

            self.model.save_pretrained(
                self.config.output_dir,
                state_dict=state_dict,
                safe_serialization=self.config.save_safetensors,
            )
        else:
            logger.info(
                "Trainer.model is not a `PreTrainedModel`, only saving its state dict."
            )

            torch.save(
                state_dict,
                os.path.join(self.config.output_dir, transformers.utils.WEIGHTS_NAME),
            )

        # Push to the Hub when `save_model` is called by the user.
        if self.config.push_to_hub:
            self.push_to_hub(commit_message="Model save")
