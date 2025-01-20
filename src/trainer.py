import logging

import multiprocessing as mp

import datasets
import torch
import torch.nn as nn
import torch.optim as optim
import transformers
import wandb
from torch.utils.data import DataLoader


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

    def train(self, checkpoint=None):
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
