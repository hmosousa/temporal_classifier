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
        self.config = config
        self.model = model
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
            valid_loss = self.valid_step(valid_loader)
            self.run.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "train/acc": train_acc,
                    "valid/loss": valid_loss,
                },
                step=self.global_step,
            )

    def train_step(self, train_loader: DataLoader):
        self.model.train()

        total_correct = 0
        total_loss = 0
        for batch in train_loader:
            self.optimizer.zero_grad()
            outputs = self.model.forward(**batch)
            logits = outputs.logits
            loss = self.criterion(logits, batch["labels"])
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            total_correct += (logits.argmax(dim=1) == batch["labels"]).sum().item()
            self.run.log({"train/loss": loss.item()}, step=self.global_step)
            self.global_step += 1

        loss = total_loss / len(train_loader)
        acc = total_correct / (
            len(train_loader) * self.config.per_device_train_batch_size
        )
        return loss, acc

    def valid_step(self, valid_loader: DataLoader):
        self.model.eval()
        total_correct = 0
        total_loss = 0
        with torch.no_grad():
            for batch in valid_loader:
                outputs = self.model.forward(batch)
                total_correct += (outputs.argmax(dim=1) == batch["label"]).sum().item()
                loss = self.criterion(outputs, batch["label"])
                total_loss += loss.item()
        loss = total_loss / len(valid_loader)
        acc = total_correct / (
            len(valid_loader) * self.config.per_device_eval_batch_size
        )
        return loss, acc
