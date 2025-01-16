import json
import logging
import multiprocessing as mp
import os
import random
import sys
from dataclasses import dataclass, field
from typing import List, Optional

import datasets
import transformers
from fire import Fire
from omegaconf import OmegaConf
from sklearn.metrics import classification_report

from src.base import RELATIONS2ID
from src.constants import CONFIGS_DIR, HF_TOKEN, NEW_TOKENS
from src.data import augment_dataset, load_dataset
from src.model.classifier import Classifier
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    default_data_collator,
    EarlyStoppingCallback,
    EvalPrediction,
    set_seed,
    Trainer,
    TrainingArguments,
)

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import send_example_telemetry

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    train_split: str = field(
        default="train",
        metadata={"help": "The split to use for the training dataset."},
    )
    valid_split: str = field(
        default="valid",
        metadata={"help": "The split to use for the validation dataset."},
    )
    test_split: str = field(
        default="test",
        metadata={"help": "The split to use for the test dataset."},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached preprocessed datasets or not."},
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    shuffle_train_dataset: bool = field(
        default=False, metadata={"help": "Whether to shuffle the train dataset or not."}
    )
    shuffle_seed: int = field(
        default=42,
        metadata={
            "help": "Random seed that will be used to shuffle the train dataset."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    metric_name: Optional[str] = field(
        default=None, metadata={"help": "The metric to use for evaluation."}
    )
    augment: bool = field(
        default=False, metadata={"help": "Whether to augment the dataset or not."}
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )

    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={
            "help": "Will enable to load a pretrained model whose head dimensions are different."
        },
    )


def main(
    config_file: str = CONFIGS_DIR / "classifier" / "debug.yaml",
):
    config = OmegaConf.load(config_file)

    if "debug" in config:
        debug = config.debug
    else:
        debug = False

    config.model["token"] = HF_TOKEN
    model_args = ModelArguments(**config.model)

    config.data["preprocessing_num_workers"] = mp.cpu_count()
    data_args = DataTrainingArguments(**config.data)

    training_args = TrainingArguments(**config.trainer)

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_classification", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Downloading and loading a dataset from the hub.
    trainset = load_dataset(
        data_args.dataset_name,
        split=data_args.train_split,
        config=data_args.dataset_config_name,
    )
    validset = load_dataset(
        data_args.dataset_name,
        split=data_args.valid_split,
        config=data_args.dataset_config_name,
    )

    raw_datasets = datasets.DatasetDict(
        {
            "train": trainset,
            "valid": validset,
        }
    )

    # Print some info about the dataset
    logger.info(f"Dataset loaded: {raw_datasets}")
    logger.info(raw_datasets)

    relations = set(raw_datasets["train"]["label"])
    num_labels = len(relations)
    label2id = {r: RELATIONS2ID[r] for r in relations}
    id2label = {RELATIONS2ID[r]: r for r in relations}

    # Load pretrained model and tokenizer
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Add new tokens to the tokenizer
    tokenizer.add_tokens(NEW_TOKENS)
    new_token_ids = tokenizer.convert_tokens_to_ids(NEW_TOKENS)

    model_config = LlamaConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task="text-classification",
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        label2id=label2id,
        id2label=id2label,
    )

    logger.info("setting problem type to multi label classification")
    model_config.problem_type = "multi_label_classification"

    logger.info(f"Adding new tokens to the model: {new_token_ids}")
    model_config.tokens_to_encode_ids = new_token_ids

    model = Classifier.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=model_config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )

    model.config.pad_token_id = model.config.eos_token_id

    # Add new tokens to the model
    model.resize_token_embeddings(len(tokenizer))

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the "
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    if "label_smoothing_factor" in config.other:
        factor = config.other.label_smoothing_factor
    else:
        factor = 0.0

    def multi_labels_to_ids(labels: List[str]) -> List[float]:
        ids = [factor / num_labels] * num_labels
        for label in labels:
            ids[label2id[label]] = 1.0 - factor / num_labels
        return ids

    def preprocess_function(examples):
        result = tokenizer(examples["text"], padding=padding)
        if label2id is not None and "label" in examples:
            result["label"] = [
                multi_labels_to_ids(labels) for labels in examples["label"]
            ]
        return result

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset.")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        if data_args.augment:
            logger.info("Augmenting the training dataset")
            train_dataset = augment_dataset(train_dataset)
        if data_args.shuffle_train_dataset:
            logger.info("Shuffling the training dataset")
            train_dataset = train_dataset.shuffle(seed=data_args.shuffle_seed)
        raw_datasets["train"] = train_dataset

    if training_args.do_eval:
        eval_dataset = raw_datasets["valid"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        if data_args.augment:
            logger.info("Augmenting the evaluation dataset")
            eval_dataset = augment_dataset(eval_dataset)
        raw_datasets["valid"] = eval_dataset

    # Running the preprocessing pipeline on all the datasets
    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
        train_dataset = raw_datasets["train"]
        eval_dataset = raw_datasets["valid"]

        logger.info(f"Dropping rows with more than {max_seq_length} tokens")
        n_train = len(train_dataset)
        train_dataset = train_dataset.filter(
            lambda x: len(x["input_ids"]) <= max_seq_length
        )
        n_dropped_train = n_train - len(train_dataset)
        logger.info(f"Dropped {n_dropped_train} rows from train dataset")

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    logger.info(
        "Using multilabel F1 for multi-label classification task, you can use --metric_name to overwrite."
    )

    def compute_metrics(p: EvalPrediction):
        logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        y_preds = logits.argmax(axis=1)
        preds = [id2label[int(y_pred)] for y_pred in y_preds.tolist()]

        y_labels = p.label_ids.argmax(axis=1)
        labels = [id2label[int(y_label)] for y_label in y_labels.tolist()]

        result = classification_report(
            y_true=labels,
            y_pred=preds,
            output_dict=True,
            zero_division=0.0,
            labels=list(label2id.keys()),
        )

        formatted_result = {}
        for metric, value in result.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    formatted_result[f"{metric}_{k}".replace(" ", "_")] = v
            else:
                formatted_result[metric] = value

        logger.info(json.dumps(formatted_result, indent=4))

        return formatted_result

    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16 or training_args.bf16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    if "early_stopping_patience" in config.other:
        callbacks = [
            EarlyStoppingCallback(
                early_stopping_patience=config.other.early_stopping_patience
            )
        ]
    else:
        callbacks = []

    # compute_loss_func = transformers.loss.loss_utils.ForSequenceClassificationLoss
    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
        # compute_loss_func=compute_loss_func,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if debug:
            checkpoint = None
        elif training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        trainer.save_model()  # Saves the tokenizer too for easy upload
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        max_eval_samples = (
            data_args.max_eval_samples
            if data_args.max_eval_samples is not None
            else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "text-classification",
    }

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    Fire(main)
