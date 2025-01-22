import logging
import multiprocessing as mp
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import torch
import transformers
from fire import Fire
from omegaconf import OmegaConf

from src.base import RELATIONS2ID
from src.constants import CONFIGS_DIR, HF_TOKEN, NEW_TOKENS
from src.data import augment_dataset, load_dataset
from src.model.classifier import ContextClassifier
from src.trainer import Trainer
from transformers.models.llama.configuration_llama import LlamaConfig
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


@dataclass
class TrainingArguments:
    output_dir: str = field(
        default="models",
        metadata={
            "help": "The output directory where the model predictions and checkpoints will be written."
        },
    )
    per_device_train_batch_size: int = field(
        default=8,
        metadata={"help": "The batch size for training."},
    )
    per_device_eval_batch_size: int = field(
        default=8,
        metadata={"help": "The batch size for evaluation."},
    )
    num_train_epochs: int = field(
        default=30,
        metadata={"help": "The number of epochs to train the model."},
    )
    learning_rate: float = field(
        default=1e-5,
        metadata={"help": "The learning rate for the optimizer."},
    )
    max_grad_norm: float = field(
        default=1.0,
        metadata={"help": "The maximum gradient norm for gradient clipping."},
    )
    load_best_model_at_end: bool = field(
        default=True,
        metadata={"help": "Whether to load the best model at the end of training."},
    )
    seed: int = field(
        default=42,
        metadata={"help": "The random seed to use for training."},
    )
    bf16: bool = field(
        default=True,
        metadata={"help": "Whether to use bfloat16 for training."},
    )
    push_to_hub: bool = field(
        default=True,
        metadata={"help": "Whether to push the model to the hub."},
    )
    hub_model_id: str = field(
        default="hugosousa/debug",
        metadata={"help": "The model id to use for the hub."},
    )
    label_smoothing_factor: float = field(
        default=0.01,
        metadata={"help": "The factor for label smoothing."},
    )
    early_stopping_patience: int = field(
        default=3,
        metadata={
            "help": "Number of epochs to wait for no improvement in validation loss before stopping."
        },
    )
    init_bias: bool = field(
        default=True,
        metadata={"help": "Whether to initialize the bias of the last layer."},
    )
    torch_compile: bool = field(
        default=False,
        metadata={"help": "Whether to use torch compile."},
    )
    save_safetensors: bool = field(
        default=True,
        metadata={"help": "Whether to save the model in safetensors format."},
    )
    lr_scheduler: Optional[dict] = field(
        default=None,
        metadata={"help": "The learning rate scheduler to use."},
    )
    freeze_backbone: bool = field(
        default=False,
        metadata={"help": "Whether to freeze the backbone."},
    )


def main(
    config_file: str = CONFIGS_DIR / "classifier" / "debug.yaml",
):
    config = OmegaConf.load(config_file)

    config.model["token"] = HF_TOKEN
    model_args = ModelArguments(**config.model)

    config.data["preprocessing_num_workers"] = mp.cpu_count()
    data_args = DataTrainingArguments(**config.data)

    training_args = TrainingArguments(**config.trainer)

    send_example_telemetry("run_classification", model_args, data_args)

    # Setup logging
    log_level = logging.INFO
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=log_level,
    )

    # Log on each process the small summary:
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Model arguments: {model_args}")

    # Set seed before initializing model.
    transformers.set_seed(training_args.seed)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

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

    relations = set(raw_datasets["train"]["label"])
    num_labels = len(relations)
    label2id = {r: RELATIONS2ID[r] for r in relations}
    id2label = {RELATIONS2ID[r]: r for r in relations}

    # Load pretrained model and tokenizer
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    tokenizer = transformers.AutoTokenizer.from_pretrained(
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
        label2id=label2id,
        id2label=id2label,
    )

    logger.info(f"Adding new tokens to the model: {new_token_ids}")
    model_config.tokens_to_encode_ids = new_token_ids

    model = ContextClassifier.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=model_config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
        torch_dtype=torch.bfloat16,
    )

    model.config.pad_token_id = model.config.eos_token_id

    # Add new tokens to the model
    model.resize_token_embeddings(
        len(tokenizer), pad_to_multiple_of=8, mean_resizing=True
    )

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

    def preprocess_function(examples):
        result = tokenizer(examples["text"], padding=padding)
        if label2id is not None and "label" in examples:
            result["label"] = [label2id[label] for label in examples["label"]]
        return result

    # Train data
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

    # Eval data
    eval_dataset = raw_datasets["valid"]
    if data_args.max_eval_samples is not None:
        max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
        eval_dataset = eval_dataset.select(range(max_eval_samples))
    if data_args.augment:
        logger.info("Augmenting the evaluation dataset")
        eval_dataset = augment_dataset(eval_dataset)
    raw_datasets["valid"] = eval_dataset

    # Running the preprocessing pipeline on all the datasets
    raw_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=not data_args.overwrite_cache,
        desc="Running tokenizer on dataset",
    )
    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["valid"]

    logger.info(
        f"Dropping rows in the training dataset with more than {max_seq_length} tokens"
    )
    n_train = len(train_dataset)
    train_dataset = train_dataset.filter(
        lambda x: len(x["input_ids"]) <= max_seq_length
    )
    n_dropped_train = n_train - len(train_dataset)
    logger.info(f"Dropped {n_dropped_train} rows from train dataset")

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = transformers.default_data_collator
    elif training_args.bf16:
        data_collator = transformers.DataCollatorWithPadding(
            tokenizer, pad_to_multiple_of=8
        )
    else:
        data_collator = None

    # Initialize our Trainer
    trainer = Trainer(
        config=training_args,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        valid_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Training
    trainer.train()


if __name__ == "__main__":
    Fire(main)
