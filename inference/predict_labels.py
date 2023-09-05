#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Predict the lables for EUINACTION tasks on the crawled data (CAP, EU/non-EU).
The script predicts depending on the inputs the lables either for the CAP or the EU/non-EU task for a single dataset.
"""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from datasets.features import Features
from datasets import Value
from typing import Optional
import pathlib

import numpy as np
from datasets import load_dataset
import torch
from scipy.special import softmax

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process

logger = logging.getLogger(__name__)
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ['RUST_BACKTRACE'] = 'full'

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our models for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on"},
    )
    country: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the country we do predictions for"},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this "
            "value if set."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    def __post_init__(self):
        if self.train_file is None:
            raise ValueError("Need a training file.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which models/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained models or models identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific models version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing models.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.

    # Get the test dataset: you can provide your own CSV/JSON test file (see below)
    # when you use `do_predict` without specifying a GLUE benchmark task.
    if training_args.do_predict:
        if data_args.test_file is not None:
            data_files = {"test": data_args.test_file}
        else:
            raise ValueError("Need a test file for `do_predict`.")
    else:
        raise ValueError("This script only runs on do_predict")

    for key in data_files.keys():
        logger.info(f"load a local file for {key}: {data_files[key]}")

    # Define where to cache datasets
    datasets_cache_d = pathlib.Path(__file__).absolute().parent.joinpath(".cache", "huggingface", "datasets")

    # Define data types
    features = Features({"text": Value("string")})

    if data_args.test_file.endswith(".csv"):
        # Loading a dataset from local csv files
        datasets = load_dataset("csv", data_files=data_files, download_mode="force_redownload", cache_dir=datasets_cache_d, features=features, encoding='latin-1')
        # Used for debugging only
        # datasets = load_dataset("csv", data_files=data_files, cache_dir=datasets_cache_d)
    else:
        # Loading a dataset from local json files
        datasets = load_dataset("json", data_files=data_files, download_mode="force_redownload", cache_dir=datasets_cache_d, features=features)
        # Used for debugging only
        # datasets = load_dataset("json", data_files=data_files, cache_dir=datasets_cache_d)

    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained models and tokenizer
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download models & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config,
    )

    # Load train data set to get the actual labels
    train_dataset = load_dataset("csv", data_files={"train": data_args.train_file}, download_mode="force_redownload", cache_dir=datasets_cache_d)
    # Used for debugging only
    # train_dataset = load_dataset("csv", data_files={"train": data_args.train_file}, cache_dir=datasets_cache_d)

    label_list = train_dataset["train"].unique("label")
    label_list.sort()  # Let's sort it for determinism
    del train_dataset

    # Preprocessing the datasets
    # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
    sentence1_key, sentence2_key = 'text', None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warn(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"models ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        return result

    datasets = datasets.map(preprocess_function, batched=True, load_from_cache_file=not data_args.overwrite_cache)

    if training_args.do_predict or data_args.test_file is not None:
        if "test" not in datasets:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = datasets["test"]
        if data_args.max_test_samples is not None:
            test_dataset = test_dataset.select(range(data_args.max_test_samples))

    # Log a few random samples from the training set:
    for index in random.sample(range(len(test_dataset)), 3):
        logger.info(f"Sample {index} of the test set: {test_dataset[index]}.")

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    if training_args.do_predict:
        logger.info("*** Test ***")

        task = data_args.task_name
        country = data_args.country

        predictions = trainer.predict(test_dataset=test_dataset).predictions
        # Get the probabilities of the predictions#
        if task == 'cap':
            predictions_probabilities = softmax(predictions, axis=1)
            predictions_probabilities.sort(axis=1)
            predictions_probabilities = predictions_probabilities[:, -2:]
            predictions = predictions.argsort(axis=1)[:, -2:]
        else:
            prediction_probabilities = softmax(predictions, axis=1)
            predictions = np.argmax(predictions, axis=1)

        output_test_file = os.path.join(training_args.output_dir, f"test_results_{task}_{country}.txt")
        if trainer.is_world_process_zero():
            with open(output_test_file, "w") as writer:
                logger.info(f"***** Test results {task} *****")
                if task == 'cap':
                    writer.write("index\tprediction\tprobability\tprediction2\tprobability_2\n")
                else:
                    writer.write("index\tprediction\tprobs\n")
                for index, label in enumerate(predictions):
                    if task == 'cap':
                        label_2 = label[0]
                        label_1 = label[1]
                        probability_2 = round(predictions_probabilities[index][0], 3)
                        probability_1 = round(predictions_probabilities[index][1], 3)
                        # label_2 is the second prediction
                        # label_1 is the first prediction
                        label_2 = label_list[label_2]
                        label_1 = label_list[label_1]
                        writer.write(f"{index}\t{label_1}\t{probability_1}\t{label_2}\t{probability_2}\n")
                    else:
                        # label_2 is the second prediction
                        # label_1 is the first prediction
                        label = label_list[label]
                        probability = round(prediction_probabilities[index][label],3)
                        writer.write(f"{index}\t{label}\t{probability}\n")


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
