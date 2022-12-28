import logging
import os

import numpy as np
import yaml
import random
import torch

import utils
from dotenv import load_dotenv

from data_loader import Loader

from transformers import (
    set_seed,
    Trainer,
    TrainingArguments,
    AutoConfig,
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback
)
from transformers.trainer_utils import get_last_checkpoint

from easydict import EasyDict
from utils import init_logger
from metric import compute_metrics

import argparse

logger = logging.getLogger(__name__)


def main(args):
    init_logger()
    load_dotenv()

    with open("config.yaml", "r") as f:
        saved_config = yaml.load(f, Loader=yaml.FullLoader)
        hparams = EasyDict(saved_config)

    hparams.dset_name = args.dset_name
    set_seed(hparams.seed)

    label_to_id = utils.get_labels()
    id_to_label = {v: k for k, v in label_to_id.items()}
    num_labels = len(label_to_id)

    if args.load_checkpoint:
        last_checkpoint = None
        if os.path.exists(hparams.checkpoint_path):
            last_checkpoint = get_last_checkpoint(hparams.checkpoint_path)
            if last_checkpoint is not None:
                logger.info(f"Checkpoint detected. Resuming from {last_checkpoint}")
                hparams.model_name_or_path = last_checkpoint
            else:
                logger.info(f"No checkpoint found, training from scratch: {hparams.model_name_or_path}")

        if not os.path.exists(hparams.checkpoint_path):
            os.makedirs(hparams.checkpoint_path)

    elif not args.load_checkpoint:
        if os.path.exists(hparams.checkpoint_path) and len(os.listdir(hparams.checkpoint_path)) > 1:
            hparams.model_name_or_path = hparams.checkpoint_path
            logger.info(f"***** Load Model from {hparams.model_name_or_path} *****")

    config = AutoConfig.from_pretrained(
        hparams.model_name_or_path,
        num_labels=num_labels,
        fineturning_task="ner",
        id2label=id_to_label,
        label2id=label_to_id,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        hparams.model_name_or_path,
        use_fast=True,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    model = AutoModelForTokenClassification.from_pretrained(
        hparams.model_name_or_path,
        config=config,
    )

    loader = Loader(hparams, tokenizer)

    train_datasets = loader.get_dataset(dataset_type="train") if args.do_train else None
    eval_datasets = loader.get_dataset(dataset_type="validation") if args.do_eval else None

    training_args = TrainingArguments(
        output_dir=hparams.checkpoint_path,
        do_train=args.do_train,
        do_eval=args.do_eval,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=hparams.train_batch_size,
        per_device_eval_batch_size=hparams.valid_batch_size,
        learning_rate=hparams.learning_rate,
        adam_epsilon=hparams.adam_epsilon,
        num_train_epochs=hparams.num_train_epochs,
        weight_decay=hparams.weight_decay,
        # logging_steps=hparams.logging_steps,
        seed=hparams.seed,
        fp16=hparams.fp16,
        warmup_steps=hparams.warmup_steps,
        warmup_ratio=hparams.warmup_ratio,
        gradient_accumulation_steps=hparams.gradient_accumulation_steps,
        metric_for_best_model="f1",
        load_best_model_at_end=True,
        log_level="info",
    )

    trainer = Trainer(args=training_args,
                      model=model,
                      train_dataset=train_datasets,
                      eval_dataset=eval_datasets,
                      tokenizer=tokenizer,
                      data_collator=data_collator,
                      compute_metrics=compute_metrics,
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
                      )

    if args.do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.save_model()

        metrics["train_samples"] = len(train_datasets)

        logger.info(metrics)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()

        metrics["eval_samples"] = len(eval_datasets)

        logger.info(metrics)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--load_checkpoint", action="store_true", help="Load checkpoint")
    parser.add_argument("--dset_name", default="klue", help="dataset name you want to use", choices=["klue", "docent"])

    args = parser.parse_args()
    main(args)
