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
    default_data_collator,
    EarlyStoppingCallback
)
from transformers.trainer_utils import get_last_checkpoint

from easydict import EasyDict
from utils import init_logger
from metric import compute_metrics

import argparse

import evaluate
from seqeval.metrics import classification_report


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

    logger.info(f"***** Load Model from {hparams.model_name_or_path} *****")
    hparams.model_name_or_path = hparams.checkpoint_path

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

    model = AutoModelForTokenClassification.from_pretrained(
        hparams.model_name_or_path,
        config=config,
    )

    loader = Loader(hparams, tokenizer)

    test_dataset = loader.get_dataset(dataset_type="test")

    preds = []
    trues = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for step, batch in enumerate(test_dataset):
        model.eval()

        trues.extend(batch["labels"])
        batch = {k: torch.tensor(t).unsqueeze(0) for k, t in batch.items()}

        outputs = model(**batch)

        pred = outputs.logits.argmax(dim=2)
        pred = pred.detach().cpu().numpy()
        preds.extend(pred)

    true_predictions = [
        [id_to_label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(preds, trues)
    ]

    true_labels = [
        [id_to_label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(preds, trues)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    logger.info(f"\n {classification_report(true_labels, true_predictions, suffix=False)}")

    for k, v in results.items():
        logger.info(f"{k}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dset_name", default="klue", help="dataset name you want to use", choices=["klue", "docent"])

    args = parser.parse_args()
    main(args)
