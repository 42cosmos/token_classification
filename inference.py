import os
import logging
import argparse

import yaml
import random
import numpy as np
from easydict import EasyDict
from dotenv import load_dotenv
from tqdm import tqdm, trange
import torch

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

import evaluate
from seqeval.metrics import classification_report

import utils
from data_loader import Loader
from metric import compute_metrics

logger = logging.getLogger(__name__)


def to_tensor(x):
    return x.detach().cpu().numpy()


def main(args):
    utils.init_logger()
    load_dotenv()

    with open("config.yaml", "r") as f:
        saved_config = yaml.load(f, Loader=yaml.FullLoader)
        hparams = EasyDict(saved_config)

    hparams.dset_name = args.dset_name
    set_seed(hparams.seed)

    label_to_id = utils.get_labels()
    id_to_label = {v: k for k, v in label_to_id.items()}
    num_labels = len(label_to_id)

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

    model = AutoModelForTokenClassification.from_pretrained(
        hparams.model_name_or_path,
        config=config,
    )
    loader = Loader(hparams, tokenizer)

    dset_type = "test" if args.dset_name == "docent" else "validation"
    test_dataset = loader.get_dataset(dataset_type=dset_type)

    preds = []
    trues = []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    model.eval()

    epoch_iterator = tqdm(test_dataset, desc="Iteration")
    debug = hparams.debug
    for step, batch in enumerate(epoch_iterator):
        trues.append(batch.pop("labels"))
        batch = {k: torch.tensor(t).unsqueeze(0).to(device) for k, t in batch.items()}
        outputs = model(**batch)
        pred = outputs.logits.argmax(dim=2)
        pred = pred.detach().cpu().numpy()
        if debug:
            transform_text = ""
            print(f'************** Step : {step} **************')
            original_text = loader.tokenizer.decode(batch['input_ids'][0][1:-1])
            decode_predictions = [id_to_label[p] for p in pred[0][1:-1]]

            print(f'Original Text : {original_text}')
            print(f'Decode Label : {decode_predictions}')
            for token_ids, label_ids in zip(batch['input_ids'][0][1:-1], pred[0][1:-1]):
                transform_text += f'{loader.tokenizer.decode(token_ids).replace("##", "")}({id_to_label[label_ids]})'
            print(transform_text)
        preds.extend(pred)

    result = compute_metrics(p=(preds, trues), inference=True, compress=True)
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dset_name", default="klue", help="dataset name you want to use", choices=["klue", "docent"])

    args = parser.parse_args()
    main(args)
