import os
import yaml
import random
import torch

from data_loader import DataLoader
from preprocess import Preprocessor
from model import RobertaForTokenClassification
from metric import Metric

from transformers import (
    set_seed,
    AutoConfig,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)

import numpy as np
from easydict import EasyDict


def main():
    with open("config.yaml", "r") as f:
        saved_config = yaml.load(f, Loader=yaml.FullLoader)
        config = EasyDict(saved_config["CFG"])

    set_seed(config.seed)

    loader = DataLoader(config)
    datasets = loader.load()

    preprocessor = Preprocessor()
    datasets = datasets.map(preprocessor, batched=True)

    model_config = AutoConfig.from_pretrained(config.PLM)
    model_config.num_labels = config.num_labels

    model = RobertaForTokenClassification(config.PLM, config=model_config)
    tokenizer = AutoTokenizer.from_pretrained(config.PLM)

    data_collator = DataCollatorForTokenClassification(tokenizer)

    model_name = config.PLM.replace("/", "_")
    run_name = f"{model_name}_{config.seed}-finetuned-ner"

    training_args = TrainingArguments(
        run_name,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.valid_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        fp16=config.fp16,
        evaluation_strategy=config.evaluation_strategy,
        save_steps=config.save_step,
        eval_steps=config.eval_step,
        logging_steps=config.logging_step,
        save_strategy=config.save_strategy,
        metric_for_best_model=config.metric_for_best_model,
    )

    metrics = Metric()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=metrics.compute_metrics,
    )

    trainer.train()
    trainer.evaluate()
