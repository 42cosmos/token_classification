import os
import yaml
import random
import torch

import wandb
from dotenv import load_dotenv

from data_loader import Loader
from trainer import Trainer

from transformers import set_seed

from easydict import EasyDict
from utils import init_logger

import argparse


def main(args):
    init_logger()
    load_dotenv()

    WANDB_API_KEY = os.getenv('WANDB_API_KEY')
    wandb.login(key=WANDB_API_KEY)

    with open("config.yaml", "r") as f:
        saved_config = yaml.load(f, Loader=yaml.FullLoader)
        config = EasyDict(saved_config["CFG"])

    set_seed(config.seed)
    wandb.init(entity=config.entity_name, project=config.project_name, config=config)

    loader = Loader(config)

    train_datasets = loader.load("train")
    test_datasets = loader.load("validation")

    trainer = Trainer(config, train_datasets, test_datasets)

    if args.do_train:
        trainer.train()

    if args.do_eval:
        trainer.load_model()
        trainer.evaluate("test", "eval")

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")

    args = parser.parse_args()
    main(args)
