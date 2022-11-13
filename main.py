import logging
import os
import yaml
import random
import torch

import wandb
from dotenv import load_dotenv

from data_loader import Loader
from trainer import Trainer

from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from easydict import EasyDict
from utils import init_logger

import argparse


def main(args):
    init_logger()
    logger = logging.getLogger(__name__)
    load_dotenv()

    WANDB_API_KEY = os.getenv('WANDB_API_KEY')
    wandb.login(key=WANDB_API_KEY)

    with open("config.yaml", "r") as f:
        saved_config = yaml.load(f, Loader=yaml.FullLoader)
        config = EasyDict(saved_config["CFG"])

    set_seed(config.seed)

    last_checkpoint = None
    if os.path.exists(config.checkpoint_path):
        last_checkpoint = get_last_checkpoint(config.checkpoint_path)
        if last_checkpoint is not None:
            logger.info(f"Checkpoint detected. Resuming from {last_checkpoint}")
            config.model_name_or_path = last_checkpoint
        else:
            logger.info(f"No checkpoint found, training from scratch: {config.model_name_or_path}")

    wandb.init(entity=config.entity_name, project=config.project_name, config=config)

    loader = Loader(config)

    train_datasets = loader.get_dataset(evaluate=False) if args.do_train else None
    test_datasets = loader.get_dataset(evaluate=True) if args.do_eval else None

    trainer = Trainer(config, train_datasets, test_datasets)

    if args.do_train:
        trainer.train()

    if args.do_eval:
        trainer.load_model()
        trainer.evaluate("test", "eval")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")

    args = parser.parse_args()
    main(args)
