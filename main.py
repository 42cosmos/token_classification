import os
import yaml
import random
import torch

import wandb
from dotenv import load_dotenv

from data_loader import DataLoader
from trainer import Trainer

from transformers import set_seed

from easydict import EasyDict
from utils import init_logger


def main():
    init_logger()

    load_dotenv()
    WANDB_API_KEY = os.getenv('WANDB_API_KEY')
    wandb.login(key=WANDB_API_KEY)

    with open("config.yaml", "r") as f:
        saved_config = yaml.load(f, Loader=yaml.FullLoader)
        config = EasyDict(saved_config["CFG"])

    wandb.init(entity=config.entity_name, project=config.project_name, config=config)
    set_seed(config.seed)

    loader = DataLoader(config)

    train_datasets = loader.load("train")
    test_datasets = loader.load("validation")

    trainer = Trainer(config, train_datasets, test_datasets)

    trainer.train()
    trainer.evaluate("test", "eval")
    wandb.finish()


if __name__ == "__main__":
    main()
