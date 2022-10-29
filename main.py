import os
import yaml
import random
import torch

from data_loader import DataLoader
from trainer import Trainer

from transformers import set_seed

from easydict import EasyDict
from utils import init_logger


def main():
    init_logger()

    with open("config.yaml", "r") as f:
        saved_config = yaml.load(f, Loader=yaml.FullLoader)
        config = EasyDict(saved_config["CFG"])

    set_seed(config.seed)

    loader = DataLoader(config)

    train_datasets = loader.load("train")
    test_datasets = loader.load("validation")

    trainer = Trainer(config, train_datasets, test_datasets)

    trainer.train()
    trainer.evaluate("test", "eval")


if __name__ == "__main__":
    main()
