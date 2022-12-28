import os
import glob

import torch
import numpy as np
from functools import partial
from datasets import load_dataset
from easydict import EasyDict
from torch.utils.data import TensorDataset
from transformers import AutoTokenizer

from utils import LABEL_MAPPING

import logging

logger = logging.getLogger(__name__)


class Loader:
    def __init__(self, CFG, tokenizer):
        self.config = CFG
        self.dset_name = CFG.dset_name
        self.task = CFG.task
        self.model_name_or_path = CFG.model_name_or_path
        self.batch_size = CFG.train_batch_size
        self.max_length = CFG.max_token_length
        self.seed = CFG.seed

        self.tokenizer = tokenizer

    def get_dataset(self, dataset_type="train"):
        model_info = self.model_name_or_path.split("/")[-1]
        cached_file_name = f"cached_{self.dset_name}_{dataset_type}-{model_info}"
        cached_features_file = os.path.join(self.config.data_dir, cached_file_name)

        if os.path.exists(cached_features_file):
            logger.info(f"Loading features from cached file {cached_features_file}")
            dataset = torch.load(cached_features_file)
        else:
            dataset = load_dataset(path=os.path.join(self.config.data_dir, self.config.dset_name),
                                   split=dataset_type)

            dataset = dataset.map(self.tokenize_and_align_labels, batched=False)
            dataset = dataset.remove_columns(["tokens", "ner_tags", "sentence", "offset_mapping"])
            torch.save(dataset, cached_features_file)
            logger.info(f"Saved features into cached file {cached_features_file}")

        return dataset

    def tokenize_and_align_labels(self, examples):
        sentence = "".join(examples["tokens"])
        tokenized_output = self.tokenizer(
            sentence,
            return_token_type_ids=True,
            return_offsets_mapping=True,
            max_length=self.max_length,
            truncation=True,
        )

        label_token_map = []

        list_label = examples["ner_tags"]
        list_label = [-100] + list_label + [-100]

        for token_idx, offset_map in enumerate(tokenized_output["offset_mapping"]):
            begin_letter_idx, end_letter_idx = offset_map
            label_begin = list_label[begin_letter_idx]
            label_end = list_label[end_letter_idx]
            token_label = np.array([label_begin, label_end])
            if label_begin == 12 and label_end == 12:
                token_label = 12
            elif label_begin == -100 and label_end == -100:
                token_label = -100
            else:
                token_label = label_begin if label_begin != 12 else 12
                token_label = label_end if label_end != 12 else 12

            label_token_map.append(token_label)

        tokenized_output["labels"] = label_token_map
        return tokenized_output
