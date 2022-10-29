import torch
import numpy as np
from functools import partial
from datasets import load_dataset
from easydict import EasyDict
from torch.utils.data import TensorDataset
from transformers import AutoTokenizer

from utils import LABEL_MAPPING


class InputFeatures:
    def __init__(self, input_ids, attention_mask, token_type_ids, label_ids):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label_ids = label_ids


class DataLoader:
    def __init__(self, CFG):
        self.config = CFG
        self.dset_name = CFG.dset_name
        self.task = CFG.task
        self.PLM = CFG.PLM
        self.batch_size = CFG.train_batch_size
        self.max_length = CFG.max_token_length
        self.seed = CFG.seed

    def load(self, mode):
        dataset = load_dataset(self.dset_name, self.task, split=mode)
        tokenizer = AutoTokenizer.from_pretrained(self.PLM)
        encode_fn = partial(self.label_tokens_ner, tokenizer=tokenizer)
        features = list(map(encode_fn, dataset))

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)

        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_label_ids)

        return dataset

    def label_tokens_ner(self, examples, tokenizer):
        sentence = "".join(examples["tokens"])
        tokenized_output = tokenizer(
            sentence,
            return_token_type_ids=True,
            return_offsets_mapping=True,
            max_length=self.config.max_token_length,
            padding=self.config.padding,
            truncation=True)
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

        tokenized_output["labels"] = [LABEL_MAPPING[t] for t in label_token_map]

        input_ids = tokenized_output["input_ids"]
        attention_mask = tokenized_output["attention_mask"]
        token_type_ids = tokenized_output["token_type_ids"]
        label_ids = tokenized_output["labels"]

        features = InputFeatures(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            label_ids=label_ids
        )

        return features
