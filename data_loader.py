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


class Loader:
    def __init__(self, CFG):
        self.config = CFG
        self.dset_name = CFG.dset_name
        self.task = CFG.task
        self.PLM = CFG.PLM
        self.batch_size = CFG.train_batch_size
        self.max_length = CFG.max_token_length
        self.seed = CFG.seed

        self.tokenizer = AutoTokenizer.from_pretrained(self.PLM)

    def load(self, mode):
        dataset = load_dataset(self.dset_name, self.task, split=mode)
        features = dataset.map(self.tokenize_and_align_labels, batched=True, remove_columns=dataset.column_names)

        all_input_ids = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f["attention_mask"] for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f["token_type_ids"] for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f["labels"] for f in features], dtype=torch.long)

        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_label_ids)

        return dataset

    @staticmethod
    def align_labels_with_tokens(labels, word_ids):
        new_labels = []
        current_word = None
        for word_id in word_ids:
            if word_id != current_word:
                # 새로운 단어의 시작 토큰.
                current_word = word_id
                label = -100 if word_id is None else labels[word_id]
                new_labels.append(label)
            elif word_id is None:
                # 특수 토큰.
                new_labels.append(-100)
            else:
                # 이전 토큰과 동일한 단어에 소속된 토큰.
                label = labels[word_id]
                # 만약 레이블이 B-XXX이면 이를 I-XXX로 변경.
                if label % 2 == 1:
                    label += 1
                new_labels.append(label)

        return new_labels

    def tokenize_and_align_labels(self, examples):
        tokenized_inputs = self.tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            padding=self.config.padding,
            max_length=self.config.max_token_length
        )
        all_labels = examples["ner_tags"]
        new_labels = []
        for i, labels in enumerate(all_labels):
            word_ids = tokenized_inputs.word_ids(i)  # 배치(batch) 인덱스 지정
            new_labels.append(self.align_labels_with_tokens(labels, word_ids))

        tokenized_inputs["labels"] = new_labels

        return tokenized_inputs
