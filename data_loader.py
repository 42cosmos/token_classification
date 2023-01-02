import os
import logging
import pandas as pd

import torch

import pyarrow as pa
import pyarrow.dataset as ds
from datasets import load_dataset, Dataset, concatenate_datasets

from tqdm import tqdm

from utils import LABEL_MAPPING

logger = logging.getLogger(__name__)


class DocentDataset:
    def __init__(self, dataset):
        self.dataset = dataset
        self.assign_sentence_and_label()
        self.phrase_to_sentence()

    def assign_sentence_and_label(self):
        self.original_phrase = [''.join(row) for row in self.dataset['tokens']]
        self.original_label = [row for row in self.dataset['ner_tags']]

    def phrase_to_sentence(self):
        self.sentence = []
        self.label = []
        error = 0
        for phrase, labels in tqdm(zip(self.original_phrase, self.original_label), total=len(self.original_phrase)):
            try:
                sentences = kss.split_sentences([" " if i.strip() == "" else i.strip() for i in phrase], strip=False)
            except:
                error += 1
                print(f'Error Occurred : {error}')
            use_len_sentence = 0
            for sentence in sentences:
                self.sentence.append(sentence)
                self.label.append(labels[use_len_sentence:use_len_sentence + len(sentence)])
                use_len_sentence += len(sentence)


class Loader:
    def __init__(self, config, tokenizer):
        self.config = config
        self.dset_name = config.dset_name
        self.task = config.task
        self.model_name_or_path = config.model_name_or_path
        self.batch_size = config.train_batch_size
        self.max_length = config.max_token_length
        self.seed = config.seed

        self.tokenizer = tokenizer

    def get_dataset(self, dataset_type="train"):
        model_info = self.model_name_or_path.split("/")[-1]
        cached_file_name = f"cached_{self.dset_name}_{dataset_type}-{model_info}"
        cached_features_file = os.path.join(self.config.data_dir, cached_file_name)

        docent_cached_file_name = f"cached_docent_{dataset_type}-koelectra-base-v3-discriminator"
        docent_cached_features_file = os.path.join(self.config.data_dir, docent_cached_file_name)

        if os.path.exists(cached_features_file):
            logger.info(f"Loading features from cached file {cached_features_file}")
            dataset = torch.load(cached_features_file)

            if dataset_type == "train" or dataset_type == "validation":
                if not os.path.exists(cached_features_file):
                    klue_dataset = load_dataset(path=os.path.join(self.config.data_dir, "klue"),
                                                split=dataset_type)

                    col_name = klue_dataset.column_names
                    klue_dataset = klue_dataset.map(self.tokenize_and_align_labels, batched=False)
                    klue_dataset = klue_dataset.remove_columns(col_name)

                    dataset = concatenate_datasets([klue_dataset, dataset])


        else:
            dataset = load_dataset(path=os.path.join(self.config.data_dir, self.dset_name),
                                   split=dataset_type)

            col_name = dataset.column_names
            dataset = dataset.map(self.tokenize_and_align_labels, batched=False)
            dataset = dataset.remove_columns(col_name)

            torch.save(dataset, cached_features_file)
            logger.info(f"Saved features into cached file {cached_features_file}")

        return dataset

    def tokenize_and_align_labels(self, examples):
        sentence = "".join(examples["tokens"])
        tokenized_output = self.tokenizer(
            sentence,
            return_token_type_ids=True if self.config.use_token_types else False,
            return_offsets_mapping=False,
            max_length=self.max_length,
            truncation=True,
        )

        list_label = examples["ner_tags"]
        tokenized_sentence = [word for word in self.tokenizer.tokenize(sentence)]
        sub_word_based_label = []
        start_tokenized_word_idx = 0
        sub_word = ''
        for idx, (char_tok, char_lbl) in enumerate(zip(sentence, list_label)):
            if char_tok == ' ':
                continue
            else:
                sub_word += char_tok
            if sub_word == tokenized_sentence[start_tokenized_word_idx].replace('##', ''):
                start_tokenized_word_idx += 1
                sub_word_based_label.append(list_label[idx - len(sub_word) + 1])
                sub_word = ''

        # based_label = [LABEL_MAPPING[i] for i in sub_word_based_label[:self.max_length - 2]]
        sub_word_based_label = [-100] + sub_word_based_label[:self.max_length - 2] + [-100]

        tokenized_output["labels"] = sub_word_based_label
        return tokenized_output
