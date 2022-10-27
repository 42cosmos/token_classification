import numpy as np
from functools import partial
from datasets import load_dataset
from easydict import EasyDict
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class DataLoader:
    def __init__(self, CFG):
        self.dset_name = CFG.dset_name
        self.task = CFG.task
        self.PLM = CFG.PLM
        self.batch_size = CFG.train_batch_size
        self.max_length = CFG.max_token_length
        self.seed = CFG.seed

    def load(self):
        dataset = load_dataset(self.dset_name, self.task)
        tokenizer = AutoTokenizer.from_pretrained(self.PLM)
        encode_fn = partial(self.label_tokens_ner, tokenizer=tokenizer)
        dataset = dataset.map(encode_fn, batched=False)
        dataset = dataset.remove_columns(column_names=["sentence", "tokens", "ner_tags", "offset_mapping"])

        return dataset

    def label_tokens_ner(self, examples, tokenizer):
        sentence = "".join(examples["tokens"])
        tokenized_output = tokenizer(
            sentence,
            return_token_type_ids=False,
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
            # else:
            #   raise ValueError("label_begin != label_end")

        tokenized_output["labels"] = label_token_map
        return tokenized_output
