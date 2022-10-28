import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from typing import Optional, Tuple, Union

from transformers import AutoConfig, BertForTokenClassification
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel


class Trainer(object):
    def __init__(self, config, train_dataset=None, test_dataset=None):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        # self.label_list = pass
        # id2label = {i: label for i, label in enumerate(self.label_list)}
        # label2id = {label: i for i, label in enumerate(self.label_list)}
        self.num_labels = config.num_labels
        self.pad_token_label_id = CrossEntropyLoss().ignore_index

        self.model_config = AutoConfig.from_pretrained(config.PLM,
                                                       num_labels=self.num_labels,
                                                       id2label=id2label,
                                                       label2id=label2id,
                                                       finetuning_task=config.task,
                                                       )
        # self.model_config, self.model_class, _ = MODEL_CLASSES[config.model_type]

        self.model = BertForTokenClassification.from_pretrained(config.PLM,
                                                                config=self.model_config
                                                                )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def train(self):
        pass
