# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import os
import copy
import random
from dataclasses import dataclass, field
import json
import pathlib
from PIL import Image
from typing import Dict, Optional, Sequence

import torch
from torch.utils.data import Dataset
import transformers
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother
from transformers import AutoTokenizer, BertForSequenceClassification
from torch import nn



@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="ckpts/bert-base-chinese")

@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    num_data: int = field(
        default=-1, metadata={"help": "Number of training data to use."}
    )
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=256,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    pretraining_length: int = field(
        default=256,
    )


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def rank0_write(*args):
    if local_rank == 0:
        with open("example.txt", "w") as f:
            f.write(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def read_jsonl(train_fn):
    res = []
    with open(train_fn) as f:
        for i, line in enumerate(f):
            try:
                res.append(json.loads(line))
            except:
                continue
    print(f"loading from {train_fn}, there are {len(res)} samples")
    return res


def write_jsonl(data, fn):
    with open(fn, "w") as f:
        for line in data:
            print(json.dumps(line), file=f)


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer, num_data: int):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer

        rank0_print("Loading data...")
        # load data
        with open(data_path) as f:
            list_data_dict = f.readlines()
        random.shuffle(list_data_dict)
        print("Num of training samples: ",len(list_data_dict))

        if num_data != -1:
            list_data_dict = list_data_dict[:num_data]
        print(len(list_data_dict))
        rank0_print("Formatting inputs...Skip in lazy mode")
        self.list_data_dict = list_data_dict

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        data = self.list_data_dict[i]
        data = data.replace('\t', '').replace('\n', '')
        text = data[:-1]
        label = int(data[-1])
        return text, label

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: AutoTokenizer
    def __call__(self, batch) -> Dict[str, torch.Tensor]:
        texts = []
        labels = []

        for text, label in batch:
            texts.append(text)
            labels.append(label)
            
        
        inputs = self.tokenizer(
                        texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True)
        inputs['labels'] = torch.LongTensor(labels)
        return inputs

def make_supervised_data_module(
    tokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, num_data=data_args.num_data)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()


    local_rank = training_args.local_rank
    model = BertForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
    )
    model.classifier = nn.Linear(768, 6)
    model.config.num_labels = 6
    model.num_labels = 6
    # model.bert.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path,
                                              model_max_length=training_args.model_max_length)

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    trainer = Trainer(
        model=model, args=training_args, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()