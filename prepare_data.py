import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import nlp
from transformers import T5Tokenizer, BartTokenizer, HfArgumentParser
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    task: str = field(
        metadata={"help": "Which task 'qa', 'qg', 'e2e_qg', 'ans_ext', 'multi'. 'multi' means 'qa', 'qg', 'ans_ext' tasks"}, 
    )
    model_type: str = field(metadata={"help": "One of 't5', 'bart'"})
    dataset_path: Optional[str] = field(
        default="data/fquad_multitask",
        metadata={"help": "Path for dataset directory"}, 
    )
    train_file_name: Optional[str] = field(
        default=None,
        metadata={"help": "name for cached train dataset"},
    )
    valid_file_name: Optional[str] = field(
        default=None,
        metadata={"help": "name for cached valid dataset"},
    )
    valid_for_qg_only: bool = field(
        default=False,
        metadata={"help": "For multitask dataset valid split should contain only qg task or all tasks."}
    )
    qg_format: Optional[str] = field(
        default='highlight_qg_format',
        metadata={"help": "How to format inputs for que generation, 'highlight_qg_format' or 'prepend_qg_format'"}, 
    )
    max_source_length: Optional[int] = field(
        default=512,
        metadata={"help": "Max input length for the source text"},
    )
    max_target_length: Optional[int] = field(
        default=32,
        metadata={"help": "Max input length for the target text"},
    )
    model_name: Optional[str] = field(
        default="airKlizz/t5-base-multi-fr-wiki-news",
        metadata={"help": "name of the model where to get the tokenizer"},
    )
    output_train_file_name: Optional[str] = field(
        default=None,
        metadata={"help": "name for the output train dataset"},
    )
    output_valid_file_name: Optional[str] = field(
        default=None,
        metadata={"help": "name for the output valid dataset"},
    )

class DataProcessor:
    def __init__(self, tokenizer, model_type="t5", max_source_length=512, max_target_length=32):
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.model_type = model_type
        self.hl_token = "<hl>"
        
        if model_type == "t5":
            self.sep_token = "<sep>"
        elif model_type == "bart":
            self.sep_token = "<sep>"
        else:
            self.sep_token = "[SEP]"
  
    def process(self, dataset):
        if self.model_type == "t5":
            dataset = dataset.map(self._add_eos_examples)
        
        dataset = dataset.map(self._add_special_tokens)
        dataset = dataset.map(self._convert_to_features, batched=True)
        
        return dataset
  
    def _add_eos_examples(self, example):
        example['source_text'] = example['source_text'] + " </s>"
        example['target_text'] = example['target_text'] + " </s>"
        return example
  
    def _add_special_tokens(self, example):
        example['source_text'] = example['source_text'].replace("{hl_token}", self.hl_token)    
        example['target_text'] = example['target_text'].replace("{sep_token}", self.sep_token)
        return example
  
    # tokenize the examples
    def _convert_to_features(self, example_batch):
        source_encoding = self.tokenizer.batch_encode_plus(
            example_batch['source_text'],
            max_length=self.max_source_length,
            padding='max_length',
            pad_to_max_length=True,
            truncation=True, 
        )
        target_encoding = self.tokenizer.batch_encode_plus(
            example_batch['target_text'],
            max_length=self.max_target_length,
            padding='max_length',
            pad_to_max_length=True,
            truncation=True, 
        )

        encodings = {
            'source_ids': source_encoding['input_ids'], 
            'target_ids': target_encoding['input_ids'],
            'attention_mask': source_encoding['attention_mask'],
        }

        return encodings


def filter_qa(example):
    return example['task'] == 'qa'

def filter_qg(example):
    return example['task'] == 'qg'

def filter_e2e_qg(example):
    return example['task'] == 'e2e_qg'

def filter_ans_ext(example):
    return example['task'] == 'ans_ext'

def filter_multi(example):
    return example['task'] != 'e2e_qg'


TASK_TO_FILTER_FN = {
    'qa': filter_qa,
    'qg': filter_qg,
    'e2e_qg': filter_e2e_qg,
    'ans_ext': filter_ans_ext,
    'multi': filter_multi
}


def main():
    parser = HfArgumentParser((DataTrainingArguments,))

    data_args = parser.parse_args_into_dataclasses()[0]

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    if data_args.model_type == 't5':
        tokenizer = T5Tokenizer.from_pretrained(data_args.model_name)
    else:
        tokenizer = BartTokenizer.from_pretrained(data_args.model_name)
    
    tokenizer.add_tokens(['<sep>', '<hl>'])

    df_train = pd.read_csv(os.path.join(data_args.dataset_path, data_args.train_file_name))
    df_valid = pd.read_csv(os.path.join(data_args.dataset_path, data_args.valid_file_name))

    df_train = df_train.drop(columns=["Unnamed: 0"])
    df_valid = df_valid.drop(columns=["Unnamed: 0"])

    train_dataset = nlp.Dataset.from_pandas(df_train)
    valid_dataset = nlp.Dataset.from_pandas(df_valid)


    processor = DataProcessor(
        tokenizer,
        model_type=data_args.model_type,
        max_source_length=data_args.max_source_length,
        max_target_length=data_args.max_target_length
    )

    train_dataset = train_dataset.filter(TASK_TO_FILTER_FN[data_args.task])
    if data_args.valid_for_qg_only and data_args.task != "e2e_qg":
        logger.info("processing valid data only for qg task")
        valid_dataset = valid_dataset.filter(filter_qg)
    else:
        logger.info(f"processing valid data only for {data_args.task} task")
        valid_dataset = valid_dataset.filter(TASK_TO_FILTER_FN[data_args.task])

    print("Train dataset:")
    df = train_dataset.data.to_pandas()
    print(df.head())
    print("Valid dataset:")
    df = valid_dataset.data.to_pandas()
    print(df.head())

    train_dataset = processor.process(train_dataset)
    valid_dataset = processor.process(valid_dataset)

    columns = ["source_ids", "target_ids", "attention_mask"]
    train_dataset.set_format(type='torch', columns=columns)
    valid_dataset.set_format(type='torch', columns=columns)

    train_path = os.path.join("data", data_args.output_train_file_name)
    valid_path = os.path.join("data", data_args.output_valid_file_name)
    
    torch.save(train_dataset, train_path)
    logger.info(f"saved train dataset at {train_path}")
    
    torch.save(valid_dataset, valid_path)
    logger.info(f"saved validation dataset at {valid_path}")
    
    tokenizer_path = f"{data_args.model_type}_tokenizer"
    if not os.path.exists(tokenizer_path):
        os.mkdir(tokenizer_path)
    tokenizer.save_pretrained(tokenizer_path)
    logger.info(f"saved tokenizer at {tokenizer_path}")


if __name__ == "__main__":
    main()

