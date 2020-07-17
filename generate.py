# coding: utf-8

######## Script: Using a QG model on a given dataset ########

# This python file contains a script that shows how to use an EncoderDecoderModel or a BART model to generate questions.
# Attention: this script is supposed to work only for CamemBERT and BART, other models were not tested.

# Author: Joachim Dublineau
# Zelros A.I.

# IMPORTS:
import pandas as pd

import time
import random
random.seed(2020)
import os

import torch

from transformers import (BertConfig,
                          BertTokenizer,
                          CamembertConfig,
                          CamembertTokenizer,
                          EncoderDecoderConfig,
                          EncoderDecoderModel,
                          BartConfig,
                          BartTokenizer,
                          BartForConditionalGeneration,
                          )

from transformers_utils import load_examples_question_generation, generate_questions
import spacy
from natixis_utils import DataPrepModelAssess, clean_dataframe, custom_tokenizer, select_keywords_spacy, separating_keywords
import argparse

# ARGUMENT PARSING

parser = argparse.ArgumentParser()
# Compulsory arguments
parser.add_argument("file_data", help="name of the file containing the contexts")
parser.add_argument("output_dir", help="name of the directory where to export the generated questions")

# Optional arguments
parser.add_argument("-bt", "--bart", help="true if bart else false, default False", type=bool, default=False, action="store")
parser.add_argument("-mi", "--max_length_input", help="max length of input sequence, default 256", type=int, default=256, action="store")
parser.add_argument("-mo", "--max_length_output", help="max_length of output sequence, defaut 21", type=int, default=21, action="store")
parser.add_argument("-ck", "--checkpoint", help="directory where to find the checkpoint of the model, default None", type=str, default=None, action='store')
parser.add_argument("-bs", "--batch_size", help="batch size for training, default 16", type=int, default=16, action='store')
parser.add_argument("-rp", "--repetition_penalty", help='repetition penalty parameter for generation, default 2', type=float, default=2.0, action="store")
parser.add_argument("-lp", "--length_penalty", help='length penalty parameter for generation, default 2', type=float, default=2.0, action="store")
parser.add_argument("-nb", "--num_beams", help="number of beams, parameter for generation, default 1", type=int, default=1, action="store")
parser.add_argument("-tp", "--temperature", help="temperature parameter for softmax in generation, default 1.0", type=float, default=1.0, action="store")

args = parser.parse_args()

if __name__ == "__main__":
    # LOADING & PREPROCESSING DATA
    print("Loading and preprocessing data...", end="", flush=True)
    t0 = time.time()
    file_data = args.file_data
    #working_directory = os.getcwd().split("/")[:-1]
    #working_directory = "/".join(working_directory)
    #params = {'scenario_file': working_directory + "/data-anna/scenario_natixis_prod.json"}
    params = {'scenario_file': file_data}
    data_prep = DataPrepModelAssess(**params)
    df_generation = clean_dataframe(data_prep.df_scenario, "context")
    data_prep.set_df_scenario(df_generation)
    df_generation = data_prep.df_cleaned_scenario
    df_generation = df_generation.drop(columns=["questions", 'title', 'index'])
    print("Done. {:.2f}s".format(time.time() - t0))
    print(df_generation.head())

     # EXTRACTING KEYWORDS
    nlp = spacy.load("fr_core_news_sm")
    nlp.tokenizer = custom_tokenizer(nlp)
    t0 = time.time()
    print("Extracting keywords...", end="", flush=True)
    df_generation = select_keywords_spacy(df_generation, "context", 5, nlp)
    df_generation = separating_keywords(df_generation, "keywords")
    print("Done. {:.2f}s".format(time.time() - t0))

    # LOADING MODEL
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)
    model_created = False

    if args.checkpoint != None:
        model_created = True
        if args.bart:
            config = BartConfig.from_json_file(args.checkpoint + "/config.json")
            model = BartForConditionalGeneration.from_pretrained(args.checkpoint + "/pytorch_model.bin", config=config)
        else:
            config = EncoderDecoderConfig.from_json_file(args.checkpoint + "/config.json")
            model = EncoderDecoderModel.from_pretrained(args.checkpoint + "/pytorch_model.bin", config=config)

    if args.bart:
        model_name = "WikinewsSum/bart-large-multi-fr-wiki-news"
        # config = BartConfig.from_pretrained(model_name)
        tokenizer = BartTokenizer.from_pretrained(model_name, do_lower_case=True)
        if not model_created:
            model = BartForConditionalGeneration.from_pretrained(model_name)
            model_created = True
    else:
        model_name = "camembert-base"
        # config = CamembertConfig.from_pretrained(model_name)
        tokenizer = CamembertTokenizer.from_pretrained(model_name, do_lower_case=True)
        if not model_created:
            model = EncoderDecoderModel.from_encoder_decoder_pretrained(model_name, model_name)
    print("Model used:", model_name)

    model.to(device)

    # GENERATION
    max_length_seq = args.max_length_input
    max_length_label = args.max_length_output

    answers = df_generation["answer_span"]
    sentences = df_generation["context"]

    generation_dataset = load_examples_question_generation(
        answers=answers,
        sentences=sentences,
        tokenizer=tokenizer,
        max_length_seq=max_length_seq,
        max_length_label=max_length_label,
        bart=args.bart,
        )

    generation_hyperparameters = {
        'min_length': 5,
        'max_length': max_length_label,
        'repetition_penalty': args.repetition_penalty,
        'length_penalty': args.length_penalty,
        'num_beams': args.num_beams,
        'temperature': args.temperature,
        'decoder_start_token_id': tokenizer.cls_token_id,
        'bos_token_id': tokenizer.cls_token_id,
        'pad_token_id': tokenizer.pad_token_id,
        'eos_token_id': tokenizer.convert_tokens_to_ids("?"),
    }
    results = generate_questions(
        model=model,
        dataset=generation_dataset,
        tokenizer=tokenizer,
        device=device,
        batch_size=args.batch_size,
        generation_hyperparameters=generation_hyperparameters,
    )

    generated_questions = []
    for batch in results:
        for question in batch:
            generated_questions.append(question)

    df_generation["question"] = generated_questions

    # SAVING
    df_generation.to_csv(args.output_dir + "/generated_questions.csv", sep=",")
