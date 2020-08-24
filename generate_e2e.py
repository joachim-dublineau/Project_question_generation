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

from transformers import (
    BertConfig,
    BertTokenizer,
    CamembertConfig,
    CamembertTokenizer,
    EncoderDecoderConfig,
    EncoderDecoderModel,
    BartConfig,
    BartTokenizer,
    BartForConditionalGeneration,
    T5Config,
    T5Tokenizer,
    T5ForConditionalGeneration,
    )
from natixis_utils import DataPrepModelAssess, clean_dataframe

import argparse

# ARGUMENT PARSING

parser = argparse.ArgumentParser()
# Compulsory arguments
parser.add_argument("file_data", help="name of the file containing the contexts")
parser.add_argument("output_dir", help="name of the directory where to export the generated questions")
parser.add_argument("file_name", help="name of the csv file that will be saved")

# Optional arguments
parser.add_argument("-bt", "--bart", help="true if bart else false, default False", type=bool, default=False, action="store")
parser.add_argument("-t5", "--t5", help='true if t5 else false, default False', type=bool, default=True, action="store")
parser.add_argument("-tk", "--tokenizer", help="name or path of where to find the tokenizer", type=str, default="", action="store")
parser.add_argument("-mi", "--max_length_input", help="max length of input sequence, default 256", type=int, default=512, action="store")
parser.add_argument("-mo", "--max_length_output", help="max_length of output sequence, defaut 21", type=int, default=21, action="store")
parser.add_argument("-ck", "--checkpoint", help="directory where to find the checkpoint of the model, default None", type=str, default=None, action='store')
parser.add_argument("-bs", "--batch_size", help="batch size for training, default 16", type=int, default=16, action='store')
parser.add_argument("-rp", "--repetition_penalty", help='repetition penalty parameter for generation, default 2', type=float, default=2.0, action="store")
parser.add_argument("-lp", "--length_penalty", help='length penalty parameter for generation, default 2', type=float, default=2.0, action="store")
parser.add_argument("-nb", "--num_beams", help="number of beams, parameter for generation, default 1", type=int, default=1, action="store")
parser.add_argument("-tp", "--temperature", help="temperature parameter for softmax in generation, default 1.0", type=float, default=1.0, action="store")

args = parser.parse_args()

def add_string(contexts, string):
    context_bis = []
    for text in contexts:
        context_bis.append(string + text)
    return context_bis

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

    # LOADING MODEL
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)
    model_created = False

    if args.checkpoint != None:
        model_created = True
        if args.bart:
            config = BartConfig.from_json_file(args.checkpoint + "/config.json")
            model = BartForConditionalGeneration.from_pretrained(args.checkpoint + "/pytorch_model.bin", config=config)
        if args.t5:
            config = T5Config.from_json_file(args.checkpoint + "/config.json")
            model = T5ForConditionalGeneration.from_pretrained(args.checkpoint + "/pytorch_model.bin", config=config)
        elif not args.bart and not args.t5:
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
        model_name = "airKlizz/t5-base-multi-fr-wiki-news"
        if args.tokenizer != "":
            tokenizer = T5Tokenizer.from_pretrained(args.tokenizer)
        else:
            print("Please provide a correct tokenizer for T5.")
        if not model_created:
            model = T5ForConditionalGeneration.from_pretrained(model_name)
            model_created = True

    model.to(device)

    # GENERATION
    max_length_seq = args.max_length_input
    max_length_label = args.max_length_output

    sentences = df_generation["context"]

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
    }

    nb_batch = len(sentences)//args.batch_size
    generated_questions = []
    list_contexts = sentences.tolist()
    for i in range(nb_batch):
        batch_contexts = list_contexts[i*args.batch_size:(i+1)*args.batch_size] if i != nb_batch - 1 else \
            list_contexts[i*args.batch_size:]
        batch_hl_contexts = add_string(batch_contexts, "generate questions: ")
        list_inputs = tokenizer.batch_encode_plus(batch_hl_contexts, padding=True, max_length=max_length_seq)
        input_ids = torch.tensor(list_inputs["input_ids"]).to(device)
        attention_mask = torch.tensor(list_inputs["attention_mask"]).to(device)
        generation_hyperparameters["input_ids"] = input_ids
        generation_hyperparameters["attention_mask"] = attention_mask
        batch_generated_tokens = model.generate(**generation_hyperparameters)
        batch_generated_questions = tokenizer.batch_decode(batch_generated_tokens)
        generated_questions += batch_generated_questions

    df_generation["question"] = generated_questions

    # SAVING
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    df_generation.to_csv(args.output_dir + "/" + args.file_name, sep=",")
    print("Generated.")

