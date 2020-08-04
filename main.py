# coding: utf-8

######## Script: Training of an EncoderDecoderModel with transformers_utils.py ########

# This python file contains a script that shows how to train an EncoderDecoderModel or a BART model.
# Attention: this script is supposed to work only for BERT, BART and CamemBERT, other models were not tested.

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

from sklearn.model_selection import train_test_split

from transformers_utils import (load_json_QuAD_v1,
                                load_json_QuAD_v2,
                                load_examples_question_generation,
                                train_question_generation,
                                generate_questions,
                                retrieval_score,
                                )

import argparse
from nlgeval import NLGEval

# ARGUMENT PARSING

parser = argparse.ArgumentParser()
# Compulsory arguments
parser.add_argument("language", help="en or fr", choices=['en', 'fr'])
parser.add_argument("file_train", help="name of the train file")
parser.add_argument("file_test", help="name of the test file")
parser.add_argument("output_dir", help="name of the directory for logs and checkpoints")

# Optional arguments
parser.add_argument("-csv", "--is_csv", help="true if the file_train and file_test are csv, default false", type=bool, default=False, action="store")
parser.add_argument("-bt", "--bart", help="true if bart else false, default False", type=bool, default=False, action="store")
parser.add_argument("-ls", "--logging_steps", help="number of steps between each evaluation, default 50", type=int, default=50, action="store")
parser.add_argument("-mi", "--max_length_input", help="max length of input sequence, default 256", type=int, default=256, action="store")
parser.add_argument("-mo", "--max_length_output", help="max_length of output sequence, defaut 21", type=int, default=21, action="store")
parser.add_argument("-ck", "--checkpoint", help="directory where to find last checkpoint, default None", type=str, default=None, action='store')
parser.add_argument("-lr", "--learning_rate", help="default learning rate, default 1e-4", type=float, default=1e-4, action='store')
parser.add_argument("-bs", "--batch_size", help="batch size for training, default 16", type=int, default=16, action='store')
parser.add_argument("-ss", "--save_steps", help="number of gradient descent steps between each saving, default 400", type=int, default=400, action='store')
parser.add_argument("-ep", "--epochs", help="number of epochs for training, default 10", type=int, default=10, action='store')
parser.add_argument("-gs", "--gradient_accumulation_steps", help='number of steps before backward step, default 50', type=int, default=50, action='store')
parser.add_argument("-wd", "--weight_decay", help="weight decay parameter for training, default 1e-5", type=float, default=1e-5, action='store')
parser.add_argument("-fb", "--file_bis", help="option to add name of piaf file (for fr model), default None", type=str, default="", action="store")
parser.add_argument("-rp", "--repetition_penalty", help='repetition penalty parameter for generation, default 2', type=float, default=2.0, action="store")
parser.add_argument("-lp", "--length_penalty", help='length penalty parameter for generation, default 2', type=float, default=2.0, action="store")
parser.add_argument("-nb", "--num_beams", help="number of beams, parameter for generation, default 1", type=int, default=1, action="store")
parser.add_argument("-tp", "--temperature", help="temperature parameter for softmax in generation, default 1.0", type=float, default=1.0, action="store")
parser.add_argument("-eo", "--evaluate_on", help="number of examples on which to evaluate the model, default 100", type=int, default=100, action="store")

args = parser.parse_args()

if __name__ == "__main__":

    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    # LOADING MODEL & DATA

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

    if args.language == 'fr':
        if args.bart:
            model_name = "WikinewsSum/bart-large-multi-fr-wiki-news"
            #config = BartConfig.from_pretrained(model_name)
            tokenizer = BartTokenizer.from_pretrained(model_name, do_lower_case=True)
            if not model_created:
                model = BartForConditionalGeneration.from_pretrained(model_name)
                model_created = True
        else:
            model_name = "camembert-base"
            #config = CamembertConfig.from_pretrained(model_name)
            tokenizer = CamembertTokenizer.from_pretrained(model_name, do_lower_case=True)
        print("Model used:", model_name)

        # FQuAD
        if args.is_csv:
            df_train = pd.read_csv(args.file_train)
            df_valid = pd.read_csv(args.file_test)
        else:
            try:
                df_train = load_json_QuAD_v2(args.file_train)
                df_valid = load_json_QuAD_v2(args.file_test)
            except:
                df_train = load_json_QuAD_v1(args.file_train)
                df_valid = load_json_QuAD_v1(args.file_test)
                pass

        if args.file_bis != "":
            piaf_df_train, piaf_df_valid = train_test_split(
                load_json_QuAD_v1(args.file_bis), test_size=0.1)
            df_train = pd.concat([df_train, piaf_df_train])
            df_train = df_train.reset_index(drop=True)
            df_valid = pd.concat([df_valid, piaf_df_valid])
            df_valid = df_valid.reset_index(drop=True)

    else:
        if args.bart:
            model_name = 'facebook/bart-large'
            # config = BartConfig.from_pretrained(model_name)
            tokenizer = BartTokenizer.from_pretrained(model_name, do_lower_case=True)
            if not model_created:
                model = BartForConditionalGeneration.from_pretrained(model_name)
                model_created = True
        else:
            model_name = 'bert-base-uncased'
            #config = BertConfig.from_pretrained(model_name)
            tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)
        print('Model used:', model_name)

        # SQuAD
        if args.is_csv:
            df_train = pd.read_csv(args.file_train)
            df_valid = pd.read_csv(args.file_test)
        else:
            try:
                df_train = load_json_QuAD_v2(args.file_train)
                df_valid = load_json_QuAD_v2(args.file_test)
            except:
                df_train = load_json_QuAD_v1(args.file_train)
                df_valid = load_json_QuAD_v1(args.file_test)
                pass

    if not model_created:
        model = EncoderDecoderModel.from_encoder_decoder_pretrained(model_name, model_name)

    model.to(device)

    max_length_seq = args.max_length_input
    max_length_label = args.max_length_output

    answers_train = df_train["answer_span"]
    sentences_train = df_train["context"]
    labels_train = df_train["question"]
    t0 = time.time()
    print("Loading train dataset...", end='', flush=True)
    train_dataset = load_examples_question_generation(
        answers=answers_train,
        sentences=sentences_train,
        labels=labels_train,
        tokenizer=tokenizer,
        max_length_seq=max_length_seq,
        max_length_label=max_length_label,
        bart=args.bart,
        )
    print("Done. {:.4f}s".format(time.time() - t0))

    print("Loading eval dataset...", end="", flush=True)
    answers_eval = df_valid["answer_span"]
    sentences_eval = df_valid["context"]
    labels_eval = df_valid["question"]
    t0 = time.time()
    eval_dataset = load_examples_question_generation(
        answers=answers_eval,
        sentences=sentences_eval,
        labels=labels_eval,
        tokenizer=tokenizer,
        max_length_seq=max_length_seq,
        max_length_label=max_length_label,
        bart=args.bart,
        )
    print("Done.{:.4f}s".format(time.time() - t0))

    print("Loading generation dataset...", end="", flush=True)
    t0 = time.time()
    idx_examples = random.sample(list(range(len(answers_eval))), 10)

    generation_dataset = load_examples_question_generation(
        answers=answers_eval[idx_examples].values,
        sentences=sentences_eval[idx_examples].values,
        labels=labels_eval[idx_examples].values,
        tokenizer=tokenizer,
        max_length_seq=max_length_seq,
        max_length_label=max_length_label,
        bart=args.bart,
        )
    print("Done. {:.4f}s".format(time.time() - t0))

    # TRAINING
    generation_hyperparameters = {
        'min_length': 5,
        'max_length': max_length_label,
        'repetition_penalty': args.repetition_penalty,
        'length_penalty': args.length_penalty,
        'num_beams': args.num_beams,
        'temperature': args.temperature,
        'decoder_start_token_id': tokenizer.cls_token_id,
        'bos_token_id': tokenizer.bos_token_id,
        'pad_token_id': tokenizer.pad_token_id,
        'eos_token_id': tokenizer.convert_tokens_to_ids("?"),
    }

    train_question_generation(
        model=model,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        num_train_epochs=args.epochs,
        train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=device,
        adam_epsilon=1e-8,
        logging_steps=args.logging_steps,
        logging_dir=args.output_dir,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=1.0,
        weight_decay=args.weight_decay,
        warmup_steps=0,
        output_dir=args.output_dir,
        max_steps=-1,
        num_cycles=7,
        evaluate_during_training=True,
        eval_dataset=eval_dataset,
        eval_batch_size=args.batch_size,
        generation_during_training=True,
        generation_dataset=generation_dataset,
        generation_hyperparameters=generation_hyperparameters,
        save_steps=args.save_steps,
        verbose=1,
        )

    # SAVING
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    save_model_dir = os.path.join(args.output_dir, 'checkpoint-last')
    os.makedirs(save_model_dir)
    model.save_pretrained(save_model_dir)

    # COMPUTING METRICS:
    print("Computing metrics...", end="", flush=True)
    idx_examples = random.sample(list(range(len(answers_eval))), args.evaluate_on)
    metric_dataset = load_examples_question_generation(
        answers=answers_eval[idx_examples].values,
        sentences=sentences_eval[idx_examples].values,
        labels=labels_eval[idx_examples].values,
        tokenizer=tokenizer,
        max_length_seq=max_length_seq,
        max_length_label=max_length_label,
        )
    results = generate_questions(
        model=model,
        dataset=metric_dataset,
        tokenizer=tokenizer,
        device=device,
        batch_size=args.batch_size,
        generation_hyperparameters=generation_hyperparameters,
    )
    references, hypothesis = [], []
    for elem in results:
        for i in range(len(elem[0])):
            references.append(elem[1][i])
            hypothesis.append(elem[0][i])
    nlgeval = NLGEval()  # loads the models
    metrics_dict = nlgeval.compute_metrics([references], hypothesis)
    print("Done.")
    str_ = ""
    with open(args.output_dir + '/logs.txt', "a") as writer:
        for metric in metrics_dict:
            str_ += metric + ": {:.3f}, ".format(metrics_dict[metric])
        str_ += "Retrieval score: {:.3f}".format(retrieval_score(hypothesis, references))
    writer.write(str_)
    print(str_)

