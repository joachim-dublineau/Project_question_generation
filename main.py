# coding: utf-8

######## Script: Training of an EncoderDecoderModel with transformers_utils.py ########

# This python file contains a script that shows how to train an EncoderDecoderModel.
# Attention: this script is supposed to work only for BERT and CamemBERT, other models
# were not tested.

# Author: Joachim Dublineau
# Zelros A.I.

# IMPORTS:
import pandas as pd

import time
import random
random.seed(2020)

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
                                )

import argparse

# ARGUMENT PARSING

parser = argparse.ArgumentParser()
# Compulsory arguments
parser.add_argument("language", help="en or fr", choices=['en', 'fr'])
parser.add_argument("file_train", help="name of the train file")
parser.add_argument("file_test", help="name of the test file")
parser.add_argument("output_dir", help="name of the directory for logs and checkpoints")

# Optional arguments
parser.add_argument("-bt", "--bart", help="true if bart else false", action="store", type=bool, default=False)
parser.add_argument("-ls", "--logging_steps", help="number of steps between each evaluation", action="store", default=50, type=int)
parser.add_argument("-ck", "--checkpoint", help="directory where to find last checkpoint", action='store', default=None)
parser.add_argument("-lr", "--learning_rate", help="default learning rate", type=float, default=1e-4, action='store')
parser.add_argument("-bs", "--batch_size", help="batch size for training", type=int, default=16, action='store')
parser.add_argument("-ss", "--save_steps", help="number of gradient descent steps between each saving", type=int, default=400, action='store')
parser.add_argument("-ep", "--epochs", help="number of epochs for training", type=int, default=10, action='store')
parser.add_argument("-gs", "--gradient_accumulation_steps", help='number of steps before backward step', type=int, default=50, action='store')
parser.add_argument("-wd", "--weight_decay", help="weight decay parameter for training", type=float, default=1e-5, action='store')
parser.add_argument("-fb", "--file_bis", help="option to add name of piaf file (for fr model)", default="", action="store")


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
        config = EncoderDecoderConfig.from_json_file(args.checkpoint + "/config.json")
        model = EncoderDecoderModel.from_pretrained(args.checkpoint + "/pytorch_model.bin", config=config)

    if args.language == 'fr':
        if args.bart:
            model_name = "WikinewsSum/bart-large-multi-fr-wiki-news"
            #config = BartConfig.from_pretrained(model_name)
            tokenizer = BartTokenizer.from_pretrained(model_name, do_lower_case=True)
            model = BartForConditionalGeneration.from_pretrained(model_name)
            model_created = True
        else:
            model_name = "camembert-base"
            #config = CamembertConfig.from_pretrained(model_name)
            tokenizer = CamembertTokenizer.from_pretrained(model_name, do_lower_case=True)
        print("Model used:", model_name)
        # FQuAD
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
            model = BartForConditionalGeneration.from_pretrained(model_name)
            model_created = True
        else:
            model_name = 'bert-base-uncased'
            #config = BertConfig.from_pretrained(model_name)
            tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)
        print('Model used:', model_name)
        # SQuAD
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

    max_length_seq = 256
    max_length_label = 21

    answers_train = df_train["answer_span"]
    sentences_train = df_train["context"]
    labels_train = df_train["question"]
    t0 = time.time()
    print("Loading train dataset...", end='', flush=True)
    train_dataset = load_examples_question_generation(answers=answers_train,
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
    eval_dataset = load_examples_question_generation(answers=answers_eval,
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

    generation_dataset = load_examples_question_generation(answers=answers_eval[idx_examples].values,
                                                           sentences=sentences_eval[idx_examples].values,
                                                           labels=labels_eval[idx_examples].values,
                                                           tokenizer=tokenizer,
                                                           max_length_seq=max_length_seq,
                                                           max_length_label=max_length_label,
                                                           bart=args.bart,
                                                           )
    print("Done. {:.4f}s".format(time.time() - t0))

    # TRAINING
    #print('input_ids:', generation_dataset[0][0].tolist())
    #print('\nattention_mask:', generation_dataset[0][1])
    #print('\ntoken_type_ids:', generation_dataset[0][2])
    #print('\ndecoder_input_ids:', generation_dataset[0][3].tolist())
    #print('\ndecoder_attention_mask:', generation_dataset[0][4])
    #print('\nlabels:', generation_dataset[0][5].tolist())  # unknown token is due to -100

    train_question_generation(model=model,
                              train_dataset=train_dataset,
                              tokenizer=tokenizer,
                              num_train_epochs=args.epochs,
                              train_batch_size=args.batch_size,
                              max_length_label=max_length_label,
                              learning_rate=args.learning_rate,
                              device=device,
                              adam_epsilon=1e-8,
                              logging_steps=args.logging_steps,
                              logging_dir=args.output_dir,
                              gradient_accumulation_steps=args.gradient_accumulation_steps,
                              max_grad_norm=1.0,
                              weight_decay=1e-5,
                              warmup_steps=0,
                              output_dir=args.output_dir,
                              max_steps=-1,
                              num_cycles=7,
                              evaluate_during_training=True,
                              eval_dataset=eval_dataset,
                              eval_batch_size=args.batch_size,
                              generation_during_training=True,
                              generation_dataset=generation_dataset,
                              save_steps=args.save_steps,
                              verbose=1,
                              )