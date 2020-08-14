# coding: utf-8

######## Script: Using a QA model to test a given dataset ########

# This python file contains a script that evaluates the QA performances of a model on a given dataset with generated questions.

# Author: Joachim Dublineau
# Zelros A.I.

from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer)

import pandas as pd
import argparse


# ARGUMENT PARSING

parser = argparse.ArgumentParser()
# Compulsory arguments
parser.add_argument("file_data", help="name of the .csv file containing the data for question answering evaluation")
parser.add_argument("language", choices=["en", 'fr'], help="fr or en")

args = parser.parse_args()

# LOADING MODEL & TOKENIZER

if args.language == "en":
    tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
    model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")

else:
    tokenizer = AutoTokenizer.from_pretrained("illuin/camembert-large-fquad")
    model = AutoModelForQuestionAnswering.from_pretrained("illuin/camembert-large-fquad")

# LOADING DATA

df_data = pd.read_csv(args.file_data)
print(df_data.head())