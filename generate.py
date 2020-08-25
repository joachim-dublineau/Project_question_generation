# coding: utf-8

######## Script: Using a QG model on a given dataset ########

# This python file contains a script that shows how to use an Hugging Face's model to generate questions.
# It can work on Insurance data or on FQuAD. For insurance it will return a json ready to fed the
# Zelros predict pipeline. For FQuAD, it will return the json file ready to be pass as argument to 
# run_squad.py script from Hugging Face.
# Attention: this script is supposed to work only for CamemBERT, T5 and BART other models were not tested.

# Author: Joachim Dublineau
# Zelros A.I.

# IMPORTS:
import pandas as pd

import time
import random
random.seed(2020)
import os
import regex as re
import torch
import tqdm

from transformers import (
    BertConfig,
    BertTokenizer,
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

from transformers_utils import (
    load_examples_question_generation, 
    generate_questions,
    load_json_QuAD_v1,
    )
import spacy
from natixis_utils import (
    DataPrepModelAssess,
    clean_dataframe, custom_tokenizer,
    select_keywords_spacy, 
    separating_keywords,
    highlight_answers,
    find_all,
    )
import argparse
import json

# ARGUMENT PARSING

parser = argparse.ArgumentParser()
# Compulsory arguments
parser.add_argument("file_data", help="name of the json file containing the contexts", type=str)
parser.add_argument("output_dir", help="name of the directory where to export the generated questions", type=str)
parser.add_argument("file_name", help="name of the output json file that will be saved", type=str)

# Optional arguments
parser.add_argument("-fq", "--is_fquad", help="boolean saying if the file is an fquad json file or not, default False", type=bool, default=False, action="store")
parser.add_argument("-bt", "--bart", help="true if bart else false, default False", type=bool, default=False, action="store")
parser.add_argument("-t5", "--t5", help='true if t5 else false, default False', type=bool, default=True, action="store")
parser.add_argument("-t5tp", "--t5_type", help="type of T5 model: multi or e2e, default multi", choices={"multi", "e2e"}, default="multi", action="store")
parser.add_argument("-pr", "--preprocessing", help="ae (answer extraction if model allows) or ke (keyword extraction with spacy), default ae", type=str, default="ae", choices={"ae", "ke"})
parser.add_argument("-rf", "--ref_file", help='file to use for non fquad type of dataset as a reference. .json', type=str, action="store")
parser.add_argument("-tk", "--tokenizer", help="name or path of where to find the tokenizer", type=str, default="", action="store")
parser.add_argument("-mi", "--max_length_input", help="max length of input sequence, default 256", type=int, default=512, action="store")
parser.add_argument("-mo", "--max_length_output", help="max_length of output sequence, defaut 21", type=int, default=21, action="store")
parser.add_argument("-ck", "--checkpoint", help="directory where to find the checkpoint of the model, default None", type=str, default=None, action='store')
parser.add_argument("-bs", "--batch_size", help="batch size for training, default 16", type=int, default=16, action='store')
parser.add_argument("-rp", "--repetition_penalty", help='repetition penalty parameter for generation, default 2', type=float, default=2.0, action="store")
parser.add_argument("-lp", "--length_penalty", help='length penalty parameter for generation, default 2', type=float, default=2.0, action="store")
parser.add_argument("-nb", "--num_beams", help="number of beams, parameter for generation, default 1", type=int, default=1, action="store")
parser.add_argument("-tp", "--temperature", help="temperature parameter for softmax in generation, default 1.0", type=float, default=1.0, action="store")
parser.add_argument("-csv", "--to_csv", help= "if the generated sentences need to be saved as csv (sep=_, encoding utf-8), default False", type=bool, default=False, action="store")

args = parser.parse_args()

if __name__ == "__main__":
    # LOADING MODEL & TOKENIZER:
    print(args)
    if args.t5_type == "e2e" and args.is_fquad == True:
        print("WARNING: e2e is meant to generate questions by context. The ouput of the script will be a csv instead of a json.") 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    model_created = False
    print("Loading model and tokenizer...", end="", flush=True)
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
        model_name = args.checkpoint

    if args.bart:
        model_name = "WikinewsSum/bart-large-multi-fr-wiki-news"
        # config = BartConfig.from_pretrained(model_name)
        tokenizer = BartTokenizer.from_pretrained(model_name, do_lower_case=True)
        if not model_created:
            model = BartForConditionalGeneration.from_pretrained(model_name)
            model_created = True

    if args.t5:
        if args.checkpoint == None:
            model_name = "airKlizz/t5-base-multi-fr-wiki-news"
        if args.tokenizer != "":
            tokenizer = T5Tokenizer.from_pretrained(args.tokenizer)
        else:
            print()
            print("Please provide a correct tokenizer for T5.")

    elif not args.bart and not args.t5:
        model_name = "camembert-base"
        tokenizer = CamembertTokenizer.from_pretrained(model_name)
        if not model_created:
            model = EncoderDecoderModel.from_encoder_decoder_pretrained(model_name, model_name)
    print("Done.")
    print("Model used:", model_name)

    model.to(device)

    # LOADING & PREPROCESSING DATA
    print("Loading and preprocessing data...", end="", flush=True)
    t0 = time.time()
    if args.is_fquad:
        df_generation = load_json_QuAD_v1(args.file_data)
        df_generation = df_generation.drop(columns=["question"])

    else:
        # PREPROCESSING NATIXIS CONTEXT
        file_data = args.file_data
        params = {'scenario_file': file_data}
        data_prep = DataPrepModelAssess(**params)
        df_generation = clean_dataframe(data_prep.df_scenario, "context")
        df_generation = df_generation.drop(columns=["questions", 'title'])
        if args.preprocessing == "ke":
            # EXTRACTING KEYWORDS
            if args.t5_type == "multi":
                nlp = spacy.load("fr_core_news_sm")
                nlp.tokenizer = custom_tokenizer(nlp)
                df_generation = select_keywords_spacy(df_generation, "context", 5, nlp)
                df_generation = separating_keywords(df_generation, "keywords") # df_generation columns=["id", "answer_span", "context"]

        else:
            # EXTRACTING ANSWERS
            if args.t5_type == "multi":
                print()                                                                                          
                print("ATTENTION: The model should be trained on answer extraction task to preprocess the data.")
                answers_span = []
                answers_span_to_add = []
                contexts = []
                for i, iterrow in enumerate(df_generation.iterrows()):
                    row = iterrow[1]
                    context = row["context"]
                    context = context[:-1]
                    df_generation.iloc[i].loc["context"] = context
                    try:
                        sentences = context.split(".")
                    except:
                        print('This context:', context[:50], "should contain at least a '.'")
                        input_model = "extract answers: " + "<hl> " + context + " <hl>"
                        continue
                    inputs_model = []
                    for i in range(0, len(sentences)):
                        inputs_model.append("extract answers: " + ". ".join(sentences[:i]) + (". <hl> " if i != 0 else "<hl> ") + \
                                            sentences[i] + ". <hl> " + ". ".join(sentences[i+1:]) + \
                                            ("." if i != len(sentences)-1 else ''))
                    inputs = tokenizer.batch_encode_plus(inputs_model, max_length= 512, truncation=True)
                    input_ids = torch.tensor(inputs["input_ids"]).to(device)
                    attention_mask = torch.tensor(inputs["attention_mask"]).to(device)
                    outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask)
                    spans = tokenizer.batch_decode(outputs)
                    for i in range(len(spans)):
                        spans[i] = spans[i].replace("<sep>", "")
                    answers_span += spans
                    answers_span_to_add.append(spans)
                    contexts += [context]*len(spans)
                    
                df_generation["answer_span"] = answers_span_to_add

    if args.t5_type == "e2e":    
        prev_context = df_generation.loc[0, "context"]
        contexts = [prev_context]
        for iterrow in df_generation.iterrows():
            row = iterrow[1]
            if row["context"] != prev_context:
                contexts.append(row["context"])
                prev_context = row["context"]
    print("Done. {:.2f}s".format(time.time() - t0))
    t0 = time.time()
    print(df_generation.head())

    # GENERATION
    print("Generating...")
    max_length_seq = args.max_length_input
    max_length_label = args.max_length_output

    if args.t5_type == "multi" and args.is_fquad == False:
        if args.preprocessing == "ke":
            answers = df_generation["answer_span"]
        else:
            answers = answers_span
    sentences = df_generation["context"]

    generation_hyperparameters = {
        'min_length': 5,
        'max_length': max_length_label,
        'repetition_penalty': args.repetition_penalty,
        'length_penalty': args.length_penalty,
        'num_beams': args.num_beams,
        'temperature': args.temperature,
    }

    def add_string(contexts, string):
        context_bis = []
        for text in contexts:
            context_bis.append(string + text)
        return context_bis

    if not args.t5:
        generation_dataset = load_examples_question_generation(
            answers=answers,
            sentences=sentences,
            tokenizer=tokenizer,
            max_length_seq=max_length_seq,
            max_length_label=max_length_label,
            bart=args.bart,
            )

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

    else:
        if args.t5_type == "multi":
            if args.is_fquad: nb_batch = len(sentences)//args.batch_size
            else:
                if args.preprocessing == "ae": nb_batch = len(contexts)//args.batch_size
                else: nb_batch = len(sentences)//args.batch_size
        else: nb_batch = len(contexts)//args.batch_size

        generated_questions = []

        if args.t5_type == "multi":
            if args.is_fquad == False:
                if type(answers) != type([0,1]):
                    list_answers = answers.tolist()
                else:
                    list_answers = answers
            else:
                list_answers = df_generation["answer_span"].tolist()

        list_contexts = sentences.tolist()
        for i in tqdm.tqdm(range(nb_batch)):
            if args.t5_type == "multi":
                if args.preprocessing == "ae" and args.is_fquad == False:
                    batch_contexts = contexts[i*args.batch_size:(i+1)*args.batch_size] if i != nb_batch - 1 else \
                        contexts[i*args.batch_size:]
                    batch_answers = answers[i*args.batch_size:(i+1)*args.batch_size] if i != nb_batch - 1 else \
                        answers[i * args.batch_size:]
                else:
                    batch_contexts = list_contexts[i*args.batch_size:(i+1)*args.batch_size] if i != nb_batch - 1 else \
                        list_contexts[i*args.batch_size:]
                    batch_answers = list_answers[i*args.batch_size:(i+1)*args.batch_size] if i != nb_batch - 1 else \
                        list_answers[i * args.batch_size:]
                batch_hl_contexts = highlight_answers(batch_answers, batch_contexts, "<hl>", "generate question: ")
            else:
                batch_contexts = contexts[i*args.batch_size:(i+1)*args.batch_size] if i != nb_batch - 1 else \
                    contexts[i*args.batch_size:]
                batch_hl_contexts = add_string(batch_contexts, "generate questions: ")
            if len(batch_hl_contexts) > 0:
                list_inputs = tokenizer.batch_encode_plus(batch_hl_contexts, padding=True, max_length=max_length_seq, truncation=True)
                input_ids = torch.tensor(list_inputs["input_ids"], ).to(device)
                attention_mask = torch.tensor(list_inputs["attention_mask"]).to(device)
                generation_hyperparameters["input_ids"] = input_ids
                generation_hyperparameters["attention_mask"] = attention_mask
                batch_generated_tokens = model.generate(**generation_hyperparameters)
                batch_generated_questions = tokenizer.batch_decode(batch_generated_tokens)
                if args.t5_type == "e2e":
                    for j in range(len(batch_generated_questions)):
                        generated = re.split("<[^<]*>", batch_generated_questions[j])[:-1]
                        batch_generated_questions[j] = generated
                generated_questions += batch_generated_questions
        
    if args.t5_type != "e2e" or args.is_fquad != True:
        df_generation["question"] = generated_questions
    print("Generated. {:.2f}s".format(time.time() - t0))

    if args.is_fquad: 
        idx_examples_eval = [1030, 649, 813, 3149] if args.t5_type == "multi" else [164, 207, 257, 760]
        print("Few samples:")
        for index in idx_examples_eval:  
            print(generated_questions[index])

    else:
        print("Few samples:")
        for i in range(5):
            print(generated_questions[i])

    # SAVING
    dict_to_save = {}
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    to_csv = False
    if args.is_fquad:
        if args.t5_type == "multi":
            data, paragraphs, qas = [], [], []

            dict_to_save['version'] = 1.1
            prev_context = df_generation.loc[0, "context"]
            prev_title = df_generation.loc[0, "doc_title"]
            for i, iterrow in enumerate(df_generation.iterrows()):
                row = iterrow[1]

                if row["context"] != prev_context:
                    paragraphs.append({
                        "context":prev_context,
                        "qas": qas
                        })
                    qas = []
                    prev_context = row["context"]
                if row["doc_title"] != prev_title:
                    theme = {
                        "title": prev_title,
                        "paragraphs": paragraphs,
                        }
                    data.append(theme)
                    prev_title = row["doc_title"]
                    paragraphs = []
                qas.append({
                    "answers": [{
                        "answer_start": int(row["answer_span_start"]),
                        "text": row["answer_span"]
                        }],
                    "question": row["question"],
                    "id": str(row["id_question"]),
                    })

            paragraphs.append({
                "context": prev_context,
                "qas": qas
                })
            theme = {
                "title": prev_title,
                "paragraphs": paragraphs,
                }
            data.append(theme)

            dict_to_save['data'] = data
        else:
            to_csv = True

    if args.to_csv or to_csv:
        if args.t5_type == "multi":
            df_generation.to_csv(os.path.join(args.output_dir, args.file_name[:list(find_all(args.file_name, "."))[-1]] + ".csv"), sep="_", encoding="utf-8")
        else:
            df_ = pd.DataFrame({"context": contexts, "questions": generated_questions})
            df_.to_csv(os.path.join(args.output_dir, args.file_name[:list(find_all(args.file_name, "."))[-1]] + ".csv"), sep="_", encoding="utf-8")
            
    if args.is_fquad == False:
        # NATIXIS TO JSON (questions, context, name, id, tags, confidentiality)
        if args.t5_type == "multi":
            contexts_new, questions_new, temp_questions = [], [], []
            prev_context = df_generation.iloc[0]["context"]
            for iterrow in df_generation.iterrows:
                row = iterrow[1]
                if row["context"] != prev_context:
                    questions_new.append(temp_questions)
                    contexts_new.append(prev_context)
                    prev_context = row["context"]
                    temp_questions = []
                temp_questions.append(row["question"])
            questions_new.append(temp_questions)
            contexts_new.append(row["context"])
            df_generation = pd.DataFrame({"context": contexts_new, "questions": questions_new})
        else:
            df_generation = pd.DataFrame({"context": contexts, "questions": generated_questions})
                
        intents = []
        file_ref = open(args.ref_file, 'r')
        dict_ref = json.load(file_ref)
        dict_ = {'nlp': dict_ref['nlp'], "datasources": [], "entities": dict_ref["entities"], "modules": []}
        for i, iterrow in enumerate(df_generation.iterrows()):
            try:
                corresponding_intent = dict_ref["intents"][i]
                row = iterrow[1]
                context = row["context"]
                questions = row["questions"]

                confidentiality = "" if "confidentiality" not in corresponding_intent['details'].keys() else \
                    corresponding_intent['details']['confidentiality']

                answers = [{
                    'tips':[],
                    'texts':[context], # est ce que la division en sous phrases est importante ou est-ce simplement pour l'affichage.
                    'next_actions':[], # est ce important pour l'entraînement
                    }]
                details_dict = {
                    'answers':answers, 
                    'tags': corresponding_intent['details']['tags'], 
                    'confidentiality': confidentiality, 
                    'sentences': questions,
                    }
                row_dict = {
                    "bot_id": '',
                    'id':corresponding_intent['id'],
                    'name':corresponding_intent['name'],
                    'title':'',
                    "created_at": "2017-07-17T10:15:13.657Z",
                    'details': details_dict,
                    }
                intents.append(row_dict)
            except:
                continue
        dict_to_save['intents'] = intents

    if args.t5_type != "e2e" or args.is_fquad == False:
        with open(os.path.join(args.output_dir, args.file_name), "w") as f:
            json.dump(dict_to_save, f)

