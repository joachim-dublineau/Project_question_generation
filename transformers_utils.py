# coding: utf-8

######## Hugging Face Utils repository for transformers model training ########

# This python file contains useful functions that will be used to train transformers
# models like BERT and CamemBERT with EncoderDecoderModel. They might not work in another
# framework.

# Author: Joachim Dublineau
# Zelros A.I.

# IMPORTS
import pandas as pd
import numpy as np
import os
from tqdm.notebook import tqdm
import json
import time
import random
random.seed(2020)

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from transformers.modeling_bart import shift_tokens_right

from transformers import (AdamW,
                          get_linear_schedule_with_warmup,
                          get_cosine_with_hard_restarts_schedule_with_warmup)
from nltk.translate.bleu_score import sentence_bleu


# UTILS FOR PROCESSING
class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.answer = text_a
        self.context = text_b
        self.question = label


class InputFeatures(object):
    def __init__(self,
                 input_ids,
                 attention_mask,
                 token_type_ids,
                 label,
                 decoder_input_ids,
                 decoder_attention_mask,
                 ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label
        self.decoder_input_ids = decoder_input_ids
        self.decoder_attention_mask = decoder_attention_mask


class DataProcessor():

    def get_data_examples(self, answers, sentences, labels=None):
        examples = []
        guid = "data"
        if labels is not None:
            if len(answers) != len(sentences) or len(answers) != len(labels):
                print("Error: sentences, answers, labels should have same length.")
                return None
            for i in range(len(sentences)):
                examples.append(InputExample(guid=guid,
                                             text_a=answers[i],
                                             text_b=sentences[i],
                                             label=str(labels[i]),
                                             )
                                )
        else:
            if len(answers) != len(sentences):
                print("Error: sentences, answers should have same length.")
                return None
            for i in range(len(sentences)):
                examples.append(InputExample(guid=guid,
                                             text_a=answers[i],
                                             text_b=sentences[i],
                                             )
                                )
        return examples


def convert_examples_to_features_question_generation(
        examples,
        tokenizer,
        max_length=512,
        max_length_label=32,
        bart=False,
        ):
    """
    This function converts a list of examples into features that can be used
    as inputs for the question generation model.
    INPUTS:
    - examples: list of object <class: InputExample>, examples to convert.
    - tokenizer: torch tokenizer object, tokenize the examples.
    - max_length: int, size of the maximum input sentence (list of tokens).
    - max_length_label: int, size of the maximum label sentence.
    - bart: boolean, saying whether the model is BART or not.
    OUTPUTS:
    - features: list of object <class: InputFeatures>, list of features for model.
    """
    processor = DataProcessor()
    features = []
    pad_token = tokenizer.pad_token_id
    for (ex_index, example) in enumerate(examples):
        ######## ENCODING INPUT ########
        # This will encode both answer and context with a separator.
        inputs = tokenizer.encode_plus(
            example.answer,
            example.context,
            add_special_tokens=True,
            max_length=max_length,
            truncation='only_second'
            )
        input_ids = inputs["input_ids"]
        token_type_ids = [0] * (len(tokenizer.encode(example.answer)) + 1)
        token_type_ids += [1] * (len(input_ids) - len(token_type_ids))

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1] * len(input_ids)

        ######## PADDING ########
        padding_length = max_length - len(input_ids)
        input_ids = input_ids + [pad_token] * padding_length
        attention_mask = attention_mask + [0] * padding_length
        token_type_ids = token_type_ids + [0] * padding_length

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask),
                                                                                            max_length)
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids),
                                                                                            max_length)


        ######## ENCODING LABEL ########
        if example.question is not None:
            if bart:
                label = tokenizer.encode_plus(
                    example.question,
                    max_length=max_length_label,
                    truncation=True,
                    )
                label_ids = label["input_ids"]
                padding_length = max_length_label - len(label_ids)
                label_ids = label_ids + [-100] * padding_length
                decoder_input_ids = shift_tokens_right(torch.tensor(label_ids).unsqueeze(0), -100).squeeze(0).tolist()
                decoder_input_ids = [x if x != -100 else pad_token for x in decoder_input_ids]
            else:
                label_ids = tokenizer.encode(
                    example.question,
                    add_special_tokens=True,
                    max_length=max_length_label,
                    truncation=True,
                    )

                decoder_input_ids = label_ids
                padding_length = max_length_label - len(label_ids)
                label_ids = label_ids + [-100] * padding_length
                decoder_input_ids = decoder_input_ids + [pad_token] * padding_length

            decoder_attention_mask = [1] * max_length_label

            assert len(label_ids) == max_length_label, "Error with input length {} vs {}".format(len(input_ids),
                                                                                                 max_length)
            assert len(decoder_input_ids) == max_length_label, "Error with input length {} vs {}".format(
                len(decoder_input_ids), max_length_label)
            assert len(decoder_attention_mask) == max_length_label, "Error with input length {} vs {}".format(
                len(decoder_attention_mask), max_length_label)

            features.append(InputFeatures(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                label=label_ids,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                )
                )
        else:
            features.append(InputFeatures(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                label=None,
                decoder_input_ids=None,
                decoder_attention_mask=None,
                )
                )
    return features


def load_examples_question_generation(
        answers,
        sentences,
        tokenizer,
        max_length_seq,
        max_length_label,
        labels=None,
        bart=False,
        ):
    """
    This function will creates features from set of answers, sentences and labels.
    INPUTS:
    - answers: list of strings, containing the answers to the question.
    - sentences: list of strings, containing the context to answer the question.
    - labels: list of strings, containing the text of the questions.
    - tokenizer: torch tokenizer object, tokenizer used for analysis.
    - max_seq_length: int, max length of the input sequence.
    - max_length_label: int, max length of the output question.
    - bart: boolean, saying whether the model we prepare the data for is BART or not.
    OUTPUTS:
    - dataset: torch TensorDataset, dataset containing input_ids, att_masks,
    token_type_ids, labels.
    """
    processor = DataProcessor()
    examples = processor.get_data_examples(answers, sentences, labels)

    features = convert_examples_to_features_question_generation(
        examples,
        tokenizer,
        max_length=max_length_seq,
        max_length_label=max_length_label,
        bart=bart,
        )

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if labels is not None:
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        all_decoder_input_ids = torch.tensor([f.decoder_input_ids for f in features], dtype=torch.long)
        all_decoder_attention_mask = torch.tensor([f.decoder_attention_mask for f in features], dtype=torch.long)
        dataset = TensorDataset(
            all_input_ids,
            all_attention_mask,
            all_token_type_ids,
            all_decoder_input_ids,
            all_decoder_attention_mask,
            all_labels,
            )
    else:
        dataset = TensorDataset(
            all_input_ids,
            all_attention_mask,
            all_token_type_ids,
            )
    return dataset

# UTILS FOR TRAINING

def train_question_generation(
        model,
        train_dataset,
        tokenizer,
        num_train_epochs,
        train_batch_size,
        learning_rate,
        device,
        adam_epsilon=1e-8,
        logging_steps=None,
        logging_dir=None,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        weight_decay=0.0,
        warmup_steps=0,
        output_dir=None,
        max_steps=-1,
        num_cycles=1.0,
        evaluate_during_training=False,
        eval_dataset=None,
        eval_batch_size=8,
        generation_during_training=False,
        generation_dataset=None,
        generation_hyperparameters=None,
        save_steps=-1,
        verbose=0,
        ):
    """
    This function trains models on the train_dataset, eval_dataset being
    optional.
    INPUTS:
    - model: Torch model, model to train.
    - train_dataset: Torch TensorDataset, used for training.
    - tokenizer: Torch tokenizer object, tokenizer used for preprocessing.
    - num_train_epochs: int, number of epochs for training.
    - train_batch_size: int, size of mini batch.
    - learning_rate: int, learning rate.
    - device: torch cuda object, describing the device on which the training will be done.
    - adam_epsilon: float, epsilon parameter for optimizer AdamW.
    - logging_steps: float, number of steps for evaluation.
    - logging_dir: str, name of the directory in which the logs will be written.
    - gradient_accumulation_steps: int, number of step before retropropagation.
    - max_grad_norm: float, maximum norm for gradient.
    - weights_decay: float, weights_decay parameter for optimizer.
    - warmup_steps: int, number of steps used for warmup.
    - output_dir: str, directory to save output.
    - max_steps: int, maximum number of step per epochs, -1 for None.
    - num_cycles: float, number of cycle for warmup.
    - evaluate_during_training: bool, saying whether to evaluate.
    - eval_dataset: Torch TensorDataset, to provide for evaluation.
    - eval_batch_size: int, batch size for evaluation dataset.
    - generation_during_training: bool, saying whether to generate some question as examples.
    - generation_dataset: TensorDataset, will be used for generation in generation_during_training=True.
    - generation_hyperparameters: dictionary, containing hyperparameters used for generation.
    - save_steps; int, number of steps between each checkpoint.
    - verbose: int, 0 for no verbose, 1 for displaying.
    OUTPUTS:
    - train_loss_history: list of floats, loss history.
    - val_loss_history: list of floats, validation loss history.
    """

    train_loss = []

    assert not (logging_steps > 0 and eval_dataset is None), "logging_steps > 0 but no eval_dataset provided"

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=train_batch_size
        )

    if logging_steps is None:
        logging_steps = len(train_dataloader) // (gradient_accumulation_steps * 5)

    if max_steps > 0:
        t_total = max_steps
        num_train_epochs = max_steps // (len(train_dataloader) // gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // gradient_accumulation_steps * num_train_epochs

    no_decay = ['bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [{'params': [p for n, p in model.named_parameters() \
                                                if not any(nd in n for nd in no_decay)],
                                     'weight_decay_rate': weight_decay},
                                    {'params': [p for n, p in model.named_parameters() \
                                                if any(nd in n for nd in no_decay)],
                                     'weight_decay_rate': 0.0}
                                    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=learning_rate,
                      eps=adam_epsilon,
                      )
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=t_total,
        num_cycles=num_cycles,
        )
    # Train
    print("***** Running training *****")
    print("  Num examples = %d" % len(train_dataset))
    print("  Num Epochs = %d" % num_train_epochs)
    print("  Batch size = %d" % train_batch_size)
    print("  Total train batch size (w. parallel, distributed & accumulation) = %d" %
          (train_batch_size * gradient_accumulation_steps))
    print("  Gradient Accumulation steps = %d" % gradient_accumulation_steps)
    print("  Total optimization steps = %d" % t_total)

    if logging_dir is not None:
        if not os.path.exists(logging_dir):
            os.makedirs(logging_dir)
        logging_file = os.path.join(logging_dir, "logs.txt")
        with open(logging_file, "w") as writer:
            writer.write("***** Running training *****\n")
            writer.write("  Num examples = %d\n" % len(train_dataset))
            writer.write("  Num Epochs = %d\n" % num_train_epochs)
            writer.write("  Batch size = %d\n" % train_batch_size)
            writer.write("  Total train batch size (w. parallel, distributed & accumulation) = %d\n" %
                         (train_batch_size * gradient_accumulation_steps))
            writer.write("  Gradient Accumulation steps = %d\n" % gradient_accumulation_steps)
            writer.write("  Total optimization steps = %d\n" % t_total)
            writer.write("\n")

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()

    train_loss_history = []
    val_loss_history = []

    start_train = time.time()
    for epoch in range(num_train_epochs):

        print(f"Epoch: {epoch + 1} / {num_train_epochs}")
        start_epoch = time.time()

        epoch_iterator = train_dataloader
        for step, batch in enumerate(epoch_iterator):
            if global_step > t_total:
                break

            ######## TRAINING STEP ########
            model.train()

            # Transfer input data to device
            batch = tuple(t.to(device) for t in batch)

            inputs = {'input_ids'               : batch[0],
                      'attention_mask'          : batch[1],
                      # 'token_type_ids'        : batch[2],
                      'decoder_input_ids'       : batch[3],
                      'decoder_attention_mask'  : batch[4],
                      'labels'                  : batch[5],
                      }

            optimizer.zero_grad()

            outputs = model(**inputs)
            loss = outputs[0]

            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()
            train_loss.append(loss.item())

            ######## LOGGING RESULTS ########
            if (step + 1) % gradient_accumulation_steps == 0:

                accumulated_loss = np.sum(train_loss[-gradient_accumulation_steps:])

                if verbose > 0:
                    print("lr: {:.10f}".format(scheduler.get_lr()[0]),
                          "loss: {:.6f}".format(accumulated_loss),
                          " -- step:", global_step, "/", t_total)

                if logging_dir is not None:
                    with open(logging_file, "a") as writer:
                        writer.write(" ".join(["lr: {:.10f}".format(scheduler.get_lr()[0]),
                                               "loss: {:.6f}".format(accumulated_loss),
                                               " -- step:", str(global_step), "/", str(t_total), "\n"]))

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if (logging_steps > 0 and global_step > 0) and (global_step % logging_steps == 0):
                    print("\nEval")
                    if evaluate_during_training:
                        # Log metrics
                        dict_print = {'step': global_step,
                                      'lr': scheduler.get_lr()[0],
                                      'tr_loss': (tr_loss - logging_loss) / logging_steps}
                        result_eval = evaluate_question_generation(
                            model=model,
                            eval_dataset=eval_dataset,
                            tokenizer=tokenizer,
                            device=device,
                            eval_batch_size=eval_batch_size,
                            generation=generation_during_training,
                            generation_dataset=generation_dataset,
                            generation_hyperparameters=generation_hyperparameters,
                            logging_dir=logging_dir,
                            verbose=1,
                            )
                        for key, value in result_eval.items():
                            dict_print['eval_{}'.format(key)] = value
                        train_loss_history.append((tr_loss - logging_loss) / logging_steps)
                        val_loss_history.append(result_eval['val_loss'])
                    print('Evaluation:', dict_print)
                    logging_loss = tr_loss

                    if logging_dir is not None:
                        with open(logging_file, "a") as writer:
                            writer.write("\nEval\n")
                            for key in sorted(dict_print.keys()):
                                writer.write("  %s = %s\n" % (key, str(dict_print[key])))
                            writer.write("\n")

                ######## SAVING MODEL ########
                if (save_steps > 0 and global_step > 0) and (global_step % save_steps == 0):
                    print("\nSave")
                    # Save model checkpoint
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    save_model_dir = os.path.join(output_dir, 'checkpoint-{}'.format(global_step))
                    os.makedirs(save_model_dir)
                    model.save_pretrained(save_model_dir)
                    print("Saving model checkpoint to %s" % save_model_dir)

                    if logging_dir is not None:
                        with open(logging_file, "a") as writer:
                            writer.write("\nSave\n")
                            writer.write("Saving model checkpoint to %s\n\n" % save_model_dir)

        end_epoch = time.time()
        print(f'Epoch {epoch + 1}/{num_train_epochs}, time = {end_epoch - start_epoch} secs')
    end_train = time.time()
    print("Train took:", end_train - start_train)

    return train_loss_history, val_loss_history


def evaluate_question_generation(
        model,
        eval_dataset,
        tokenizer,
        device,
        eval_batch_size=8,
        generation=False,
        generation_dataset=None,
        generation_hyperparameters=None,
        logging_dir=None,
        verbose=1,
        ):
    """
    This function evaluates the loss and accuracy of the model
    on the evaluation set.
    INPUTS:
    - model: Torch Hugging Face model, model to train.
    - eval_dataset: Torch TensorDataset, dataset for evaluation.
    - tokenizer: Torch tokenizer object, tokenizer for preprocessing.
    - device: Torch cuda object, describing the device on which the evaluation will be done.
    - eval_batch_size: int, size of mini-batch.
    - eval_output_dir: string, directory to save results.
    - generation: boolean, saying whether to generate.
    - generation_dataset: Torch TensorDataset, used for generation.
    - generation_hyperparameters: dictionary, hyperparameters for generation.
    - logging_dir: str, name of the directory where to write the logs.
    - verbose: int, 0 for no verbose, 1 for displaying.
    OUTPUTS:
    - results: dictionnary, containing val_loss.
    """
    evaluation_loss = []

    eval_batch_size = eval_batch_size
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=eval_batch_size,
        )

    # Eval
    if verbose > 0:
        print("***** Running evaluation *****")
        print("  Num examples = %d", len(eval_dataset))
        print("  Batch size = %d", eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    iterator = tqdm(eval_dataloader, desc="Evaluating") if verbose > 0 else eval_dataloader

    num_examples = 0
    for batch in iterator:
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids'               : batch[0],
                      'attention_mask'          : batch[1],
                      # 'token_type_ids'        : batch[2],
                      'decoder_input_ids'       : batch[3],
                      'decoder_attention_mask'  : batch[4],
                      'labels'                  : batch[5],
                      }
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            eval_loss += tmp_eval_loss.mean().item()

            evaluation_loss.append(tmp_eval_loss.mean().item())

        nb_eval_steps += 1
        num_examples += eval_batch_size
    eval_loss = eval_loss / nb_eval_steps

    result = {"val_loss": eval_loss}

    ######## GENERATION ########
    if generation:
        questions = generate_questions(
            model,
            generation_dataset,
            tokenizer,
            device,
            batch_size=1,
            generation_hyperparameters=generation_hyperparameters,
            )

        print('Examples:')
        for question in questions:
            print(question[0][0])
            # print("Target:", question[1])
        print('\n')

        if logging_dir is not None:
            logging_file = os.path.join(logging_dir, "logs.txt")
            with open(logging_file, "a") as writer:
                writer.write("\nExamples:\n")
                for question in questions:
                    writer.write(question[0][0])
                    writer.write("\n")
                writer.write("\n")

    return result


def generate_questions(
        model,
        dataset,
        tokenizer,
        device,
        batch_size,
        generation_hyperparameters,
):
    """
    This function generates the question with the model on the given dataset.
    INPUTS:
    - model: Torch Hugging Face model, model to use.
    - dataset: Torch TensorDataset, dataset for prediction.
    - tokenizer: Torch tokenizer object.
    - device: Torch cuda object, describing the device on which the calculation will be done.
    - batch_size: int, size of mini-batch.
    - generation_hyperparameters: dictionary, hyperparameters for generation.
    OUTPUTS:
    - results: list of string, predictions.
    """
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    iterator = dataloader
    results_generation = []
    is_labeled = -1
    for batch in iterator:
        results = []
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        if is_labeled == -1: is_labeled = (len(batch) == 6)
        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      # 'token_type_ids'        : batch[2],
                      }
            inputs.update(generation_hyperparameters)
            outputs = model.generate(**inputs)

        for output in outputs:
            results.append(tokenizer.decode(
                output.squeeze(0),
                skip_special_tokens=True,
                )
                )
        if is_labeled:
            labels = []
            for label in batch[5]:
                label[label == -100] = tokenizer.pad_token_id
                labels.append(tokenizer.decode(
                    label.squeeze(0),
                    skip_special_tokens=True,
                    )
                    )

            results_generation.append((results, labels))
        else:
            results_generation.append(results)
    return results_generation

# UTILS FOR DATA LOADING

def load_json_QuAD_v2(file_name):
    questions_list = []
    id_question_list = []
    answer_span_list = []
    answer_span_start_list = []
    id_context_list = []
    contexts_list = []
    doc_titles_list = []

    with open(file_name) as json_file:
        train_file = json.load(json_file)
        for data in train_file['data']:
            for id_paragraph, paragraph in enumerate(data['paragraphs']):
                for qa in paragraph['qas']:
                    is_impossible = qa['is_impossible']
                    if not is_impossible:
                        doc_titles_list.append(data['title'])
                        questions_list.append(qa['question'])
                        id_question_list.append(qa['id'])
                        answer_span_list.append(qa['answers'][0]['text'])
                        answer_span_start_list.append(qa['answers'][0]['answer_start'])
                        contexts_list.append(paragraph['context'])
                        id_context_list.append(id_paragraph)

    df_ = pd.DataFrame(
        {'id_question': id_question_list,
        'question': questions_list,
        'answer_span': answer_span_list,
        'answer_span_start': answer_span_start_list,
        'id_context': id_context_list,
        'context': contexts_list,
        'doc_title': doc_titles_list
        })
    return df_


def load_json_QuAD_v1(file_name):
    questions_list = []
    id_question_list = []
    answer_span_list = []
    answer_span_start_list = []
    id_context_list = []
    contexts_list = []
    doc_titles_list = []

    with open(file_name) as json_file:
        train_file = json.load(json_file)
        for data in train_file['data']:
            for id_paragraph, paragraph in enumerate(data['paragraphs']):
                for qa in paragraph['qas']:
                    doc_titles_list.append(data['title'])
                    questions_list.append(qa['question'])
                    id_question_list.append(qa['id'])
                    answer_span_list.append(qa['answers'][0]['text'])
                    answer_span_start_list.append(qa['answers'][0]['answer_start'])
                    contexts_list.append(paragraph['context'])
                    id_context_list.append(id_paragraph)

    df_ = pd.DataFrame(
        {'id_question': id_question_list,
        'question': questions_list,
        'answer_span': answer_span_list,
        'answer_span_start': answer_span_start_list,
        'id_context': id_context_list,
        'context': contexts_list,
        'doc_title': doc_titles_list
        })
    return df_

# UTILS FOR METRIC

def retrieval_score(generated_sequences, labels_sequences):
    """
    This function compute the retrieval score based on the bleu score.
    INPUTS:
    - generated_sequences: list, list of list of strings of the words of each generated sequence.
    - labels_sequences: list, list of list of string of the words of each label.
    OUTPUS:
    - retrieval_score: float, retrieval score as described in the report.
    - bleu_score: float, average BLEU score
    """
    n = len(generated_sequences)
    bleu_scores = np.zeros((n, n), dtype=np.float)
    retrieval_score = 0
    for i in range(n):
        for j in range(n):
            bleu_scores[i, j] = sentence_bleu(
                references=[labels_sequences[j]],
                hypothesis=generated_sequences[i],
                weights={1, 1, 0, 0},
                )
        argsort = np.argsort(-bleu_scores[i, :])
        index = np.where(argsort == i)[0][0]
        retrieval_score += index
    return 1 - retrieval_score/n**2, np.mean(np.diag(bleu_scores))
