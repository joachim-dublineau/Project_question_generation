# Project_question_generation

## Question Generation with Transformers on SQuAD and FQuAD

**Zelros A.I.** 
<p align="center">
  <img width="30%", src="https://www.actuia.com/wp-content/uploads/2018/12/zelros-696x348.png">
</p>

This repository contains a script able to train an EncoderDecoderModel, a BART model of a T5 model from Hugging Face's transformers library (https://github.com/huggingface/transformers). They can be trained with different methods: on question generation task only, on multi tasks (QG, QA and answer extraction) or on end-to-end question generation (one context --> * questions). Scripts for evaluation as well as for generation are also available.


### Train a model on QG task only
The main.py file trains a model on Question Generation task only.
It works with BART, BERT and CamemBERT only and it uses SQuAD (https://rajpurkar.github.io/SQuAD-explorer/dataset/) and FQuAD (https://storage.googleapis.com/illuin/fquad/train.json.zip dataset type for training.
 https://storage.googleapis.com/illuin/fquad/valid.json.zip) to train these previous models.
 
In order to make the evaluate function works, please make sure to install this library: https://github.com/Maluuba/nlg-eval .

Example with only positional arguments:
```
python main.py en train-v2.0.json dev-v2.0.json model_Bert2Bert
```

All arguments:

positional arguments:

- {en,fr}: en or fr
- file_train: name of the train file
- file_test: name of the test file
- output_dir: name of the directory for logs and checkpoints

optional arguments:

-  -h, --help show this help message and exit
-  -csv, --is_csv: true if file_train is a csv file, default False
-  -bt, --bart BART: true if bart else false, default False
-  -ls, --logging_steps LOGGING_STEPS: number of steps between each evaluation, default 50
-  -mi, --max_length_input MAX_LENGTH_INPUT: max length of input sequence, default 256
-  -mo, --max_length_output MAX_LENGTH_OUTPUT: max_length of output sequence, defaut 21
-  -ck, --checkpoint CHECKPOINT: directory where to find last checkpoint, default None
-  -lr, --learning_rate LEARNING_RATE: default learning rate, default 1e-4
-  -bs, --batch_size BATCH_SIZE: batch size for training, default 16
-  -ss, --save_steps SAVE_STEPS: number of gradient descent steps between each saving, default 400
-  -ep, --epochs EPOCHS: number of epochs for training, default 10
-  -gs, --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS: number of steps before backward step, default 50
-  -wd, --weight_decay WEIGHT_DECAY: weight decay parameter for training, default 1e-5
-  -fb, --file_bis FILE_BIS: option to add name of piaf file (for fr model), default None
-  -rp, --repetition_penalty REPETITION_PENALTY: repetition penalty parameter for generation, default 2
-  -lp, --length_penalty LENGTH_PENALTY: length penalty parameter for generation, default 2
-  -nb, --num_beams NUM_BEAMS: number of beams, parameter for generation, default 1
-  -tp, --temperature TEMPERATURE: temperature parameter for softmax in generation, default 1.0
-  -eo, --evaluate_on EVALUATE_ON: number of examples on which to evaluate the model, default 100

### Train on multi task or e2e
In order to train a transformers model on multi task or on end-to-end task, we use a script adapated from Patil-Suraj repository (see credits): run_qg_multi or run_qg_e2e. The training informations are given in this wandb folder: https://app.wandb.ai/joachim_dublineau/question_generation_french . An Wandb account is necessary to make this file work without any change.


For e2e:

```bash
# data preprocessing
python data/fquad_multitask/fquad_multitask.py

python prepare_data.py --task e2e_qg --valid_for_qg_only --model_type t5 --dataset_path data/fquad_multitask \
--qg_format highlight_qg_format --max_source_length 512 --max_target_length 32 --train_file_name train_data_e2e_qg_t5.pt \
--valid_file_name valid_data_e2e_qg_t5.pt

# training
python run_qg_e2e.py --model_name_or_path airKlizz/t5-base-multi-fr-wiki-news --model_type t5 \
--tokenizer_name_or_path t5_qg_tokenizer --output_dir ../t5-fr-e2e-hl/run0 --train_file_path data/mix_train_data_e2e_t5.pt \
--valid_file_path data/fquad_valid_data_e2e_t5.pt --per_device_train_batch_size 3 --per_device_eval_batch_size 3 \
--gradient_accumulation_steps 64 --learning_rate 1e-4 --num_train_epochs 20 --seed 42 --do_train --do_eval \
--evaluate_during_training --logging_steps 20 --eval_steps 50 
```



### Eval
This script is taken from patil-suraj repository (see credits).

optional arguments:
- -h, --help            show this help message and exit
- --model_name_or_path MODEL_NAME_OR_PATH Path to pretrained model or model identifier from huggingface.co/models
- --valid_file_path VALID_FILE_PATH Path for cached valid dataset
- --model_type MODEL_TYPE One of 't5', 'bart'
- --tokenizer_name_or_path TOKENIZER_NAME_OR_PATH Pretrained tokenizer name or path if not the same as model_name
- --num_beams NUM_BEAMS num_beams to use for decoding
- --max_decoding_length MAX_DECODING_LENGTH maximum length for decoding
- --output_path OUTPUT_PATH path to save the generated questions.
- --batch_size BATCH_SIZE batch size for eval.

### Generate with a model
It is also possible to generate questions based on a dataset that only contains contexts. In order to do so, the script
generate.py should be used.
You will also need to configure spacy with:
```python -m spacy download fr_core_news_sm```


```
python generate.py fquad_valid.json generated_questions generated_questions.json -fq -t5 -t5tp multi -ck my_multi_model -tk t5_qg_tokenizer
```

All arguments:
positional arguments:
- file_data             name of the json file containing the contexts
- output_dir            name of the directory where to export the generated questions
- file_name             name of the output json file that will be saved

optional arguments:
- -h --help            show this help message and exit
- -q --is_fquad IS_FQUAD boolean saying if the file is an fquad json file or not, default False
- -bt --bart BART true if bart else false, default False
- -t5 --t5 T5 true if t5 else false, default False
- -t5tp --t5_type {multi,e2e} type of T5 model: multi or e2e, default multi
- -pr --preprocessing {ke,ae} ae (answer extraction if model allows) or ke (keyword extraction with spacy), default ae
- -rf --ref_file REF_FILE file to use for non fquad type of dataset as a reference. .json (ZELROS ONLY)
- -tk --tokenizer TOKENIZER name or path of where to find the tokenizer
- -mi --max_length_input MAX_LENGTH_INPUT max length of input sequence, default 256
- -mo --max_length_output MAX_LENGTH_OUTPUT max_length of output sequence, defaut 21
- -ck --checkpoint CHECKPOINT directory where to find the checkpoint of the model, default None
- -bs --batch_size BATCH_SIZE batch size for training, default 16
- -rp --repetition_penalty REPETITION_PENALTY repetition penalty parameter for generation, default 2
- -lp --length_penalty LENGTH_PENALTY length penalty parameter for generation, default 2
- -nb --num_beams NUM_BEAMS number of beams, parameter for generation, default 1
- -tp --temperature TEMPERATURE temperature parameter for softmax in generation, default 1.0
- -csv --to_csv TO_CSV if the generated sentences need to be saved as csv (sep=_, encoding utf-8), default False
- -lg --language {en,fr} en or fr, default fr
- -mn --model_name MODEL_NAME name of the model if no checkpoint, default None

                        
### Credits:
Big thanks and credits to https://github.com/patil-suraj/question_generation for the multi task training code.
