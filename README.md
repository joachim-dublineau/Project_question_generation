# Project_generation

## Question Generation with Transformers on SQuAD and FQuAD

**Zelros A.I.** 
<p align="center">
  <img width="30%", src="https://www.actuia.com/wp-content/uploads/2018/12/zelros-696x348.png">
</p>


### Train a model
This repository contains some functions and a script able to train an EncoderDecoderModel or a BART model from Hugging Face's transformers library (https://github.com/huggingface/transformers).

It works with BART, BERT and CamemBERT and it uses SQuAD (https://rajpurkar.github.io/SQuAD-explorer/dataset/) and FQuAD (https://storage.googleapis.com/illuin/fquad/train.json.zip
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

### Generate with a model
It is also possible to generate questions based on a dataset that only contains contexts. In order to do so, the script
generate.py should be used.
You will also need to configure spacy with:
```python -m spacy download fr_core_news_sm```


```
python generate.py scenario_natixis_prod.json generated_questions
```

All arguments:

positional arguments:
- file_data: name of the file containing the contexts
- output_dir: name of the directory where to export the generated questions
- file_name: name of the csv file that will be saved

optional arguments

- -bt, --bart BART: true if bart else false, default False
- -mi, --max_length_input MAX_LENGTH_INPUT: max length of input sequence, default 256
- -mo, --max_length_output MAX_LENGTH_OUTPUT: max_length of output sequence, defaut 21
- -ck, --checkpoint CHECKPOINT: directory where to find the checkpoint of the model, default None
- -bs, --batch_size BATCH_SIZE: batch size for training, default 16
- -rp, --repetition_penalty REPETITION_PENALTY: repetition penalty parameter for generation, default 2
- -lp, --length_penalty LENGTH_PENALTY: length penalty parameter for generation, default 2
- -nb, --num_beams NUM_BEAMS: number of beams, parameter for generation, default 1
- -tp, --temperature TEMPERATURE: temperature parameter for softmax in generation, default 1.0
