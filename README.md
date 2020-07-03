# Project_generation

## Question Generation with Transformers on SQuAD and FQuAD

**Zelros A.I.** [Z](logo.jpg)

This repository contains some functions and a script able to train an EncoderDecoderModel or a BART model from Hugging Face's transformers library (https://github.com/huggingface/transformers).

It works with BART, BERT and CamemBERT and it uses SQuAD (https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json
https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json) and FQuAD (https://storage.googleapis.com/illuin/fquad/train.json.zip
 https://storage.googleapis.com/illuin/fquad/valid.json.zip) to train these previous models.

Example with only positional arguments:
```
python main.py en train-v2.0.json dev-v2.0.json model_Bert2Bert
```

All arguments:

positional arguments:

 - {en,fr}               en or fr
 - file_train            name of the train file
 - file_test             name of the test file
 - output_dir            name of the directory for logs and checkpoints
  

optional arguments:

 - -h, --help            show this help message and exit
 - -bt,--bart BART true if bart else false
 - -ck CHECKPOINT, --checkpoint CHECKPOINT directory where to find last checkpoint
 - -lr LEARNING_RATE, --learning_rate LEARNING_RATE default learning rate
 - -bs BATCH_SIZE, --batch_size BATCH_SIZE batch size for training
 - -ss SAVE_STEPS, --save_steps SAVE_STEPS number of gradient descent steps between each saving
 - -ep EPOCHS, --epochs EPOCHS number of epochs for training
 - -gs GRADIENT_ACCUMULATION_STEPS, --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS number of steps before backward step
 - -wd WEIGHT_DECAY, --weight_decay WEIGHT_DECAY weight decay parameter for training
