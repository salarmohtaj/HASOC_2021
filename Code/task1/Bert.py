import matplotlib.pyplot as plt
import pandas as pd
import torch

# Preliminaries
from torchtext.legacy.data import Field, TabularDataset, BucketIterator

# Models
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Training
import torch.optim as optim

# Evaluation
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import os
import sys
import pickle

from transformers import BertForSequenceClassification, AdamW, BertTokenizer, get_linear_schedule_with_warmup


runmane = "lstmcharacter"
number_of_epochs = int(30)
learning_rate = float(0.001)
dropout = float(0.5)
language_name = str("English")
# number_of_epochs = int(2)
# learning_rate = float(0.0001)
# language_name = str("english")
destination_folder = "Model/"+runmane+"_"+language_name+"_"+str(number_of_epochs)+"_"+str(learning_rate)+"_"+str(dropout)


print(number_of_epochs)
print(learning_rate)
print(dropout)
print(language_name)
print(destination_folder)

source_folder = "data/"+language_name+"/"
try:
    os.mkdir(destination_folder)
except:
    pass

model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels = 1,
        output_attentions = False,
        output_hidden_states = False)
optimizer = AdamW(model.parameters(),
                  lr=2e-5,
                  eps=1e-8)
#total_steps = len(train_iter) * N_EPOCHS
#scheduler = get_linear_schedule_with_warmup(optimizer,
#                                          num_warmup_steps=0,  # Default value in run_glue.py
#                                          num_training_steps=total_steps)


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def tokenize_and_cut(sentence):
    tokens = tokenizer.tokenize(sentence)
    max_input_length = tokenizer.max_model_input_sizes['bert-base-uncased']
    tokens = tokens[:max_input_length-2]
    return tokens


def prepare_data():
    text_field  = Field(batch_first=True,
                      use_vocab=False,
                      tokenize=tokenize_and_cut,
                      preprocessing=tokenizer.convert_tokens_to_ids,
                      init_token=tokenizer.cls_token_id,
                      eos_token=tokenizer.sep_token_id,
                      pad_token=tokenizer.pad_token_id,
                      unk_token=tokenizer.unk_token_id)

    label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
    # text_field = Field(tokenize="spacy", lower=True, include_lengths=True, batch_first=True)
    #text_field = Field(tokenize=tokenize, lower=True, include_lengths=True, batch_first=True)
    # fields = [('label', label_field), ('title', text_field), ('text', text_field), ('titletext', text_field)]
    # fields = [('label', label_field), (None, None), ('text', text_field), (None, None)]


    #label_field = Field(sequential=False, use_vocab=False)
    fields = [('label', label_field), ('text', text_field)]

    train, valid, test = TabularDataset.splits(path=source_folder, train='train.csv', validation='valid.csv',
                                               test='test.csv', format='CSV', fields=fields, skip_header=True)

    # print(train[0])
    # print(train[0].__dict__.keys())
    # print(train[0].text)
    #

    # Iterators
    train_iter = BucketIterator(train, batch_size=32, sort_key=lambda x: len(x.text), device=device, sort=True,
                                sort_within_batch=True)
    valid_iter = BucketIterator(valid, batch_size=32, sort_key=lambda x: len(x.text), device=device, sort=True,
                                sort_within_batch=True)
    test_iter = BucketIterator(test, batch_size=32, sort_key=lambda x: len(x.text), device=device, sort=True,
                               sort_within_batch=True)

    return train_iter, valid_iter


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
