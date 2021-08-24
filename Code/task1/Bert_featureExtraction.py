
# Libraries

import matplotlib.pyplot as plt
import torch

# Preliminaries

from torchtext.legacy.data import Field, TabularDataset, BucketIterator, Iterator, LabelField

# Models

import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification

# Training

import torch.optim as optim

# Evaluation

from torchtext import data

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification


from torchtext import datasets
from torchtext import data
from transformers import BertTokenizer
import torch
import pickle
import random
import numpy as np
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch.optim as optim
import time


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




device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_input_length = tokenizer.max_model_input_sizes['bert-base-uncased']
init_token_idx = tokenizer.cls_token_id
eos_token_idx = tokenizer.sep_token_id
pad_token_idx = tokenizer.pad_token_id
unk_token_idx = tokenizer.unk_token_id

def tokenize_and_cut(sentence):
    tokens = tokenizer.tokenize(sentence)
    tokens = tokens[:max_input_length-2]
    return tokens

TEXT = Field(batch_first = True,
                  use_vocab = False,
                  tokenize = tokenize_and_cut,
                  preprocessing = tokenizer.convert_tokens_to_ids,
                  init_token = init_token_idx,
                  eos_token = eos_token_idx,
                  pad_token = pad_token_idx,
                  unk_token = unk_token_idx)
#LABEL = data.LabelField(dtype = torch.float)
#LABEL = Field(sequential=False, use_vocab=False, dtype='torch.float')
#LABEL = Field(sequential=False, use_vocab=False)
LABEL = LabelField(dtype = torch.float)
#fields = [(None, None), ('Sentence', TEXT),('R_Score', LABEL), (None, None), (None, None)]
# TabularDataset
fields = [('label', LABEL), ('text', TEXT)]

# TabularDataset
train_data, valid_data, test_data = TabularDataset.splits(path=source_folder, train='train.csv', validation='valid.csv', test='test.csv',format='CSV', fields=fields, skip_header=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#training_data=data.TabularDataset(path='data/DE19.csv', format='tsv', fields=fields, skip_header=True)
#train_data, test_data, valid_data = training_data.split(split_ratio=[0.6, 0.2, 0.2])

print(f"Number of training examples: {len(train_data)}")
print(f"Number of validation examples: {len(valid_data)}")
print(f"Number of testing examples: {len(test_data)}")
print(vars(train_data.examples[6]))
tokens = tokenizer.convert_ids_to_tokens(vars(train_data.examples[6])['text'])
print(tokens)
LABEL.build_vocab(train_data)
BATCH_SIZE = 128


train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size = BATCH_SIZE,
    sort_within_batch=True,
    sort_key=lambda x: len(x.text),
    device = device)

bert = BertModel.from_pretrained('bert-base-uncased')


class BERTGRUSentiment(nn.Module):
    def __init__(self,
                 bert,
                 hidden_dim,
                 output_dim,
                 n_layers,
                 bidirectional,
                 dropout):

        super().__init__()

        self.bert = bert

        embedding_dim = bert.config.to_dict()['hidden_size']

        self.rnn = nn.GRU(embedding_dim,
                          hidden_dim,
                          num_layers=n_layers,
                          bidirectional=bidirectional,
                          batch_first=True,
                          dropout=0 if n_layers < 2 else dropout)

        self.out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text):

        # text = [batch size, sent len]

        with torch.no_grad():
            embedded = self.bert(text)[0]

        # embedded = [batch size, sent len, emb dim]

        _, hidden = self.rnn(embedded)

        # hidden = [n layers * n directions, batch size, emb dim]

        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            hidden = self.dropout(hidden[-1, :, :])

        # hidden = [batch size, hid dim]

        output = self.out(hidden)

        # output = [batch size, out dim]

        #return output
        return torch.sigmoid(output)



HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.25

model = BERTGRUSentiment(bert,
                         HIDDEN_DIM,
                         OUTPUT_DIM,
                         N_LAYERS,
                         BIDIRECTIONAL,
                         DROPOUT)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} parameters')
for name, param in model.named_parameters():
    if name.startswith('bert'):
        param.requires_grad = False

print(f'The model has {count_parameters(model):,} trainable parameters')

for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)


optimizer = optim.Adam(model.parameters())
criterion = nn.BCELoss()
model = model.to(device)
criterion = criterion.to(device)



def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division
    acc = correct.sum() / len(correct)
    return acc


def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:
        optimizer.zero_grad()

        predictions = model(batch.text).squeeze(1)
        print(predictions)

        loss = criterion(predictions, batch.label)

        acc = binary_accuracy(predictions, batch.label)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text).squeeze(1)
            loss = criterion(predictions, batch.label)
            #acc = binary_accuracy(predictions, batch.R_Score)
            epoch_loss += loss.item()
            #epoch_acc += acc.item()
    return epoch_loss / len(iterator)#, epoch_acc / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


N_EPOCHS = 5

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    start_time = time.time()
    print(start_time)
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    #valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
    valid_loss = evaluate(model, valid_iterator, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut6-model.pt')

    print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    #print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')


model.load_state_dict(torch.load('tut6-model.pt'))

#test_loss, test_acc = evaluate(model, test_iterator, criterion)
test_loss = evaluate(model, test_iterator, criterion)

#print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
print(f'Test Loss: {test_loss:.3f}')
def predict_sentiment(model, tokenizer, sentence):
    model.eval()
    tokens = tokenizer.tokenize(sentence)
    tokens = tokens[:max_input_length-2]
    indexed = [init_token_idx] + tokenizer.convert_tokens_to_ids(tokens) + [eos_token_idx]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(0)
    prediction = torch.sigmoid(model(tensor))
    return prediction.item()