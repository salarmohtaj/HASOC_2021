import matplotlib.pyplot as plt
import pandas as pd
import torch
from torchtext.legacy.data import Field, LabelField, TabularDataset, BucketIterator
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import BertModel, BertForSequenceClassification, AdamW, BertConfig, BertTokenizer, get_linear_schedule_with_warmup
import os
import re
import pickle
import demoji
import time
import numpy as np
from collections import OrderedDict


########################################################################
runmane = "lstmcharacter"
#BATCH_SIZE = 32
number_of_epochs = int(50)

hidden_size = 256
OUTPUT_DIM = 1
number_of_layers = 2
BIDIRECTIONAL = True
dropout = float(0.25)

learning_rate = float(0.001)
language_name = str("English")
bert_model_name = "bert-base-uncased"
MAX_LEN = 128

print(number_of_epochs)
destination_folder = "Model/"+runmane+"_"+language_name+"_"+str(number_of_epochs)+"_"+str(learning_rate)+"_"+str(dropout)


########################################################################
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



source_folder = "data/"
try:
    os.mkdir("Model")
except:
    pass
try:
    os.mkdir(destination_folder)
except:
    pass


def text_preprocess(text):
    text = re.sub("@([A-Za-z0-9_]+)", "username", text)
    text = re.sub(r"http\S+", "link", text)
    text = demoji.replace_with_desc(text, sep=" ")
    text = re.sub("[ ]+", " ", text)
    return text

tokenizer = BertTokenizer.from_pretrained(bert_model_name)

# A small helper function that will call the BERT tokenizer and truncate.
def bert_tokenize(text):
    text = text_preprocess(text)
    return tokenizer.tokenize(text)[:MAX_LEN - 2]

# add the [CLS] token in the beginning and [SEP] at the end, and that we use a dummy padding token
# compatible with BERT.
text_field = Field(sequential=True, tokenize=bert_tokenize, pad_token=tokenizer.pad_token,
                   init_token=tokenizer.cls_token, eos_token=tokenizer.sep_token,
                   batch_first=True
                   )

#label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
#label_field = LabelField(is_target=True, dtype=torch.float)
label_field = LabelField(is_target=True, dtype=torch.float)
fields = [(None, None), ("_id", None), ('text', text_field), ('label', label_field), ('task_2', None)]


dataset = TabularDataset(path=source_folder+"rawData.csv", format='CSV', fields=fields, skip_header=True)

#print(dataset[0].__dict__.keys())
#print(dataset[0].text)
#print(dataset[0].label)
#print(len(dataset))

train, test, valid = dataset.split([0.7, 0.1, 0.2], stratified=True) ## Keeping the same ratio of labels in the train, valid and test datasets
text_field.build_vocab(train, min_freq=2)
label_field.build_vocab(train)
#print(label_field.vocab.itos)
#print(label_field.vocab.stoi)
label_field.vocab.stoi = OrderedDict([('HOF', 1), ('NOT', 0)])

#print(len(train))
#print(len(valid))
# Here, we tell torchtext to use the vocabulary of BERT's tokenizer.
# .stoi is the map from strings to integers, and itos from integers to strings.
text_field.vocab.stoi = tokenizer.vocab
text_field.vocab.itos = list(tokenizer.vocab)

# Iterators
train_iter = BucketIterator(train, batch_size=BATCH_SIZE, sort_key=lambda x: len(x.text),device=device, sort=True, sort_within_batch=True)
valid_iter = BucketIterator(valid, batch_size=BATCH_SIZE, sort_key=lambda x: len(x.text),device=device, sort=True, sort_within_batch=True)
test_iter = BucketIterator(test, batch_size=BATCH_SIZE, sort_key=lambda x: len(x.text),device=device, sort=True, sort_within_batch=True)



bert = BertModel.from_pretrained(bert_model_name)


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


BIDIRECTIONAL = True

model = BERTGRUSentiment(bert,
                         hidden_size,
                         OUTPUT_DIM,
                         number_of_layers,
                         BIDIRECTIONAL,
                         dropout)


# Save and Load Functions
def save_checkpoint(save_path, model, optimizer, valid_loss):
    if save_path == None:
        return
    state_dict = {'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'valid_loss': valid_loss}
    torch.save(state_dict, save_path)
    #print(f'Model saved to ==> {save_path}')

def load_checkpoint(load_path, model, optimizer):
    if load_path == None:
        return
    state_dict = torch.load(load_path, map_location=device)
    #print(f'Model loaded from <== {load_path}')
    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    return state_dict['valid_loss']

def save_metrics(save_path, train_loss_list, valid_loss_list, epoch_counter_list):
    if save_path == None:
        return
    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'epoch_counter_list': epoch_counter_list}
    torch.save(state_dict, save_path)
    #print(f'Model saved to ==> {save_path}')

def load_metrics(load_path):
    if load_path == None:
        return
    state_dict = torch.load(load_path, map_location=device)
    #print(f'Model loaded from <== {load_path}')
    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['epoch_counter_list']


print("###################################")
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} parameters')

for name, param in model.named_parameters():
    if name.startswith('bert'):
        param.requires_grad = False

print(f'The model has {count_parameters(model):,} trainable parameters')
print("###################################")
def print_grad_parameters(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)


optimizer = optim.Adam(model.parameters())
criterion = nn.BCELoss()
model = model.to(device)
criterion = criterion.to(device)


def binary_accuracy(preds, y):
    #rounded_preds = torch.round(torch.sigmoid(preds))
    rounded_preds = torch.round(preds)
    correct = (rounded_preds == y).float() #convert into float for division
    acc = correct.sum() / len(correct)
    return acc



def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for batch in iterator:
        optimizer.zero_grad()
        labels = batch.label
        #text = batch.text.t()
        text = batch.text
        output = model(text).squeeze(1)
        loss = criterion(output, labels)
        acc = binary_accuracy(output, labels)
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
            labels = batch.label
            # text = batch.text.t()
            text = batch.text
            output = model(text).squeeze(1)
            loss = criterion(output, labels)
            acc = binary_accuracy(output, labels)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


N_EPOCHS = 5
best_valid_loss = float('inf')

training_stats = []
train_loss_list = []
valid_loss_list = []
epoch_counter_list = []
last_best_loss = 0



for epoch in range(number_of_epochs):
    start_time = time.time()
    train_loss, train_acc = train(model, train_iter, optimizer, criterion)
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    valid_loss, valid_acc = evaluate(model, valid_iter, criterion)
    train_loss_list.append(train_loss)
    valid_loss_list.append(valid_loss)
    epoch_counter_list.append(epoch)
    print(f'Epoch: {epoch + 1:02}/{number_of_epochs} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    print(f'\tVal. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')
    if valid_loss < best_valid_loss:
        last_best_loss = epoch
        print("\t---> Saving the model <---")
        best_valid_loss = valid_loss
        ##torch.save(model.state_dict(), 'tut6-model.pt')
        save_checkpoint(destination_folder + '/model.pt', model, optimizer, best_valid_loss)
        save_metrics(destination_folder + '/metrics.pt', train_loss_list, valid_loss_list, epoch_counter_list)
    training_stats.append(
        {
            'epoch': epoch + 1,
            'Training Loss': train_loss,
            'Valid. Loss': valid_loss,
            'Training Accuracy': train_acc,
            #'Valid. Accur.': avg_val_accuracy,
            'Training Time': epoch_mins,
        }
    )
save_metrics(destination_folder + '/metrics.pt', train_loss_list, valid_loss_list, epoch_counter_list)
print('Finished Training!')

#train_loss_list, valid_loss_list, epoch_counter_list = load_metrics(destination_folder + '/metrics.pt')
# plt.plot(epoch_counter_list, train_loss_list, label='Train')
# plt.plot(epoch_counter_list, valid_loss_list, label='Valid')
# plt.xlabel('Epoch number')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

# Evaluation Function
def test_evaluation(model, test_loader, version='title', threshold=0.5):
    y_pred = []
    y_true = []
    model.eval()
    with torch.no_grad():
        for ((text, text_len), labels) in test_loader:
            labels = labels.to(device)
            text = text.to(device)
            text_len = text_len.to(device)
            output = model(text, text_len)
            output = (output > threshold).int()
            y_pred.extend(output.tolist())
            y_true.extend(labels.tolist())

    print('Classification Report:')
    print(classification_report(y_true, y_pred, labels=[1, 0], digits=4))
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    #ax = plt.subplot()
    #sns.heatmap(cm, annot=True, ax=ax, cmap='Blues', fmt="d")
    #ax.set_title('Confusion Matrix')
    #ax.set_xlabel('Predicted Labels')
    #ax.set_ylabel('True Labels')
    #ax.xaxis.set_ticklabels(['FAKE', 'REAL'])
    #ax.yaxis.set_ticklabels(['FAKE', 'REAL'])


#best_model = LSTM().to(device)
best_model = BERTGRUSentiment(bert,
                         hidden_size,
                         OUTPUT_DIM,
                         number_of_layers,
                         BIDIRECTIONAL,
                         dropout).to(device)
#best_model.load_state_dict(torch.load('tut6-model.pt'))


load_checkpoint(destination_folder + '/model.pt', best_model, optimizer)
optimizer = optim.Adam(model.parameters())
test_evaluation(best_model, test_iter)