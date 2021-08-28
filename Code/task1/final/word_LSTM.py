import matplotlib.pyplot as plt
import pandas as pd
import torch
from torchtext.legacy.data import Field, LabelField, TabularDataset, BucketIterator
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import re
import pickle
import demoji
import time
from collections import OrderedDict
import spacy
spacy_en = spacy.load('en')



########################################################################
runmane = "lstmcharacter"
BATCH_SIZE = 32
number_of_epochs = int(50)

learning_rate = float(0.001)
language_name = str("English")

embedding_size = 300
hidden_size = 256
number_of_layers = 2
dropout = float(0.5)
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
    return [tok.text for tok in spacy_en.tokenizer(text)]

#tokenizer = lambda text: list(text)
def fake_tokenizer(text):
    return text

text_field = Field(tokenize = fake_tokenizer, sequential = True, preprocessing = text_preprocess, lower = True, include_lengths = True, batch_first = True)
#text_field = Field(tokenize = "spacy", sequential = True, preprocessing = None, lower = True, include_lengths = True, batch_first = True)
#label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
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
label_field.vocab.stoi = OrderedDict([('HOF', 1), ('NOT', 0)])

#print(label_field.vocab.itos)
#print(label_field.vocab.stoi)
#print(len(train))
#print(len(valid))

# Iterators
train_iter = BucketIterator(train, batch_size=BATCH_SIZE, sort_key=lambda x: len(x.text),device=device, sort=True, sort_within_batch=True)
valid_iter = BucketIterator(valid, batch_size=BATCH_SIZE, sort_key=lambda x: len(x.text),device=device, sort=True, sort_within_batch=True)
test_iter = BucketIterator(test, batch_size=BATCH_SIZE, sort_key=lambda x: len(x.text),device=device, sort=True, sort_within_batch=True)



class LSTM(nn.Module):
    def __init__(self,
                 input_size,
                 embedding_size = 300,
                 hidden_size = 256,
                 number_of_layers = 2,
                 dropout = 0.5):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=embedding_size,
                            hidden_size=hidden_size,
                            num_layers=number_of_layers,
                            batch_first=True,
                            bidirectional=True)
        self.drop = nn.Dropout(p=dropout)
        self.fc = nn.Linear(2*hidden_size, 1)
    def forward(self, text, text_len):
        text_emb = self.embedding(text)
        packed_input = pack_padded_sequence(text_emb, text_len.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        out_forward = output[range(len(output)), text_len - 1, :self.hidden_size]
        out_reverse = output[:, 0, self.hidden_size:]
        out_reduced = torch.cat((out_forward, out_reverse), 1)
        text_fea = self.drop(out_reduced)
        text_fea = self.fc(text_fea)
        text_fea = torch.squeeze(text_fea, 1)
        text_out = torch.sigmoid(text_fea)
        return text_out


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


# Training Function
def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for ((text, text_len), labels) in iterator:
        optimizer.zero_grad()
        labels = labels.to(device)
        text = text.to(device)
        text_len = text_len.to(device)
        output = model(text, text_len)
        loss = criterion(output, labels)
        acc = binary_accuracy(output, labels)
        loss.backward()
        optimizer.step()
        # update running values
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for ((text, text_len), labels) in iterator:
            labels = labels.to(device)
            text = text.to(device)
            text_len = text_len.to(device)
            output = model(text, text_len)
            loss = criterion(output, labels)
            acc = binary_accuracy(output, labels)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


input_size = len(text_field.vocab)
model = LSTM(input_size,
             embedding_size,
             hidden_size,
             number_of_layers,
             dropout).to(device)
criterion=nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'The model has {count_parameters(model):,} parameters')


def binary_accuracy(preds, y):
    #rounded_preds = torch.round(torch.sigmoid(preds))
    rounded_preds = torch.round(preds)
    correct = (rounded_preds == y).float() #convert into float for division
    acc = correct.sum() / len(correct)
    return acc

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

best_valid_loss=float("Inf")

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
        #torch.save(model.state_dict(), 'tut6-model.pt')
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
    if ((epoch - last_best_loss) > 9):
        print("################")
        print("Termination because of lack of improvement in the last 10 epochs")
        print("################")
        break

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
best_model = LSTM(input_size,
             embedding_size,
             hidden_size,
             number_of_layers,
             dropout).to(device)
#best_model.load_state_dict(torch.load('tut6-model.pt'))


load_checkpoint(destination_folder + '/model.pt', best_model, optimizer)
optimizer = optim.Adam(best_model.parameters(), lr=0.001)
test_evaluation(best_model, test_iter)