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



########################################################################
runmane = "lstmcharacter"
BATCH_SIZE = 32
number_of_epochs = int(30)
learning_rate = float(0.001)
dropout = float(0.5)
language_name = str("English")
# number_of_epochs = int(2)
# learning_rate = float(0.0001)
# language_name = str("english")
print(number_of_epochs)
print(learning_rate)
print(dropout)
print(language_name)
destination_folder = "Model/"+runmane+"_"+language_name+"_"+str(number_of_epochs)+"_"+str(learning_rate)+"_"+str(dropout)
########################################################################
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



source_folder = "data/"
try:
    os.mkdir(destination_folder)
except:
    pass


def text_preprocess(text):
    text = re.sub("@([A-Za-z0-9_]+)", "username", text)
    text = re.sub(r"http\S+", "link", text)
    text = demoji.replace_with_desc(text, sep=" ")
    text = re.sub("[ ]+", " ", text)
    return list(text)

#tokenizer = lambda text: list(text)
def fake_tokenizer(text):
    return text

text_field = Field(tokenize = fake_tokenizer, sequential = True, preprocessing = text_preprocess, lower = True, include_lengths = True, batch_first = True)
#label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
label_field = LabelField(is_target=True, dtype=torch.float)
fields = [(None, None), ("_id", None), ('text', text_field), ('label', label_field), ('task_2', None)]


dataset = TabularDataset(path=source_folder+"rawData.csv", format='CSV', fields=fields, skip_header=True)

#print(dataset[0].__dict__.keys())
#print(dataset[0].text)
#print(dataset[0].label)
#print(len(dataset))

train, test, valid = dataset.split([0.8, 0.1, 0.1], stratified=True) ## Keeping the same ratio of labels in the train, valid and test datasets
text_field.build_vocab(train, min_freq=2)
label_field.build_vocab(train)
#print(label_field.vocab.itos)
#print(label_field.vocab.stoi)
#print(len(train))
#print(len(valid))

# Iterators
train_iter = BucketIterator(train, batch_size=BATCH_SIZE, sort_key=lambda x: len(x.text),device=device, sort=True, sort_within_batch=True)
valid_iter = BucketIterator(valid, batch_size=BATCH_SIZE, sort_key=lambda x: len(x.text),device=device, sort=True, sort_within_batch=True)
test_iter = BucketIterator(test, batch_size=BATCH_SIZE, sort_key=lambda x: len(x.text),device=device, sort=True, sort_within_batch=True)
































class LSTM(nn.Module):

    def __init__(self, dimension=256):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(len(text_field.vocab), 300)
        self.dimension = dimension
        self.lstm = nn.LSTM(input_size=300,
                            hidden_size=dimension,
                            num_layers=2,
                            batch_first=True,
                            bidirectional=True)
        self.drop = nn.Dropout(p=dropout)

        self.fc = nn.Linear(2*dimension, 1)

    def forward(self, text, text_len):

        text_emb = self.embedding(text)

        packed_input = pack_padded_sequence(text_emb, text_len.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        out_forward = output[range(len(output)), text_len - 1, :self.dimension]
        out_reverse = output[:, 0, self.dimension:]
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
    print(f'Model saved to ==> {save_path}')


def load_checkpoint(load_path, model, optimizer):
    if load_path == None:
        return

    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')

    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])

    return state_dict['valid_loss']


def save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list):
    if save_path == None:
        return

    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'global_steps_list': global_steps_list}

    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_metrics(load_path):
    if load_path == None:
        return

    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')

    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']


# Training Function

def train(model,
          optimizer,
          criterion=nn.BCELoss(),
          train_loader=train_iter,
          valid_loader=valid_iter,
          num_epochs=5,
          eval_every=len(train_iter) // 2,
          file_path=destination_folder,
          best_valid_loss=float("Inf")):
    # initialize running values
    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []

    # training loop
    model.train()
    for epoch in range(num_epochs):
        #for (labels, (title, title_len), (text, text_len), (titletext, titletext_len)), _ in train_loader:
        for ((text, text_len), labels) in train_loader:
            labels = labels.to(device)
            #titletext = titletext.to(device)
            #titletext_len = titletext_len.to(device)

            text = text.to(device)
            text_len = text_len.to(device)
            output = model(text, text_len)

            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update running values
            running_loss += loss.item()
            global_step += 1

            # evaluation step
            if global_step % eval_every == 0:
                model.eval()
                with torch.no_grad():
                    # validation loop
                    #for (labels, (title, title_len), (text, text_len), (titletext, titletext_len)), _ in valid_loader:
                    #for (labels, (text, text_len)), _ in valid_loader:
                    for ((text, text_len),labels) in valid_loader:
                        labels = labels.to(device)
                        #titletext = titletext.to(device)
                        #titletext_len = titletext_len.to(device)
                        text = text.to(device)
                        text_len = text_len.to(device)
                        output = model(text, text_len)

                        loss = criterion(output, labels)
                        valid_running_loss += loss.item()

                # evaluation
                average_train_loss = running_loss / eval_every
                average_valid_loss = valid_running_loss / len(valid_loader)
                train_loss_list.append(average_train_loss)
                valid_loss_list.append(average_valid_loss)
                global_steps_list.append(global_step)

                # resetting running values
                running_loss = 0.0
                valid_running_loss = 0.0
                model.train()

                # print progress
                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, global_step, num_epochs * len(train_loader),
                              average_train_loss, average_valid_loss))

                # checkpoint
                if best_valid_loss > average_valid_loss:
                    best_valid_loss = average_valid_loss
                    save_checkpoint(file_path + '/model.pt', model, optimizer, best_valid_loss)
                    save_metrics(file_path + '/metrics.pt', train_loss_list, valid_loss_list, global_steps_list)

    save_metrics(file_path + '/metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
    print('Finished Training!')



model = LSTM().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train(model=model, optimizer=optimizer, num_epochs=number_of_epochs)


train_loss_list, valid_loss_list, global_steps_list = load_metrics(destination_folder + '/metrics.pt')
plt.plot(global_steps_list, train_loss_list, label='Train')
plt.plot(global_steps_list, valid_loss_list, label='Valid')
plt.xlabel('Global Steps')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluation Function

def evaluate(model, test_loader, version='title', threshold=0.5):
    y_pred = []
    y_true = []

    model.eval()
    with torch.no_grad():
        #for (labels, (title, title_len), (text, text_len), (titletext, titletext_len)), _ in test_loader:
        for ((text, text_len), labels) in test_loader:
            labels = labels.to(device)
            #titletext = titletext.to(device)
            #titletext_len = titletext_len.to(device)
            etext = text.to(device)
            text_len = text_len.to(device)
            #output = model(titletext, titletext_len)
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


best_model = LSTM().to(device)
optimizer = optim.Adam(best_model.parameters(), lr=0.001)

load_checkpoint(destination_folder + '/model.pt', best_model, optimizer)
evaluate(best_model, test_iter)