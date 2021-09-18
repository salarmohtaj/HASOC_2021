import matplotlib.pyplot as plt
import pandas as pd
import torch
from torchtext.legacy.data import Field, LabelField, TabularDataset, BucketIterator
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import BertForSequenceClassification, AdamW, BertConfig, BertTokenizer, get_linear_schedule_with_warmup
import os
import re
import pickle
import demoji
import time
import numpy as np
from collections import OrderedDict
import argparse





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Master script to prepare the ACR test')
    parser.add_argument("--epochs", help="Number of epochs, default=2", default=2)
    parser.add_argument("--bert", help="Bert model (base or large), default=base", default="base")
    args = parser.parse_args()


########################################################################
runmane = "bert_finetune"
BATCH_SIZE = 32
number_of_epochs = int(args.epochs)
bert_model = str(args.bert)

bert_model_name = "bert-"+bert_model+"-uncased"
MAX_LEN = 128
language_name = str("English")
destination_folder = "Model/"+runmane+"_"+language_name+"_"+str(bert_model)
report_address = "Reports/"+runmane+"_"+language_name+"_"+"_"+str(bert_model)
########################################################################

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



source_folder = "data/"
try:
    os.mkdir("Model")
except:
    pass
try:
    os.mkdir("Reports")
except:
    pass
try:
    os.mkdir(destination_folder)
except:
    pass
with open(report_address,"a+") as f:
    f.write(runmane+"_"+language_name+"_"+str(bert_model)+"\n\n")







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
label_field = LabelField(is_target=True)
fields = [(None, None), ("_id", None), ('text', text_field), ('label', label_field), ('task_2', None)]


dataset = TabularDataset(path=source_folder+"rawData.csv", format='CSV', fields=fields, skip_header=True)

#print(dataset[0].__dict__.keys())
#print(dataset[0].text)
#print(dataset[0].label)
#print(len(dataset))

train, test, valid = dataset.split([0.7, 0.1, 0.2], stratified=True) ## Keeping the same ratio of labels in the train, valid and test datasets
text_field.build_vocab(train, min_freq=2)
label_field.build_vocab(train)
#label_field.vocab.stoi = OrderedDict([('HOF', 1), ('NOT', 0)])
print(f'the train, validation and test sets includes {len(train)},{len(valid)} and {len(test)} instances, respectively')

# print(label_field.vocab.itos)
# print(label_field.vocab.stoi)

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

model = BertForSequenceClassification.from_pretrained(
    bert_model_name, # Use the 12-layer BERT model, with an uncased vocab.
    num_labels=2, # The number of output labels--2 for binary classification.
                    # You can increase this for multi-class tasks.
    output_attentions=False, # Whether the model returns attentions weights.
    output_hidden_states=False, # Whether the model returns all hidden-states.
).to(device)

def parameters_stat(model):
    params = list(model.named_parameters())
    print('The BERT model has {:} different named parameters.\n'.format(len(params)))
    print('==== Embedding Layer ====\n')
    for p in params[0:5]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
    print('\n==== First Transformer ====\n')
    for p in params[5:21]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
    print('\n==== Output Layer ====\n')
    for p in params[-4:]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

#parameters_stat(model)

optimizer = AdamW(model.parameters(),
                  lr = 2e-5,eps = 1e-8)
total_steps = len(train_iter) * number_of_epochs
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)



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

criterion = torch.nn.BCELoss()
training_stats = []

# Training Function
def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for batch in iterator:
        optimizer.zero_grad()
        labels = batch.label
        #text = batch.text.t()
        text = batch.text
        output = model(text, labels=labels)
        loss = output["loss"]
        logits = output["logits"]
        logits = logits.detach().cpu().numpy()
        label_ids = labels.to('cpu').numpy()
        acc = flat_accuracy(logits, label_ids)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
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
            #text = batch.text.t()
            text = batch.text
            output = model(text, labels=labels)
            loss = output["loss"]
            logits = output["logits"]
            logits = logits.detach().cpu().numpy()
            label_ids = labels.to('cpu').numpy()
            acc = flat_accuracy(logits, label_ids)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            #logits = logits.detach().cpu().numpy()
            #label_ids = batch.label.to('cpu').numpy()
            #total_eval_accuracy += flat_accuracy(logits, label_ids)
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

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
        for batch in test_loader:
            labels = batch.label.to(device)
            #text = batch.text.t().to(device)
            text = batch.text.to(device)
            #labels = batch.label
            #text = batch.text.t()
            output = model(text, labels=labels)
            logits = output["logits"]
            logits = logits.detach().cpu().numpy()
            #label_ids = labels.to('cpu').numpy()
            #acc = flat_accuracy(logits, label_ids)
            output = np.argmax(logits, axis=1).flatten()
            labels = labels.flatten()

            #output = model(text, text_len)
            #output = (output > threshold).int()
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
best_model = BertForSequenceClassification.from_pretrained(
    bert_model_name, # Use the 12-layer BERT model, with an uncased vocab.
    num_labels=2, # The number of output labels--2 for binary classification.
                    # You can increase this for multi-class tasks.
    output_attentions=False, # Whether the model returns attentions weights.
    output_hidden_states=False, # Whether the model returns all hidden-states.
).to(device)
#best_model.load_state_dict(torch.load('tut6-model.pt'))


load_checkpoint(destination_folder + '/model.pt', best_model, optimizer)
optimizer = optim.Adam(best_model.parameters(), lr=0.001)
test_evaluation(best_model, test_iter)