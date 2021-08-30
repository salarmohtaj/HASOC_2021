import pandas as pd
from sklearn.model_selection import train_test_split
import re
import demoji

raw_data_path = "../../Data/2021/Subtask 1/English/en_Hasoc2021_train.csv"
destination_folder = '../../Data/2021/Subtask 1/English'

train_test_ratio = 0.90
train_valid_ratio = 0.90

first_n_words = 200


def preprocess(text):
    text = text.lower()
    text = re.sub("@([A-Za-z0-9_]+)", "username", text)
    text = re.sub(r"http\S+", "link", text)
    text = demoji.replace_with_desc(text, sep=" ")
    text = re.sub("[ ]+", " ", text)
    text = text.lower()
    return text


def trim_string(x):
    x = str(x).split(maxsplit=first_n_words)
    x = ' '.join(x[:first_n_words])
    return x

# Read raw data
df_raw = pd.read_csv(raw_data_path,sep=",")
df_raw = df_raw[['text', 'task_1']]
df_raw = df_raw.rename(columns={"text": "text", "task_1": "label"})
df_raw = df_raw[(df_raw["label"]=="HOF") | (df_raw["label"]=="NOT")]

print(df_raw.shape)
# Prepare columns
df_raw['label'] = (df_raw['label'] == 'HOF').astype('int')
#df_raw['titletext'] = df_raw['title'] + ". " + df_raw['text']
#df_raw['titletext'] = df_raw['title']
#print(df_raw.dtypes)
#df_raw = df_raw.reindex(columns=['label', 'title', 'text', 'titletext'])
df_raw = df_raw.reindex(columns=['label', 'text'])
#f_raw = df_raw.reindex(columns=['label', 'titletext'])


# Drop rows with empty text
df_raw.drop( df_raw[df_raw.text.str.len() < 5].index, inplace=True)

# Trim text and titletext to first_n_words
df_raw['text'] = df_raw['text'].apply(trim_string)
#df_raw['titletext'] = df_raw['titletext'].apply(trim_string)
df_raw['text'] = df_raw['text'].apply(preprocess)
#df_raw['titletext'] = df_raw['titletext'].apply(preprocess)


# Split according to label
df_real = df_raw[df_raw['label'] == 0]
df_fake = df_raw[df_raw['label'] == 1]
#df_real = df_real[:len(df_fake)]
# Train-test split
df_real_full_train, df_real_test = train_test_split(df_real, train_size = train_test_ratio, random_state = 1)
df_fake_full_train, df_fake_test = train_test_split(df_fake, train_size = train_test_ratio, random_state = 1)

# Train-valid split
df_real_train, df_real_valid = train_test_split(df_real_full_train, train_size = train_valid_ratio, random_state = 1)
df_fake_train, df_fake_valid = train_test_split(df_fake_full_train, train_size = train_valid_ratio, random_state = 1)

# Concatenate splits of different labels
df_train = pd.concat([df_real_train, df_fake_train], ignore_index=True, sort=False)
df_valid = pd.concat([df_real_valid, df_fake_valid], ignore_index=True, sort=False)
df_test = pd.concat([df_real_test, df_fake_test], ignore_index=True, sort=False)
print("1")
# Write preprocessed data

df_train = df_train.sample(frac = 1)
df_valid = df_valid.sample(frac = 1)
df_test = df_test.sample(frac = 1)
print(df_train.head())

df_train.to_csv(destination_folder + '/train.csv', index=False)
df_valid.to_csv(destination_folder + '/valid.csv', index=False)
df_test.to_csv(destination_folder + '/test.csv', index=False)
print("2")