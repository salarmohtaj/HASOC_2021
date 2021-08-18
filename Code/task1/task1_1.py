import pandas as pd

df = pd.read_csv("../../Data/2021/Subtask 1/English/en_Hasoc2021_train.csv",sep=",")
print(df.head())
print(df.shape)
print(df.columns)

df1 = df[['_id', 'text', 'task_1']]
print(df1.shape)
print(df1.columns)