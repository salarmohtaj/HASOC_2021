import pandas as pd

df = pd.read_csv("rawData.csv",index_col=0)
pd.set_option('display.max_colwidth', None)
print(df.head())
print(df.shape)
df1 = df[['text', 'task_1', 'task_2']]
df1 = df1[df1['task_2'] == "HATE"]
df1 = df1.sample(frac=1)
print(df1.iloc[1])