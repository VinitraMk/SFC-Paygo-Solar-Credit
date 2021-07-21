import pandas as pd

train = pd.read_csv('./input/Train.csv')
test = pd.read_csv('./input/Test.csv')
data = pd.concat([train, test])
metadata = pd.read_csv('./input/metadata.csv')
data = pd.merge(data, metadata, on='ID', how='inner')
data.to_csv('./input/data.csv')