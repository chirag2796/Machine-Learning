# From: https://towardsdatascience.com/multi-class-text-classification-with-lstm-1590bee1bd17
import pandas as pd


df = pd.read_csv('files\\datasets\\consumer_complaints.csv')
df.drop(df.columns.difference(['product', 'consumer_complaint_narrative']), 1, inplace=True)
df.loc[df['product'] == 'Credit card', 'product'] = 'Credit card or prepaid card'
df.loc[df['product'] == 'Prepaid card', 'product'] = 'Credit card or prepaid card'
print(df['product'].value_counts())

