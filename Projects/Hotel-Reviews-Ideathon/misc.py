import pandas as pd
a = [
     [1, 'a'],
     [1, 'aa'],
     [1, 'aaa'],
     [2, 'ab'],
     [2, 'ab'],
     [2, 'abb'],
     [2, 'abbb'],
]

df = pd.DataFrame(a, columns=['label', 'data'])
one_indices = df.index[df['label'] == 1].tolist()
two_indices = df.index[df['label'] == 2].tolist()
two_indices = two_indices[:3]
df1 = df.iloc[one_indices]
df2 = df.iloc[two_indices]
df = df2.append(df1)
print(df)
