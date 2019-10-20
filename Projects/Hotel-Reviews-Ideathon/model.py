import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from random import shuffle

label = "is_bad_review"
# reviews_df = pd.read_pickle("files\\reviews_df.pickle")
reviews_df = pd.read_csv("files\\dataset.csv")

good_reviews_indices = reviews_df.index[reviews_df[label] == 0].tolist()
bad_reviews_indices = reviews_df.index[reviews_df[label] == 1].tolist()
# shuffle(good_reviews_indices)
good_reviews_indices = good_reviews_indices[:2400]
good_reviews_df = reviews_df.iloc[good_reviews_indices]
bad_reviews_df = reviews_df.iloc[bad_reviews_indices]
reviews_df = good_reviews_df.append(bad_reviews_df)

# reviews_df.drop('index')
# feature selection
# print(reviews_df[label].value_counts())
# exit()

ignore_cols = ['index', 'review', 'is_bad_review', 'review_clean', 'neg', 'neu', 'pos', 'compound', 'nb_chars', 'nb_words']
features = [c for c in reviews_df.columns if c not in ignore_cols]

# split the data into train and test
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(reviews_df[features], reviews_df[label], test_size = 0.20, random_state = 42)


# train a random forest classifier
rf = RandomForestClassifier(n_estimators = 100, random_state = 42)
rf.fit(X_train, y_train)

with open("files\\saved_model\\rf_model_balanced.pickle", 'wb') as file:
    pickle.dump(rf, file, protocol=pickle.HIGHEST_PROTOCOL)