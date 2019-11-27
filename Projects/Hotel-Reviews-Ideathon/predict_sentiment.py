import pickle

model = None
with open("files\\saved_model\\rf_model_balanced.pickle", 'rb') as file:
    model = pickle.load(file)

reviews = [
    '''
    I had a couple of meetings in the nearby area and hence chosen this hotel. Rooms are below average. No amenities present. TV seemed to be from stone age. The moment I stepped in, I started searching for another hotel in the same locality. Some of the staffs lack basic manners. The only plus point is that this hotel is in Brigade road; a shoppers street. So I was able to stay out than stay in the hotel.
    '''
]

import pandas as pd

doc2vec_model = None
tfidf = None
with open("files\\saved_model\\doc2vec_model.pickle", 'rb') as file:
    doc2vec_model = pickle.load(file)
with open("files\\saved_model\\tfidf.pickle", 'rb') as file:
    tfidf = pickle.load(file)
# with open("files\\saved_model\\tfidf_result.pickle", 'rb') as file:
#     tfidf_result = pickle.load(file)


# read data
reviews_df = pd.DataFrame(reviews, columns=['review'])


# Clean the data
# return the wordnet object value corresponding to the POS tag
from nltk.corpus import wordnet


def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


import string
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer


def clean_text(text):
    # lower text
    text = text.lower()
    # tokenize text and remove puncutation
    text = [word.strip(string.punctuation) for word in text.split(" ")]
    # remove words that contain numbers
    text = [word for word in text if not any(c.isdigit() for c in word)]
    # remove stop words
    stop = stopwords.words('english')
    text = [x for x in text if x not in stop]
    # remove empty tokens
    text = [t for t in text if len(t) > 0]
    # pos tag text
    pos_tags = pos_tag(text)
    # lemmatize text
    text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
    # remove words with only one letter
    text = [t for t in text if len(t) > 1]
    # join all
    text = " ".join(text)
    return (text)


# clean text data
reviews_df["review_clean"] = reviews_df["review"].apply(lambda x: clean_text(x))


doc2vec_df = reviews_df["review_clean"].apply(lambda x: doc2vec_model.infer_vector(x.split(" "))).apply(pd.Series)
doc2vec_df.columns = ["doc2vec_vector_" + str(x) for x in doc2vec_df.columns]
reviews_df = pd.concat([reviews_df, doc2vec_df], axis=1)

tfidf_result = tfidf.transform(reviews_df["review_clean"]).toarray()
tfidf_df = pd.DataFrame(tfidf_result, columns = tfidf.get_feature_names())
tfidf_df.columns = ["word_" + str(x) for x in tfidf_df.columns]
tfidf_df.index = reviews_df.index
reviews_df = pd.concat([reviews_df, tfidf_df], axis=1)

ignore_cols = ['review', 'is_bad_review', 'review_clean', 'neg', 'neu', 'pos', 'compound', 'nb_chars', 'nb_words']
features = [c for c in reviews_df.columns if c not in ignore_cols]

# df = reviews_df[features]
# df.to_csv("files\\temp.csv")
# exit()
print(model.predict(reviews_df[features]))
category = "Positive" if model.predict(reviews_df[features]) == 0 else "Negative"
print(category)
