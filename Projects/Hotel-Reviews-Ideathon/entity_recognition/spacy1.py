import re
import string
import nltk
import spacy
import pandas as pd
import numpy as np
import math
from tqdm import tqdm

from spacy.matcher import Matcher
from spacy.tokens import Span
from spacy import displacy

pd.set_option('display.max_colwidth', 200)

# load spaCy model
nlp = spacy.load("en_core_web_sm")

# sample text
text = "The bunk room in Hyatt was perfect as I was travelling with my 17 year old niece. Arrived tired from a delayed flight and the front desk staff were kind and welcoming, just what I needed. Room was small, but as to be expected in NYC but amazingly quiet as it was facing the back. Wine hour was appreciated and the view from the rooftop was amazing. The only complaint would be the weird shower curtain on the shower that seemed makeshift and not practical"

# create a spaCy object
doc = nlp(text)

# # print token, dependency, POS tag
# for tok in doc:
#   print(tok.text, "-->",tok.dep_,"-->", tok.pos_)


#define the pattern
pattern = [{'POS':'NOUN'},
           {'LOWER': 'such'},
           {'LOWER': 'as'},
           {'POS': 'PROPN'}]

# Matcher class object
# Matcher class object
matcher = Matcher(nlp.vocab)

#define the pattern
pattern = [{'DEP':'amod', 'OP':"?"}, # adjectival modifier
           {'POS':'NOUN'},
           {'LOWER': 'such'},
           {'LOWER': 'as'},
           {'POS': 'PROPN'}]

matcher.add("matching_1", None, pattern)
matches = matcher(doc)

span = doc[matches[0][1]:matches[0][2]]
print(span.text)

doc = nlp(text)

# # print dependency tags and POS tags
# for tok in doc:
#   print(tok.text, "-->",tok.dep_, "-->",tok.pos_)

# Matcher class object
matcher = Matcher(nlp.vocab)

# define the pattern
pattern = [{'DEP': 'amod', 'OP': "?"},
           {'POS': 'NOUN'},
           {'LOWER': 'and', 'OP': "?"},
           {'LOWER': 'or', 'OP': "?"},
           {'LOWER': 'other'},
           {'POS': 'NOUN'}]

matcher.add("matching_1", None, pattern)

matches = matcher(doc)
span = doc[matches[0][1]:matches[0][2]]
print(span.text)