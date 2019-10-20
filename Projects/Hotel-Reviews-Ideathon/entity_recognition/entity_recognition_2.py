from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree

text = '''
The bunk room in Hyatt was perfect as I was travelling with my 17 year old niece. Arrived tired from a delayed flight and the front desk staff were kind and welcoming, just what I needed. Room was small, but as to be expected in NYC but amazingly quiet as it was facing the back. Wine hour was appreciated and the view from the rooftop was amazing. The only complaint would be the weird shower curtain on the shower that seemed makeshift and not practical
'''

def get_continuous_chunks(text):
     chunked = ne_chunk(pos_tag(word_tokenize(text)))
     continuous_chunk = []
     current_chunk = []
     for i in chunked:
             if type(i) == Tree:
                     current_chunk.append(" ".join([token for token, pos in i.leaves()]))
             elif current_chunk:
                     named_entity = " ".join(current_chunk)
                     if named_entity not in continuous_chunk:
                             continuous_chunk.append(named_entity)
                             current_chunk = []
             else:
                     continue
     return continuous_chunk

print(get_continuous_chunks(text))