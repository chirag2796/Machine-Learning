import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

text = '''
I had a couple of meetings in the nearby area and hence chosen this hotel. Rooms are below average. No amenities present. TV seemed to be from stone age. The moment I stepped in, I started searching for another hotel in the same locality. Some of the staffs lack basic manners. The only plus point is that this hotel is in Brigade road; a shoppers street. So I was able to stay out than stay in the hotel.
'''
def is_match_valid(match):
    if len(list(filter(lambda word: word not in stop_words, match))) <= 1:
        return False
    elif len(list(filter(lambda word: len(word)>=3, match))) <= 1:
        return False
    else:
        return True

sentences = sent_tokenize(text)
for sentence in sentences:
    sent = nltk.word_tokenize(sentence)
    sent = nltk.pos_tag(sent)
    nouns = list(filter(lambda match: match[1][:2]=='NN', sent))
    adjectives = []
    prepositions = []
    for i in range(len(sent)):
        word, tag = sent[i]
        if tag == 'JJ':
            prev_word = sent[i-1][0] if i>0 else ""
            next_word = sent[i+1][0] if i<len(sent)-1 else ""
            adjectives.append((prev_word, word, next_word))
        elif tag == 'IN':
            prev_word = sent[i - 1][0] if i > 0 else ""
            next_word = sent[i + 1][0] if i < len(sent) - 1 else ""
            prepositions.append((prev_word, word, next_word))

    # for match in list(adjectives):
    #     if not is_match_valid(match):
    #         adjectives.remove(match)
    for match in list(nouns):
        if not is_match_valid(match):
            nouns.remove(match)


    print('==================')
    print(sentence)
    print(sent)
    print(nouns)
    print(adjectives)
    print(prepositions)
    print("------------------")