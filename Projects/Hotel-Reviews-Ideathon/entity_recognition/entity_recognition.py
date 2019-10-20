import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

text = '''
This hostel is 100% the worst hostel I’ve ever visited and I’ve stayed in some pretty dodgy places!

The staff are rude
How having 3 bunk high is legal I don’t know
The rooms are so hot, not airflow at all
My dorm absolutely stunk - can’t have been properly cleaned for a long time
The showers were filthy and you had to do a 3 point turn just to get in them!

I would NEVER stay again, plenty of much better hostels locally for the same price!
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