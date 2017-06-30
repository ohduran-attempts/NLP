import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

# tokenizing: grouping words and sentences.
# corpus: body of text
# lexicon: the words and their means.
# there is a difference between
# bull: investor for "upward market"
# bull: regular for "animal of some sort"

EXAMPLE_TEXT = 'All right: This is me beginning to understand NPL. That means Natural Language processing!'

(word_tokenize(EXAMPLE_TEXT))  # will separate text in words.
(sent_tokenize(EXAMPLE_TEXT))  # will separate text in sentences.

# The main idea is that computers simply do not, and will not, ever understand words.
# But, hey, humans don't, either **GASP**

from nltk.corpus import stopwords
stop_words = sorted(set(stopwords.words('english')))  # all the words in English language that do not mean anything without context.

filtered_sentence = [w for w in word_tokenize(EXAMPLE_TEXT) if not w in stop_words]

# STEMMING
# normalizing method for words with the same meaning

from nltk.stem import PorterStemmer

ps = PorterStemmer()  # an instance of the Porter algorithm class

example_words = ['python','pythoner','pythoning','pythoned']

def print_stemmer(word_list):
    for w in word_list:
        print ps.stem(w)

# print_stemmer(example_words)  # this will return python,python,python...

new_text = "Let us not forget the importance of being consistent while programming in Python"

# print_stemmer(word_tokenize(new_text))  # you can see that being is transformed into be, programming into program, and so.

# Speech Tagging
# labeling words as nouns, adjectives, verbs, etc...
# even by tense, type of adjective, modal verbs, plural nouns...

from nltk.corpus import state_union  #SotU speeches
from nltk.tokenize import PunktSentenceTokenizer  # unsupervised ML, things are getting pretty serious now.

train_text = state_union.raw('2005-GWBush.txt')
sample_text = state_union.raw('2006-GWBush.txt')

# We train the tokenizer using the 2005 speech...
custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
# ...and we tokenize the speech of 2006.
tokenized = custom_sent_tokenizer.tokenize(sample_text)

def process_content(t):
    try:
        for i in t:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            print tagged
    except Exception as e:
        print str(e)

# process_content(tokenized) # This will clasify every word in the Speech Tagging

# CHUNKING

# Some clarification of Regex being specially used here
used_RegEx = {'+': 'match 1 or more',
              '?': 'match 0 or 1',
              '*': 'match 0 or more',
              '.': 'anything except new line'}

def process_content_2(t):
    try:
        for i in t:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)

            # <RB.?>* - "0 or more of any tense of adverb"
            # <VB.?>* - "0 or more of any tense of verb"
            # <NNP>+  - "One or more nouns"
            # <NN>?   - "Zero or one singular noun"
            chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}"""

            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)
            for subtree in chunked.subtrees():
                print subtree
    except Exception as e:
        print str(e)

# process_content_2(tokenized) # we are making sense of the text using this.

# CHINKING
# Still, we have some words that we do not want.
# chinking is removing a chunk from a chunk, and that one is called chink.

def process_content_3(t):
    try:
        for i in t:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)

            # <RB.?>* - "0 or more of any tense of adverb"
            # <VB.?>* - "0 or more of any tense of verb"
            # <NNP>+  - "One or more nouns"
            # <NN>?   - "Zero or one singular noun"
            chunkGram = r"""Chunk: {<.*>+}
                                    }<VB.?|IN|DT|TO>+{"""
            # the last part removes any verb, preposition, determiners, or the word "to"
            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)
            for subtree in chunked.subtrees():
                print subtree
    except Exception as e:
        print str(e)

# process_content_3(tokenized)

# Named Entity Recognition

# The idea is for the machine to be able to pull out places, people, things, locations, money figures, etc.
# built in: AWESOME

def process_content4(t):
    try:
        for i in t:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            namedEnt = nltk.ne_chunk(tagged, binary = True)
            print namedEnt
    except Exception as e:
        print str(e)

# process_content4(tokenized)

# LEMMATIZING
# Stemming can often create non-existent words, whereas lemmas have to be words

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer() # instance of WNL

# lemmatize takes a string and looks for a noun, unless stated something else
# using 'pos = ...'

# print(lemmatizer.lemmatize("cats"))
# print(lemmatizer.lemmatize("cacti"))
# print(lemmatizer.lemmatize("geese"))
# print(lemmatizer.lemmatize("rocks"))
# print(lemmatizer.lemmatize("python"))
# print(lemmatizer.lemmatize("better", pos="a"))  # adjective
# print(lemmatizer.lemmatize("best", pos="a"))  # adjective
# print(lemmatizer.lemmatize("run"))
# print(lemmatizer.lemmatize("run", 'v'))  # verb

# The Corpora: collection of Corpus

from nltk.corpus import gutenberg

sample = gutenberg.raw('bible-kjv.txt')

tok = sent_tokenize(sample)

for x in range(5):
    pass
    # print tok[x]  # And God saw the light, that it was good (and so on.)

# WordNet
# Lexical database for English, created by Princeton

from nltk.corpus import wordnet

syns = wordnet.synsets('program')

# print syns[0].name() --> "plan.n.01"
# print syns[0].lemmas()[0].name() --> plan
# print syns[0].definition() --> a series of steps to be carried out or goals to be accomplished
# print syns[0].examples() --> they dew up a six-step plan

def find_syn_ant(word):
    """Find synonyms and antonyms of a word given"""
    synonyms = []
    antonyms = []

    for syn in wordnet.synsets(str(word)):
        for l in syn.lemmas():
            synonyms.append(l.name())
            if l.antonyms():
                antonyms.append(l.antonyms()[0].name())
    print set(synonyms)
    print set(antonyms)

# find_syn_ant('good')

# We can also use WordNet to compare similarity of two words
# and their tenses, incorporating Wu and Palmer method


w1 = wordnet.synset('ship.n.01')
w2 = wordnet.synset('boat.n.01')


def similar(x,y):
    return x.wup_similarity(y)*100

# print similar(w1,w2) --> 91%
w3 = wordnet.synset('car.n.01')
w4 = wordnet.synset('cat.n.01')

# print similar(w1,w3) --> 69%
# print similar(w1,w4) --> 32%
