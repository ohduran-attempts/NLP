"""New data set training."""
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
# Now, we can build our data using .txt files
# in a similar way.


def documenting(text1, text2):
    """Create the document list."""
    documents = []
    for r in text1.split('/n'):
        documents.append((r, 'pos'))
    for r in text2.split('/n'):
        documents.append((r, 'pos'))
    return documents
    # Now, all words. Same as before.


def all_wording(text1, text2):
    """Create the all_words."""
    all_words = []
    short_pos_words = word_tokenize(text1)
    short_neg_words = word_tokenize(text2)

    for w in short_pos_words:
        all_words.append(w.lower())
    for w in short_neg_words:
        all_words.append(w.lower())

    return nltk.FreqDist(all_words)


with open("review_text/positive.txt", "r") as f:
    short_pos = f.read

with open("review_text/negative.txt", "r") as f:
    short_neg = f.read

all_words = all_wording(short_pos, short_neg)
documents = documenting(short_pos, short_neg)

word_features = list(all_words.keys())[:5000]
