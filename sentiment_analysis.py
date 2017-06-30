# TEXT CLASSIFICATION

# Make out whether we are in front of a politics or military text,
# or the gender of the author. Spam or not spam, for instance.

# Sentiment analysis algorithm.
import nltk
from nltk.corpus import movie_reviews
import random

# In each category (positive/negative), take all files by their id,
# then store the word_tokenize version for each, and its label (pos/neg)
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# Let's see one of them, randomly
random.shuffle(documents)
print documents[1]

# Create a Frequency Distribution of these words
all_words = nltk.FreqDist([w.lower() for w in movie_reviews.words()])

# # Print the 15 most common words
# print all_words.most_common(15)
# # Print how many occurrences of 'stupid'
# print all_words['stupid']

word_features = list(all_words.keys())[:3000]  # top 3000 most common words


def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features


# print (find_features(movie_reviews.words('neg/cv000_29416.txt')))
featuresets = [(find_features(rev), category) for (rev, category) in documents]


# NAIVE BAYES Classifier
