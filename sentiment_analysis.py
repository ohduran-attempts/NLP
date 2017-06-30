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
# random.shuffle(documents)
# print documents[1]

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

# set the training set, the first half of data of featuresets
training_set = featuresets[:1900]
testing_set = featuresets[1900:]

# Define, and train, our classifier:
classifier = nltk.NaiveBayesClassifier.train(training_set)

# What is the accuracy level?
print "Classifier accuracy: ",
print nltk.classify.accuracy(classifier, testing_set) * 100,
print " percent."

# This will show the most common words for positive and negatives
# print classifier.show_most_informative_features(15)
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify.scikitlearn import SklearnClassifier

print "Original Naive Bayes Algo accuracy percent:",
print nltk.classify.accuracy(classifier, testing_set) * 100
classifier.show_most_informative_features(15)

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print "MNB_classifier accuracy percent:",
nltk.classify.accuracy(MNB_classifier, testing_set) * 100

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print "BernoulliNB_classifier accuracy percent:",
print nltk.classify.accuracy(BernoulliNB_classifier, testing_set) * 100

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print "LogisticRegression_classifier accuracy percent:",
print nltk.classify.accuracy(LogisticRegression_classifier, testing_set) * 100

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print "SGDClassifier_classifier accuracy percent:",
print nltk.classify.accuracy(SGDClassifier_classifier, testing_set) * 100

SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
print "SVC_classifier accuracy percent:",
print nltk.classify.accuracy(SVC_classifier, testing_set) * 100

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print "LinearSVC_classifier accuracy percent:",
print nltk.classify.accuracy(LinearSVC_classifier, testing_set) * 100

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print "NuSVC_classifier accuracy percent:",
print nltk.classify.accuracy(NuSVC_classifier, testing_set) * 100


# MNB			- 66%
# Bernouuli	    - 64%
# LogReg        - 72%
# SGD           - 52%
# SVC 		    -  0%
# LinearSVC     - 67%
# NuSVC 		- 64%

# Sure, we can dump SVC. But what if we combine algorithms?

# Using OO programming, we can make a Classifier inherit from all others.

from nltk.classify import ClassifierI
from statistics import mode


class VoteClassifier(ClassifierI):
    """The all-at-once classifier."""

    def __init__(self, *classifiers):
        """Init contains all classifiers."""
        self.classifiers = classifiers

    def classify(self, features):
        """Classification method."""
        votes = []
        for c in self.classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        """Confidence of the votes."""
        votes = []
        for c in self.classifiers:
            v = c.classify(features)
            votes.append(v)
        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf


# Now let's put it all together
voted_classifier = VoteClassifier(classifier,
                                  NuSVC_classifier,
								  LinearSVC_classifier,
								  SGDClassifier_classifier,
								  MNB_classifier,
								  BernoulliNB_classifier,
								  LogisticRegression_classifier)

print("voted_classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, testing_set))*100)
print("Classification:", voted_classifier.classify(testing_set[0][0]), "Confidence %:",voted_classifier.confidence(testing_set[0][0])*100)
print("Classification:", voted_classifier.classify(testing_set[1][0]), "Confidence %:",voted_classifier.confidence(testing_set[1][0])*100)
print("Classification:", voted_classifier.classify(testing_set[2][0]), "Confidence %:",voted_classifier.confidence(testing_set[2][0])*100)
print("Classification:", voted_classifier.classify(testing_set[3][0]), "Confidence %:",voted_classifier.confidence(testing_set[3][0])*100)
print("Classification:", voted_classifier.classify(testing_set[4][0]), "Confidence %:",voted_classifier.confidence(testing_set[4][0])*100)
print("Classification:", voted_classifier.classify(testing_set[5][0]), "Confidence %:",voted_classifier.confidence(testing_set[5][0])*100)
