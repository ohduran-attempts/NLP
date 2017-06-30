"""Saving the classifier using pickle (Serialization)"""


def serialize_classifier(classifier):
    import pickle
    with open('naivebayes.pickle', 'wb') as save_classifier:
        pickle.dump(classifier, save_classifier)


def unserialized_classifier():
    import pickle
    with open('naivebayes.pickle', 'rb') as classifier_f:
        return pickle.load(classifier_f)
