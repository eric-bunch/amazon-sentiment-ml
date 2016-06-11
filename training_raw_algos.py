from sys import argv

# Natural language processing
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Machine learning
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn import svm
from sklearn import tree
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from nltk.classify import ClassifierI

# Misc
from statistics import mode
import pickle
import random
import string
import gzip
import math
from collections import Counter, defaultdict
import numpy as np


script, genre = argv

stop_words = set(stopwords.words("english"))
all_words = []
documents = []
#reviewScoreHash = {}

def to_binary(score):
	if score >= 4:
		return "pos"
	else:
		return "neg"



def find_features(document):
	words = set(document)
	features = {}
	for w in word_features:
		features[w] = (w in words)

	return features


with gzip.open(genre + '_train.txt.gz') as f:
	for index, line in enumerate(f):
		if index % 2 == 0 and len(line.split('\t')[1]) <= 140 and line.split('\t')[0] != 3:
			score = int(line.split('\t')[0])
			text = line.split('\t')[1]
			#review_text = [w.lower() for w in word_tokenize(text) if w.isalpha() and not w in stop_words]
			review_text = [w.lower() for w in word_tokenize(text) if w.isalpha()]

			all_words += review_text
			documents.append( (review_text, to_binary(score)) )


random.shuffle(documents)

all_words = nltk.FreqDist(all_words)
number_of_words = len(all_words.keys())
word_feature_percentage = 0.1

print "Total number of words: ", number_of_words

#word_features = list(all_words.keys())[:1000]
word_features = list(all_words.keys()[:int(math.floor( number_of_words * word_feature_percentage ))])
#word_features = list(all_words.keys()[:1000])

featuresets = [(find_features(rev), category) for (rev, category) in documents]

ml_hash = {SklearnClassifier(MultinomialNB()): 'Multinomial Naive Bayes', SklearnClassifier(BernoulliNB()): 'Bernoulli Naive Bayes',
				SklearnClassifier(LogisticRegression()): 'Logistic Regression', SklearnClassifier(SGDClassifier()): 'SGD Classifier',
				SklearnClassifier(LinearSVC()): 'Linear SVC' }


for algo in ml_hash.keys():
	classifier = algo.train(featuresets)
	with open(genre + " classifiers/Twitter/" + ml_hash[algo] + ".pickle", "wb") as save_classifier:
		pickle.dump(classifier, save_classifier)
