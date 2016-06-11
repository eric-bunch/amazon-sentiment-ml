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
from sentiment_lexicon_classifiers import LexiconClassifier
from lexicon_algos_classifiers import LexiconAlgoClassifier
from voted_classifier import scale_amaz_to_afinn, to_binary, find_features, get_word_features, VoteClassifier


# Misc
import os
from statistics import mode
import shutil
from shutil import copyfile
import pickle
import random
import string
import gzip
import math
from collections import Counter, defaultdict
import numpy as np
#from voted_classifier import scale_amaz_to_afinn, to_binary, find_features, get_word_features




def scale_afinn_to_amazon(score):
	return 0.4*(int(score)) + 3




script, genre = argv

stop_words = set(stopwords.words("english"))


word_features = get_word_features(genre)
#word_features += amazon.keys()

documents = []


with gzip.open(genre + '_test.txt.gz') as f:
	for index, line in enumerate(f):
		if index % 2 == 0 and len(line.split('\t')[1]) <= 140:
			score = int(line.split('\t')[0])
			text = line.split('\t')[1]
			#review_text = [w.lower() for w in word_tokenize(text) if w.isalpha() and not w in stop_words]
			review_text = [w.lower() for w in word_tokenize(text) if w.isalpha()]

			documents.append( (review_text, to_binary(score)) )


random.shuffle(documents)

featuresets = [(find_features(rev, word_features), category) for (rev, category) in documents]

ml_raw_hash = {SklearnClassifier(MultinomialNB()): 'Multinomial Naive Bayes', SklearnClassifier(BernoulliNB()): 'Bernoulli Naive Bayes',
				SklearnClassifier(LogisticRegression()): 'Logistic Regression', SklearnClassifier(SGDClassifier()): 'SGD Classifier',
				SklearnClassifier(LinearSVC()): 'Linear SVC' }

out = open(genre + "_test_output.txt", "w")

for algo in ml_raw_hash:
	with open(genre + " classifiers/Twitter/" + ml_raw_hash[algo] + ".pickle", "rb") as classifier_file:
		classifier = pickle.load(classifier_file)
	accuracy = (nltk.classify.accuracy(classifier, featuresets)*100)
	print ml_raw_hash[algo] + " accuracy percent: ",accuracy
	out.write(ml_raw_hash[algo] + " accuracy percent: ")
	out.write(str(accuracy))
	out.write("\n")
	# if accuracy >= 63:
	# 	copyfile(genre + " classifiers/Twitter/" + ml_raw_hash[algo] + ".pickle",
	# 										genre + " classifiers/Twitter/Good/" + ml_raw_hash[algo] + ".pickle")

#ml_hash = {svm.SVC() : 'Support vector machine', tree.DecisionTreeClassifier() : 'Decision tree classifier', GaussianNB() : 'Naive Bayes (Gaussian)',
#					KNeighborsClassifier(n_neighbors=3) : '3-Nearest Neighbors', linear_model.LogisticRegression() : 'Logistic Regression', RandomForestClassifier() : 'Random Forest'}

# afinn = dict(map(lambda (k,v): (k, int(v) ),
#                      [ line.split('\t') for line in open("AFINN-111.txt") ]))
#
# afinn = defaultdict(lambda: 0, afinn)
#
# afinn_scaled = dict(map(lambda (k,v): (k, scale_afinn_to_amazon(int(v)) ),
#                      [ line.split('\t') for line in open("AFINN-111.txt") ]))
#
# afinn_scaled = defaultdict(lambda: 0, afinn_scaled)
#
#
# amazon_scaled = dict(map(lambda (k,v): (k, scale_amaz_to_afinn(int(v)) ),
#                      [ line.split('\t') for line in open(genre + " amazon sentiment lexicon.tsv") ]))
#
#
# amazon_scaled = defaultdict(lambda: 0, amazon_scaled)
#
#
# amazon = dict(map(lambda (k,v): (k, int(v) ),
#                      [ line.split('\t') for line in open(genre + " amazon sentiment lexicon.tsv") ]))
#
#
# amazon = defaultdict(lambda: 0, amazon)




"""
for algo in ml_hash:
	for (lexicon, lexicon_name) in [(afinn, "afinn"), (amazon, "amazon")]:
		with open(genre + " classifiers/Twitter/" + genre + " " + lexicon_name + " " + ml_hash[algo] + ".pickle", "rb") as classifier_file:
			classifier = LexiconAlgoClassifier(pickle.load(classifier_file), lexicon)


		accuracy = (nltk.classify.accuracy(classifier, featuresets)*100)
		print ml_hash[algo] + " " + lexicon_name + " accuracy percent: ",accuracy
		if accuracy >= 68:
			copyfile(genre + " classifiers/Twitter/" + genre + " " + lexicon_name + " " + ml_hash[algo] + ".pickle",
												genre + " classifiers/Twitter/Good/" + genre + " " + lexicon_name + " " + ml_hash[algo] + ".pickle")

"""
#list_of_classifiers = []

# lexicons = [(afinn, 'afinn'), (amazon_scaled, 'amazon')]
#
# for lexicon in lexicons:
# 	classifier = LexiconClassifier(lexicon[0])
# 	print lexicon[1] + " lexicon accuracy percent: ",(nltk.classify.accuracy(classifier, featuresets)*100)
# 	out.write(lexicon[1] + " lexicon accuracy percent: ")
# 	out.write(str(nltk.classify.accuracy(classifier, featuresets)*100))
# 	out.write("\n")


"""
for filename in os.listdir(os.getcwd() + "/" + genre + " classifiers/Twitter/Good/"):
	with open(genre + " classifiers/Twitter/Good/" + filename) as classifier_file:
		if genre in filename and 'afinn' in filename:
			list_of_classifiers.append(LexiconAlgoClassifier( pickle.load(classifier_file), afinn_scaled ))
		elif genre in filename and 'amazon' in filename:
			list_of_classifiers.append(LexiconAlgoClassifier(pickle.load(classifier_file), amazon ))
		else:
			list_of_classifiers.append(SklearnClassifier(pickle.load(classifier_file)))
"""

#list_of_classifiers.append(LexiconClassifier(amazon_scaled))
#list_of_classifiers.append(LexiconClassifier(afinn))


#voted_classifier = VoteClassifier(list_of_classifiers)

#print "Voted classifier accuracy percent: ",(nltk.classify.accuracy(VoteClassifier(list_of_classifiers), featuresets)*100)





out.close()
