# Natural language processing
import nltk
#from nltk.tokenize import word_tokenize
#from nltk.corpus import stopwords
from nltk.classify import ClassifierI


class SklearnClassifier(ClassifierI):
	def __init__(self, classifier):
		self.classifier = classifier
		
	def classify(self, features):
		return self.classifier.classify(features)
