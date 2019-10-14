#predictor_api.py - contains functions to run model
#website2

import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()
	
import nltk
import re

	#Lemmatization
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

import pandas as pd

from pprint import pprint

import string

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer   

from sklearn.model_selection import KFold

from sklearn.metrics import f1_score, make_scorer

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

import pickle

import pickle_util

clf = pickle_util.load_clf('clf_bigrams_pickled')
vectorizer = pickle_util.load_vectorizer('vectorizer_bigrams_pickled')
predictor = pickle_util.load_predictor('predictor_pickled')

def predict_onion(raw_text):
	"""makes prediction of whether post is Onion or not"""
	prediction = clf.predict_proba(vectorizer.transform([predictor.preprocessing_entities(raw_text)]))
	return prediction 

def make_prediction(input_title):
	pred_probs = predict_onion(input_title)
	probs = [{'name':'Not the Onion','prob':float(pred_probs[:, 0])}, {'name':'The Onion', 
	'prob':float(pred_probs[:, 1])}]
	return(input_title, probs)

if __name__ == '__main__':
	print('Checking to see what string predicts')
	print('input string is: ')
	title_in = 'trump'
	pprint(title_in)
	x_input, probs = make_prediction(title_in)
	print(f'Input values: {x_input}')
	print('Output probabilities')
	pprint(probs)