#predictor_api.py - contains functions to run model
#website2

import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()
	
import nltk
import re
import os

	#Lemmatization
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

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
	title_entities, title_ents_types = predictor.get_entities(input_title)
	title_noentities = predictor.preprocessing(input_title)
	whole_title = title_entities + title_noentities
	probs_individual = []
	for element in whole_title:
		prob = predict_onion(element)
		probs_individual.append((prob, element))

	probs = [{'name':'Not the Onion','prob':round(float(pred_probs[:, 0]),2)}, {'name':'The Onion', 
	'prob':round(float(pred_probs[:, 1]), 2)}]
	for i in range(len(probs_individual)):
		probs.append({'entity/word':probs_individual[i][1], 'Not the Onion prob':round(float(probs_individual[i][0][:,0]),2),
			'Onion prob':round(float(probs_individual[i][0][:,1]),2)})
	return(input_title, probs)

if __name__ == '__main__':
	print('Checking to see what string predicts')
	print('input string is: ')
	title_in = 'nation'
	pprint(title_in)
	x_input, probs = make_prediction(title_in)
	print(f'Input values: {x_input}')
	print('Output probabilities')
	pprint(probs)