#predictor_api.py - contains functions to run model

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

"""
def import_file(csv_file):
	all_posts = pd.read_csv(csv_file, index_col = 0)
	return all_posts
"""

def spacy_ner(title):
	"""run spacy on title to get named entities"""
	nlp = en_core_web_sm.load()
	spacy_title = nlp(title)
	return spacy_title

def replace(string, substitutions):
	"""replace string with some substitution"""
	substrings = sorted(substitutions, key=len, reverse=True)
	regex = re.compile('|'.join(map(re.escape, substrings)))
	return regex.sub(lambda match: substitutions[match.group(0)], string)

def get_entities(title):
	"""return entities and entity types"""
	spacy_title = spacy_ner(title)
	spacy_title_ents = [str(X) for X in spacy_title.ents]
	spacy_title_ents_types = [X.label_ for X in spacy_title.ents]
	return spacy_title_ents, spacy_title_ents_types

def remove_entities(title):
    """remove entities from title"""
    entities, ent_types = get_entities(title)
    if entities == []:
        return title
    else:
        substitutions = {}
        for X in entities:
            substitutions[X] = ''
        output = replace(title, substitutions)
        return output

def preprocessing(title):
	"""full pre-processing of title including removing entities, lowercasing,
	getting rid of numbers, getting rid of punctuation, word tokenization, 
	lemmatization, getting rid of stopwords"""

	lemmatizer=WordNetLemmatizer()

	stop_words = set(stopwords.words('english'))

	title_noentities = remove_entities(title)
	title_lower = title_noentities.lower()
	title_lower_nonumbers = re.sub(r'\d+','', str(title_lower))
	no_punctuation = re.sub(r'[^\w\s]','', title_lower_nonumbers)
	tokenized_title = word_tokenize(no_punctuation)
	new_title = []
	for word in tokenized_title:
		new_word = lemmatizer.lemmatize(word)
		new_title.append(new_word)
	final_title = [i for i in new_title if not i in stop_words]
	return final_title

def preprocessing_entities(title):
    """adding back named entities and their types and appending to title"""
    title_entities, title_ents_types = get_entities(title)
    title_noentities = preprocessing(title)
    title_all = title_noentities + title_entities + title_ents_types
    return title_all

"""

def get_all_titles_final(csv_file):
	all_posts = import_file(csv_file)
	all_titles_ents = all_posts['title'].apply(lambda x: preprocessing_entities(x))
	final_posts_df = pd.DataFrame({'title':all_titles_ents, 'Onion?':all_posts['Onion?']})
	return final_posts_df

"""

"""
def dummy_fun(doc):
    #just a dummy function
    return doc

def make_xy_bigrams(df, vectorizer=None):
    #making X and y arrays
    if vectorizer is None:
    	vectorizer = CountVectorizer(tokenizer=dummy_fun,
    	preprocessor=dummy_fun, ngram_range=(1,2))
    X = vectorizer.fit_transform(df['title'])
    X = X.tocsc()  # some versions of sklearn return COO format
    y = df['Onion?'].values.astype(np.int)
    return X, y
"""
"""
def cv_score(clf, X, y, scorefunc):
	result = 0.
	nfold = 5
	for train, test in KFold(nfold, random_state=42).split(X): # split data into train/test groups, 5 times
		clf.fit(X[train], y[train]) # fit the classifier, passed is as clf.
		result += scorefunc(clf, X[test], y[test]) # evaluate score function on held-out data
	return result / nfold # average

def make_f1_scorer():
	f1_scorer = make_scorer(f1_score)
	return f1_scorer

"""
"""
def get_best_alpha(alphas_list, csv_file):
	best_alpha = None
	maxscore = -np.inf

	for alpha in alphas_list:
		vectorizer = CountVectorizer(min_df = 5, tokenizer = dummy_fun, \
			preprocessor=dummy_fun, ngram_range=(1,2))
		final_posts_df = get_all_titles_final(csv_file)
		Xthis, ythis = make_xy_bigrams(final_posts_df, vectorizer)
		X_train_this, X_test_this, y_train_this, y_test_this = \
		train_test_split(Xthis, ythis, test_size=0.2, random_state=42)
		clf = MultinomialNB(alpha=alpha).fit(X_train_this, y_train_this)
		f1_scorer = make_f1_scorer()
		score = cv_score(clf, X_train_this, y_train_this, f1_scorer)
		if score > maxscore:
			maxscore = score
			best_alpha = alpha
		return best_alpha 
"""

"""

def train_and_fit(csv_file, alphas_list):
	vectorizer = CountVectorizer(min_df = 5, tokenizer = dummy_fun, 
		preprocessor=dummy_fun, ngram_range=(1,2))
	final_posts_df = get_all_titles_final(csv_file)
	X, y = make_xy_bigrams(final_posts_df, vectorizer)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	best_alpha = get_best_alpha(alphas_list, csv_file)
	clf = MultinomialNB(alpha=best_alpha).fit(X_train, y_train)
	return clf, vectorizer
"""

def predict_onion(raw_text):
	"""makes prediction of whether post is Onion or not"""
	clf = pickle.load(open('pickled_model_bigrams.pkl', 'rb'))
	vectorizer = pickle.load(open('pickled_model_bigrams_vectorizer.pkl', 'rb'))
	prediction = clf.predict_proba(vectorizer.transform([preprocessing_entities(raw_text)]))
	return prediction #FIX THIS LATER - CONVERT ARRAY TO LIST

def make_prediction(input_title):
	pred_probs = predict_onion(input_title)

	probs = [{'name':'Not the Onion','prob':float(pred_probs[:, 0])}, {'name':'The Onion', 
	'prob':float(pred_probs[:, 1])}]
	return(input_title, probs)

if __name__ == '__main__':
	print('Checking to see what string predicts')
	print('input string is: ')
	title_in = 'Lindsey Graham prank called by Russians pretending to be Turkish defense minister'
	pprint(title_in)
	x_input, probs = make_prediction(title_in)
	print(f'Input values: {x_input}')
	print('Output probabilities')
	pprint(probs)