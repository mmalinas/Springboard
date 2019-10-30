import pickle_util
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

lemmatizer=WordNetLemmatizer()

stop_words = set(stopwords.words('english'))

class Predictor:
	"""initialize Predictor class with csv file and list of alphas"""
	def __init__(self, csv_file, alphas_list):
		self.csv_file = csv_file
		self.alphas_list = alphas_list

	def import_file(self):
		all_posts = pd.read_csv(self.csv_file, index_col = 0, sep='|') #reads csv file
		return all_posts

	def spacy_ner(self, title):
		"""run spacy on title to get named entities"""
		spacy_title = nlp(title) #applies spacy nlp to title
		return spacy_title

	def replace(self, string, substitutions):
		"""replace string with some substitution"""
		substrings = sorted(substitutions, key=len, reverse=True)
		regex = re.compile('|'.join(map(re.escape, substrings)))
		return regex.sub(lambda match: substitutions[match.group(0)], string)

	def get_entities(self, title):
		"""return entities and entity types"""
		spacy_title = self.spacy_ner(title)
		spacy_title_ents = [str(X) for X in spacy_title.ents]
		spacy_title_ents_types = [X.label_ for X in spacy_title.ents]
		return spacy_title_ents, spacy_title_ents_types

	def remove_entities(self, title):
		"""remove entities from title"""
		entities, ent_types = self.get_entities(title)
		if entities == []: #if there are no entities, just return the title
			return title
		else: #else, substitute entity with an empty string
			substitutions = {} 
			for X in entities:
				substitutions[X] = ''
			output = self.replace(title, substitutions)
			return output

	def preprocessing(self, title):
		"""full pre-processing of title including removing entities, lowercasing,
		getting rid of numbers, getting rid of punctuation, word tokenization, 
		lemmatization, getting rid of stopwords"""
		title_noentities = self.remove_entities(title) #remove entities
		title_lower = title_noentities.lower() #make lowercase
		title_lower_nonumbers = re.sub(r'\d+','', str(title_lower)) #subsitute numbers using regex
		no_punctuation = re.sub(r'[^\w\s]','', title_lower_nonumbers) #substitute punctuation using regex
		tokenized_title = word_tokenize(no_punctuation) #tokenize sentence
		new_title = []
		for word in tokenized_title:
			new_word = lemmatizer.lemmatize(word) #lemmatize words
			new_title.append(new_word)
		final_title = [i for i in new_title if not i in stop_words]
		return final_title

	def preprocessing_entities(self, title):
		"""adding back named entities and their types and appending to title"""
		title_entities, title_ents_types = self.get_entities(title)
		title_noentities = self.preprocessing(title)
		title_all = title_noentities + title_entities + title_ents_types 
		return title_all

	def get_all_titles_final(self):
		all_posts = self.import_file()
		all_titles_ents = all_posts['title'].apply(lambda x: self.preprocessing_entities(x)) #preprocess all titles
		final_posts_df = pd.DataFrame({'title':all_titles_ents, 'Onion?':all_posts['Onion?']}) #make DF with processed titles
		return final_posts_df

	def dummy_fun(self, doc):
		#just a dummy function
		return doc

	def make_xy_bigrams(self, df, vectorizer=None):
		#making X and y arrays
		if vectorizer is None:
			vectorizer = CountVectorizer(tokenizer=self.dummy_fun,
				preprocessor=self.dummy_fun, ngram_range=(1,2))
		X = vectorizer.fit_transform(df['title'])
		X = X.tocsc()  # some versions of sklearn return COO format
		y = df['Onion?'].values.astype(np.int)
		return X, y

	def cv_score(self, clf, X, y, scorefunc):
		result = 0.
		nfold = 5
		for train, test in KFold(nfold, random_state=42).split(X): # split data into train/test groups, 5 times
			clf.fit(X[train], y[train]) # fit the classifier, passed is as clf.
			result += scorefunc(clf, X[test], y[test]) # evaluate score function on held-out data
		return result / nfold # average

	def make_f1_scorer(self):
		f1_scorer = make_scorer(f1_score)
		return f1_scorer

	def get_best_alpha(self):
		best_alpha = None
		maxscore = -np.inf

		for alpha in self.alphas_list:
			vectorizer = CountVectorizer(min_df = 5, tokenizer = self.dummy_fun, \
				preprocessor=self.dummy_fun, ngram_range=(1,2)) #make vectorizer
			final_posts_df = self.get_all_titles_final() #make dataframe with final titles
			Xthis, ythis = self.make_xy_bigrams(final_posts_df, vectorizer) #make X and y arrays
			X_train_this, X_test_this, y_train_this, y_test_this = \
			train_test_split(Xthis, ythis, test_size=0.2, random_state=42) #get training and test sets
			clf = MultinomialNB(alpha=alpha).fit(X_train_this, y_train_this) #make model
			f1_scorer = self.make_f1_scorer()
			score = self.cv_score(clf, X_train_this, y_train_this, f1_scorer) #get CV score
			if score > maxscore:
				maxscore = score
				best_alpha = alpha
		#print(best_alpha)
		return best_alpha 

	def train_and_fit(self):
		vectorizer = CountVectorizer(min_df = 5, tokenizer = self.dummy_fun, #make CountVectorizer
			preprocessor=self.dummy_fun, ngram_range=(1,2))
		final_posts_df = self.get_all_titles_final() #get all titles
		X, y = self.make_xy_bigrams(final_posts_df, vectorizer)
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
		best_alpha = self.get_best_alpha()
		clf = MultinomialNB(alpha=best_alpha).fit(X_train, y_train)
		return clf, vectorizer

if __name__ == '__main__':
	from class_def import Predictor
	doc = Predictor('all_posts_reddit_onionandnotonion_2.csv', [.1, 1, 5, 10, 50]) #make Predictor instance
	clf, vectorizer = doc.train_and_fit()
	pickle_util.save_clf(clf) #save model
	pickle_util.save_vectorizer(vectorizer) #save vectorizer
	pickle_util.save_predictor(doc) #save whole predictor
	print('success!')