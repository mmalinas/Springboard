#import spacy
#from spacy import displacy
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