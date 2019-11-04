import pickle

def save_clf(doc):

	from class_def import Predictor

	write_file = open('clf_bigrams_pickled', 'wb')
	pickle.dump(doc, write_file)

def save_vectorizer(doc):

	from class_def import Predictor

	write_file = open('vectorizer_bigrams_pickled', 'wb')
	pickle.dump(doc, write_file)

def save_predictor(doc):

	from class_def import Predictor

	write_file = open('predictor_pickled', 'wb')
	pickle.dump(doc, write_file)

def load_clf(file_path):
	from class_def import Predictor

	doc_file = open(file_path, 'rb')
	return pickle.load(doc_file)

def load_vectorizer(file_path):
	from class_def import Predictor

	doc_file = open(file_path, 'rb')
	return pickle.load(doc_file)

def load_predictor(file_path):
	from class_def import Predictor

	doc_file = open(file_path, 'rb')
	return pickle.load(doc_file)
