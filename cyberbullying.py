# Data processing tools
import praw
import pickle
import pandas
import numpy

# Machine learning tools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.test import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

# Natural language processing tools
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
import re


class CyberbullyingDetectionEngine:
    """ Deals with training and deploying cyberbullying detection"""
    def __init__(self):
        self.corpus = None
        self.tags = None
        self.lexicons = None
        self.vectorizer = None
        self.model = None
        self.metrics = None

    class CustomVectorizer:
        """Extracts features from text and vectorizes"""
        def __init__(self, lexicons):
            self.lexicons = lexicons

        def transform(self, corpus):
            """Returns numpy array of word vectors"""
            word_vectors = []
            for text in corpus:
                features = []
                for k, v in self.lexicons.items():
                    features.append(len([w for w in word_tokenize(text)
                        if w in v]))

                    word_vectors.append(features)

                return numpy.array(word_vectors)

    def _simplify(self, corpus):
        """Takes in list of strings and removes stopwords, converts to
        lowercase, removes non-alphanumeric characters, and stems
        each word """

        stop_words = set(stopwords.words('english'))
        stemmer = SnowballStemmer('english')

        def clean(text):
            text = re.sub('[^a-zA-Z0-9]', ' ', text)
            words = [stemmer.stem(w) for w in word-tokenize(text.lower())
                if w not in stop_words]
            return " ".join(words)

        return [clean(text) f0r text in corpus]

    def _get_lexicon(self, path):
        """Takes in path to text file and returns set containing every word
        in file"""
        words = set()
        with open(path) as file:
            for line in file:
                words.update(line.strip().split(' '))

        return words

    def _model_metrics(self, features, tags):
        """Takes in testing data and returns dictionary of metrics"""
        tp = 0 #total positive
        fp = 0 #false positive
        tn = 0
        fn = 0

        predictions = self.model.predict(features)
        for r in zip(predictions, tags):
            if r[0] == 1 and r[1] == 1:
                tp += 1
            elif r[0] == 1 and r[1] == 0:
                fp += 1
            elif r[0] == 0 and r[1] == 1:
                fn += 1
            else:
                tn += 1

            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            return {
                'precision': precision,
                'recall': recall,
                'f1': (2 * precision * recall) / (precision + recall)
            }

    def load_corpus(self, path, corpus_col, tag_col):
        """Takes in path to pickled pandas dataframe, name of corpus column,
        and name of tag column, extracting tagged corpus"""
        data = pandas.read_pickle(path)[[corpus_col, tag_col]].values
        self.corpus = [row[0] for row in data]
        self.tags = [row[1] for row in data]

    def load-lexicon(self, fname):
        """Loads set of words from text file"""
        if self.lexicons is None:
            self.lexicons = {}

        self.lexicons[fname] = self._get_lexicon('./data/' + fname + 'txt')

    def load_model(self, model_name):
        """Loads ML model, corresponding feature vectorizer, and performance
        metrics"""
        self.model = pickle.load(open('./models/' + model_name +
            '_ml_model.pkl', 'rb'))
        self.vectorizer = pickle.load(open('./models/' + model_name +
            '_vectorizer.pkl', 'rb'))
        self.metrics = pickle.load(open('./models/' + model_name +
            '_metrics.pkl', 'rb'))

    def train_using_bow(self):
        """Trans model using Bag of Words on loaded corpus and tags"""
        corpus = self._simplify(self.corpus)
        self.vectorizer = CountVectorizer()
        self.vectorizer.fit(corpus)

        bag_of_words = self.vectorizer.transform(corpus)
        x_train, x_test, y_train, y_test = train_test_split(bag_of_words,
            self.tags, test_size = 0.2, stratify = self.tags)

        self.model = MultinomialNB()
        self.model.fit(x_train, y_train)

        self.metrics = self._model_metrics(x_test, y_test)

    def train_using_tfidf(self):
