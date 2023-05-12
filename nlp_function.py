import nltk
import string
import numpy as np

from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

nltk.download("punkt")
nltk.download("stopwords")
stopwords_list = stopwords.words("indonesian")

factory = StemmerFactory()
stemmer = factory.create_stemmer()


def tokenization(raw_text):
    return nltk.word_tokenize(raw_text.lower())


def remove_punctuation(token):
    unPunctuation_token = [word for word in token if word not in string.punctuation]
    return unPunctuation_token


def remove_stopWords(unPunctuation_token):
    unStopWords_token = [
        word for word in unPunctuation_token if word not in stopwords_list
    ]
    return unStopWords_token


def stemming_token(unStopWords_token):
    return stemmer.stem(unStopWords_token)


def vectorization(clean_token, all_token):
    bag = np.zeros(len(all_token), dtype=np.float32)
    for idx, word in enumerate(all_token):
        if word in clean_token:
            bag[idx] = 1.0

    return bag
