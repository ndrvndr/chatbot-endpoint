import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords_list = stopwords.words('indonesian')
stopwords_list.extend(new_stopword)

import string
import numpy as np

import Sastrawi
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def tokenization(raw_text):
    return nltk.word_tokenize(raw_text.lower())

def remove_punctuation(token):
    unPunctuation_token = [word for word in token if word not in string.punctuation]
    return unPunctuation_token

def remove_stopWords(unPunctuation_token):
    unStopWords_token = [word for word in unPunctuation_token if word not in stopwords_list]
    return unStopWords_token

def stemming_token(unStopWords_token):
    return stemmer.stem(unStopWords_token)

def vectorization(clean_token, all_token):
    tokenize_text =  [stemming_token(word) for word in clean_token]

    bag = np.zeros(len(all_token), dtype=np.float32)
    for idx, word in enumerate(all_token):
        if word in clean_token:
            bag[idx] = 1.0
    
    return bag

sentence = [
        "Apakah ada kos yang dekat dengan UVERS?",
        "Apakah Universitas Universal memiliki nama kos-kosan yang dekat dengan kampus?",
        "Apakah universitas ini memiliki layanan bantuan dalam mencari kos-kosan yang sesuai dengan kebutuhan calon mahasiswa?",
        "Bagaimana cara mencari kos-kosan yang dekat dengan universitas? Apakah ada sumber atau media yang disarankan?",
        "Apakah kos-kosan dekat universitas terdapat fasilitas yang memadai seperti internet, tempat parkir, atau akses transportasi umum?",
        "Apakah kos-kosan dekat universitas tersedia dalam berbagai tipe kamar atau hanya satu tipe kamar saja?",
        "Berapa harga rata-rata kos-kosan yang dekat dengan universitas?"
      ]

# for s in sentence:
#     tokens = tokenization(s)
#     tokens = remove_punctuation(tokens)
#     tokens = remove_stopWords(tokens)
#     tokens = [stemming_token(w) for w in tokens]
#     print(tokens)

# all_words = ["the", "red", "dog", "cat", "eats", "food"]

# sentence = ["the red dog", "cat eats dog", "dog eats food", "red cat eats"]

# token = [tokenization(sentence[0]), 
#              tokenization(sentence[1]),
#              tokenization(sentence[2]),
#              tokenization(sentence[3])]

# bag = [vectorization(token[0], all_words),
#            vectorization(token[1], all_words),
#            vectorization(token[2], all_words),
#            vectorization(token[3], all_words)]

# for bag in bag:
#     print(bag)
