from flask import Flask, render_template, request
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import os
import numpy as np

import matplotlib.pyplot as plt
import re
from pickle import dump
from pickle import load

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer


import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LSTM, Input, Dense, Embedding, TimeDistributed
from tensorflow.keras.models import Sequential

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

app = Flask(__name__)

# # Load the pickled model and word-to-id mapping
# with open('lstm_model.pkl', 'rb') as file:
#     pickle_model = pickle.load(file)

model_h5_filename = "lstm_w_770.h5"
loaded_model = load_model(model_h5_filename)
# loaded_model = tf.keras.models.load_model(model_h5_filename , custom_objects={'Rating': model_by_name})

# Load vocab and word-to-id mapping
temp_file = open('x_train', 'rb')
x_train = pickle.load(temp_file)
temp_file.close()

vocab = set()
for sentence in x_train:
    for word in sentence:
        vocab.add(word)

vocab.add('')  # for dummy words, to avoid adding a word that has a meaning

word2id = {word: id for id, word in enumerate(vocab)}

MAX_SEQ_LEN = 125
dummy = word2id['']

@app.route('/', methods=['GET', 'POST'])
def index():
    result = ''
    
    if request.method == 'POST':
        book_name = request.form.get('bookName')
        review = request.form.get('review')
        # Perform sentiment analysis here
        prediction = lstm_predict(review, loaded_model)
        # result = f"Book: {book_name}\nReview: {review}\nPrediction: {prediction}"
        result = prediction
    return render_template('index.html', result=result)

def clean_sentence(sentence: str) -> list:
  tags = re.compile("(<review_text>|<\/review_text>)")
  sentence = re.sub(tags, '', sentence)
  sentence = sentence.lower()
  email_urls = re.compile("(\bhttp.+? | \b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b)")
  sentence = re.sub(email_urls, '', sentence)
  ats = re.compile('@')
  sentence = re.sub(ats, 'a', sentence)
  punc = re.compile("[^\w\s(\w+\-\w+)]")
  sentence = re.sub(punc, '', sentence)
  sentence = word_tokenize(sentence)
  sentence = [word for word in sentence if not word in stopwords.words()]
  return sentence

def encode_sentence(old_sentence):
    encoded_sentence = []
    dummy = word2id['']
    for word in old_sentence:
        try:
            encoded_sentence.append(word2id[word])
        except KeyError:
            encoded_sentence.append(dummy)  # the none char
    return encoded_sentence


def lstm_predict(sentence:str, loaded_model):
  sentence = clean_sentence(sentence)
  ready_sentence = encode_sentence(sentence)
  ready_sentence = pad_sequences(sequences = [ready_sentence],
                                 maxlen=MAX_SEQ_LEN,
                                 dtype='int32',
                                 padding='post',
                                 truncating='post',
                                 value = dummy)

  # Predict
  prediction = round(loaded_model.predict(ready_sentence)[0][0])
#   prediction = loaded_model.predict(ready_sentence)[0][0]
  print (prediction)
  if prediction==0:
    return "Negative Review"
  elif prediction==1:
    return "Positive Review"
  else:
    print('Error')

if __name__ == '__main__':
    app.run(debug=True)
