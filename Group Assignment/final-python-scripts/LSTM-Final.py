# -*- coding: utf-8 -*-
'''
This file contains the trial for the final model run - LSTM + Bidirectiional + Postprocessing- with Hyperparameter Tuning.
For best results, run the corresponding Python file.
The outputs have been presented and discussed in the report.
'''
#Import necessary libraries
import nltk
import pandas as pd
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Embedding, LSTM, Dropout, Bidirectional
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import re
import tensorflow as tf
from tensorflow import keras
import numpy as np
from nltk.tokenize import word_tokenize
from tensorflow.keras.optimizers import Adam 
from keras.constraints import maxnorm

#Reading in the dataset
tweetData = pd.read_csv('../data/Feature-Engineered.csv', index_col=False)

# Added in to avoid formatting error
labels = np.array(tweetData['tweettype'])
y = []
for i in range(len(labels)):
    if labels[i] == 'sadness':
        y.append(0)
    elif labels[i] == 'neutral':
        y.append(1)
    elif labels[i] == 'joy':
        y.append(2)
    elif labels[i] == 'love':
        y.append(3)
    elif labels[i] == 'enthusiasm':
        y.append(4)
    elif labels[i] == 'anger':
        y.append(5)
    elif labels[i] == 'surprise':
        y.append(6)
    elif labels[i] == 'relief':
        y.append(7)
    elif labels[i] == 'fear':
        y.append(8)
y = np.array(y)
labels = tf.keras.utils.to_categorical(y, 9, dtype="float32")
del y


def featureEngineering(tweet):
    # Lower case tweet
    tweetMod = tweet.lower()
    # Replace URLs with a space in the message
    tweetMod = re.sub('https?:\/\/[a-zA-Z0-9@:%._\/+~#=?&;-]*', ' ', tweetMod)
    # Replace ticker symbols with a space. The ticker symbols are any stock symbol that starts with $.
    tweetMod = re.sub('\$[a-zA-Z0-9]*', ' ', tweetMod)
    # Replace StockTwits usernames with a space. The usernames are any word that starts with @.
    tweetMod = re.sub('\@[a-zA-Z0-9]*', ' ', tweetMod)
    # Replace everything not a letter or apostrophe with a space
    tweetMod = re.sub('[^a-zA-Z\']', ' ', tweetMod)
    # Remove single letter words
    tweetMod = ' '.join([w for w in tweetMod.split() if len(w) > 1])

    return tweetMod


# Process for all tweets
tweetData['modTweet'] = [featureEngineering(tweet) for tweet in tweetData['tweet']]

def lemmatizeTweet(tweet):
  words = [word for word in word_tokenize(tweet) if (word.isalpha()==1)]
  # Remove stop words
  stop = set(stopwords.words('english'))
  words = [word for word in words if (word not in stop)]
  # Lemmatize words (first noun, then verb)
  wnl = nltk.stem.WordNetLemmatizer()
  lemmatized = [wnl.lemmatize(wnl.lemmatize(word, 'n'), 'v') for word in words]
  return " ".join(lemmatized)

tweetData['lemmatizedText'] = tweetData["modTweet"].apply(lambda x:lemmatizeTweet(x))

#Tokenize the input data
tokenizer = Tokenizer(num_words=27608, split=' ')
tokenizer.fit_on_texts(tweetData['lemmatizedText'].values)
X = tokenizer.texts_to_sequences(tweetData['lemmatizedText'].values)
X = pad_sequences(X)

#Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

"""
Basic Model Structure
Embedding + 2 Bidirectional layers + Dropouts
"""

keras.backend.clear_session()
model_dropout = Sequential()
model_dropout.add(Embedding(input_dim = 128,output_dim = 8,input_length = X.shape[1]))
model_dropout.add(Dropout(rate=0.5))
model_dropout.add(Bidirectional(LSTM(units=256, kernel_initializer= 'normal', return_sequences=True, kernel_constraint=maxnorm(4))))
model_dropout.add(Dropout(rate=0.5))
model_dropout.add(Bidirectional(LSTM(units=128, kernel_initializer= 'normal', return_sequences=False)))
model_dropout.add(Dense(9, activation='softmax'))
optimizer = Adam(lr=0.001)

#Compile and test model
model_dropout.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['categorical_accuracy'])
history = model_dropout.fit(X_train, Y_train, epochs = 50, batch_size=512, validation_data=(X_test, Y_test))

#Plotting the training accuracies
plt.figure(1)
plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.savefig('LSTM-Bidirectional-Final-Accuracy.png')

#Plotting the losses
plt.figure(2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('cross-entropy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.savefig('LSTM-Bidirectional-Final-Loss.png')

"""Model trial after clubbing output labels"""
tweetData = pd.read_csv('../data/Postprocessed-Feature-Engineered.csv', index_col=False)

#New labels
labels = np.array(tweetData['tweettype'])
y = []
for i in range(len(labels)):
    if labels[i] == 'negative':
        y.append(0)
    elif labels[i] == 'neutral':
        y.append(1)
    elif labels[i] == 'positive':
        y.append(2)

y = np.array(y)
labels = tf.keras.utils.to_categorical(y, 3, dtype="float32")
del y

#Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

keras.backend.clear_session()
model_dropout = Sequential()
model_dropout.add(Embedding(input_dim = 128,output_dim = 8,input_length = X.shape[1]))
model_dropout.add(Dropout(rate=0.5))
model_dropout.add(Bidirectional(LSTM(units=256, kernel_initializer= 'normal', return_sequences=True, kernel_constraint=maxnorm(4))))
model_dropout.add(Dropout(rate=0.5))
model_dropout.add(Bidirectional(LSTM(units=128, kernel_initializer= 'normal', return_sequences=False)))
model_dropout.add(Dense(3, activation='softmax'))
optimizer = Adam(lr=0.001)

#Compile and train the model
model_dropout.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['categorical_accuracy'])
history = model_dropout.fit(X_train, Y_train, epochs = 50, batch_size=512, validation_data=(X_test, Y_test))

#Plotting the accuracies
plt.figure(3)
plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.savefig('LSTM-Postprocessed-Accuracy.png')

#Plotting the losses
plt.figure(4)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('cross-entropy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.savefig('LSTM-Postprocessed-Loss.png')