import nltk
import pandas as pd
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Dropout, Bidirectional, ConvLSTM2D, Flatten, Conv1D, Attention, Input
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import re
import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras import backend as K
from keras.layers import Layer
from nltk.tokenize import word_tokenize
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.optimizers import Adam 
from keras.constraints import maxnorm

tweetData = pd.read_csv('Postprocessed-Feature-Engineered.csv', index_col=False)

labels = np.array(tweetData['tweettype'])
y = []
for i in range(len(labels)):
    if labels[i] == 'positive':
        y.append(0)
    elif labels[i] == 'negative':
        y.append(1)
    elif labels[i] == 'neutral':
        y.append(2)
y = np.array(y)
labels = tf.keras.utils.to_categorical(y, 3, dtype="float32")
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

tokenizer = Tokenizer(num_words=27608, split=' ')
tokenizer.fit_on_texts(tweetData['lemmatizedText'].values)
X = tokenizer.texts_to_sequences(tweetData['lemmatizedText'].values)
X = pad_sequences(X)

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
model_dropout.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['categorical_accuracy'])

history = model_dropout.fit(X_train, Y_train, epochs = 50, batch_size=512, validation_data=(X_test, Y_test))

# plotting the accuracies for the training epochs
plt.figure(1)
plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.savefig('LSTM-Postprocessed-Accuracy.png')

# plotting the losses for the training epochs
plt.figure(2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('cross-entropy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.savefig('LSTM-Postprocessed-Loss.png')