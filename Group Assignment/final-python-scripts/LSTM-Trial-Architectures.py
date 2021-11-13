#!/usr/bin/env python
# coding: utf-8

# In[91]:


'''
This file contains the trial for the model run - Attention, Covolutional, Bidirectional, Forward-Backward LSTM
and permutations of the above - without Hyperparameter Tuning.
For best results, run the corresponding Python file.
The outputs have been presented and discussed in the report.
'''
import nltk
import pandas as pd
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import wordcloud
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Dropout, Bidirectional, ConvLSTM2D, Flatten, Conv1D, Attention, Input
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt
import re
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras import backend as K


# In[29]:


tweetData = pd.read_csv('featureEngineeredFinal.csv', index_col=False)
tweetData


# In[31]:


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

tokenizer = Tokenizer(num_words=27608, split=' ')
tokenizer.fit_on_texts(tweetData['lemmatizedText'].values)
X = tokenizer.texts_to_sequences(tweetData['lemmatizedText'].values)
X = pad_sequences(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, labels, test_size=0.3, random_state=42)


# ### Initial Bidirectional

# In[18]:


embed_dim = 64
keras.backend.clear_session()
model_dropout = Sequential()
model_dropout.add(Embedding(27608,embed_dim,input_length = X.shape[1]))
model_dropout.add(Dropout(rate=0.4))
model_dropout.add(Bidirectional(LSTM(units=128, return_sequences=True)))
model_dropout.add(Dropout(rate=0.4))
model_dropout.add(Bidirectional(LSTM(units=128, return_sequences=False)))
model_dropout.add(Dense(1, activation='softmax'))

model_dropout.summary()


# In[20]:


model_dropout.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[21]:


history = model_dropout.fit(X_train, Y_train, epochs = 20, batch_size=64, validation_data=(X_test, Y_test))


# In[23]:


# plotting the accuracies for the training epochs
plt.figure(1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.savefig('1b_accuracies.png')


# In[24]:


# plotting the losses for the training epochs
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('cross-entropy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.savefig('1c_losses.png')


# ### Single Bidirectional

# In[36]:


embed_dim = 64
keras.backend.clear_session()
model_dropout = Sequential()
model_dropout.add(Embedding(2000,embed_dim,input_length = X.shape[1]))
model_dropout.add(Dropout(rate=0.4))
model_dropout.add(Bidirectional(LSTM(units=128, return_sequences=False)))
model_dropout.add(Dense(9, activation='softmax'))
model_dropout.summary()


# In[37]:


model_dropout.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[38]:


history = model_dropout.fit(X_train, Y_train, epochs = 20, batch_size=64, validation_data=(X_test, Y_test))


# In[39]:


# plotting the accuracies for the training epochs
plt.figure(1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.savefig('2b_accuracies.png')


# In[40]:


# plotting the losses for the training epochs
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('cross-entropy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.savefig('2c_losses.png')


# ### Convolutional 1D

# In[60]:


keras.backend.clear_session()

model_dropout = Sequential()
model_dropout.add(Embedding(128,embed_dim,input_length = X.shape[1]))
model_dropout.add(Conv1D(filters=32,
               kernel_size=8,
               strides=1,
               activation='relu',
               padding='same'))
model_dropout.add(LSTM(units=128, return_sequences=False))
model_dropout.add(Dense(9, activation='softmax'))

model_dropout.summary()


# In[61]:


model_dropout.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[62]:


history = model_dropout.fit(X_train, Y_train, epochs = 20, batch_size=64, validation_data=(X_test, Y_test))


# In[63]:


# plotting the accuracies for the training epochs
plt.figure(1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.savefig('3b_accuracies.png')


# In[64]:


# plotting the losses for the training epochs
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('cross-entropy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.savefig('3c_losses.png')


# ### Convolutional 1D + Dropouts

# In[65]:


keras.backend.clear_session()

model_dropout = Sequential()
model_dropout.add(Embedding(128,embed_dim,input_length = X.shape[1]))
model_dropout.add(Conv1D(filters=32,
               kernel_size=8,
               strides=1,
               activation='relu',
               padding='same'))
model_dropout.add(Dropout(rate=0.4))
model_dropout.add(LSTM(units=128, return_sequences=False))
model_dropout.add(Dropout(rate=0.4))
model_dropout.add(Dense(9, activation='softmax'))

model_dropout.summary()


# In[66]:


model_dropout.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[67]:


history = model_dropout.fit(X_train, Y_train, epochs = 20, batch_size=64, validation_data=(X_test, Y_test))


# In[68]:


# plotting the accuracies for the training epochs
plt.figure(1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.savefig('4b_accuracies.png')


# In[69]:


# plotting the losses for the training epochs
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('cross-entropy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.savefig('4c_losses.png')


# ### Convolutional 1D + LSTM

# In[76]:


keras.backend.clear_session()

model_dropout = Sequential()
model_dropout.add(Embedding(128,embed_dim,input_length = X.shape[1]))
model_dropout.add(Conv1D(filters=32,
               kernel_size=8,
               strides=1,
               activation='relu',
               padding='same', kernel_regularizer=keras.regularizers.l1()))
model_dropout.add(LSTM(units=128, return_sequences=False))
model_dropout.add(Dense(9, activation='softmax'))

model_dropout.summary()


# In[77]:


model_dropout.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[78]:


history = model_dropout.fit(X_train, Y_train, epochs = 20, batch_size=64, validation_data=(X_test, Y_test))


# In[79]:


# plotting the accuracies for the training epochs
plt.figure(1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.savefig('5b_accuracies.png')


# In[80]:


# plotting the losses for the training epochs
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('cross-entropy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.savefig('5c_losses.png')


# ### Attention + Bidirectional + LSTM permutations

# In[89]:


from keras.layers import Layer
class attention(Layer):
    def __init__(self,**kwargs):
        super(attention,self).__init__(**kwargs)

    def build(self,input_shape):
        self.W=self.add_weight(name="att_weight",shape=(input_shape[-1],1),initializer="normal")
        self.b=self.add_weight(name="att_bias",shape=(input_shape[1],1),initializer="zeros")        
        super(attention, self).build(input_shape)

    def call(self,x):
        et=K.squeeze(K.tanh(K.dot(x,self.W)+self.b),axis=-1)
        at=K.softmax(et)
        at=K.expand_dims(at,axis=-1)
        output=x*at
        return K.sum(output,axis=1)

    def compute_output_shape(self,input_shape):
        return (input_shape[0],input_shape[-1])

    def get_config(self):
        return super(attention,self).get_config()


# In[100]:


keras.backend.clear_session()
embed_dim = 8

inputs=Input((28,))
x=Embedding(128,embed_dim,input_length = X.shape[1],            embeddings_regularizer=keras.regularizers.l2(.001))(inputs)
att_in=LSTM(128,return_sequences=True,dropout=0.3,recurrent_dropout=0.2)(x)
att_out=attention()(att_in)
outputs=Dense(9,activation='softmax',trainable=True)(att_out)
model=keras.Model(inputs,outputs)
model.summary()


# In[101]:


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[102]:


history = model.fit(X_train, Y_train, epochs = 20, batch_size=64, validation_data=(X_test, Y_test))


# In[103]:


# plotting the accuracies for the training epochs
plt.figure(1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.savefig('6b_accuracies.png')


# In[104]:


# plotting the losses for the training epochs
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('cross-entropy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.savefig('6c_losses.png')


# In[ ]:


class attention(Layer):
    def __init__(self,**kwargs):
        super(attention,self).__init__(**kwargs)

    def build(self,input_shape):
        self.W=self.add_weight(name="att_weight",shape=(input_shape[-1],1),initializer="normal")
        self.b=self.add_weight(name="att_bias",shape=(input_shape[1],1),initializer="zeros")
        super(attention, self).build(input_shape)

    def call(self,x):
        et=K.squeeze(K.tanh(K.dot(x,self.W)+self.b),axis=-1)
        at=K.softmax(et)
        at=K.expand_dims(at,axis=-1)
        output=x*at
        return K.sum(output,axis=1)

    def compute_output_shape(self,input_shape):
        return (input_shape[0],input_shape[-1])

    def get_config(self):
        return super(attention,self).get_config()

keras.backend.clear_session()
embed_dim = 8
inputs=Input((28,))
x=Embedding(128,embed_dim,input_length = X.shape[1],            embeddings_regularizer=keras.regularizers.l2(.001))(inputs)
bidirectional_in = Bidirectional(LSTM(128,return_sequences=True,dropout=0.3,recurrent_dropout=0.2))(x)
att_in=Bidirectional(LSTM(128,return_sequences=True,dropout=0.3,recurrent_dropout=0.2))(bidirectional_in)
att_out=attention()(att_in)
outputs=Dense(9,activation='softmax',trainable=True)(att_out)
model=keras.Model(inputs,outputs)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

