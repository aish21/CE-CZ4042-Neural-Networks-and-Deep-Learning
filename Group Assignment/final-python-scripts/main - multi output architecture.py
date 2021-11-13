import nltk
import pandas as pd
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Embedding, LSTM, MaxPooling1D, Dropout, Bidirectional, ConvLSTM2D, Flatten, Conv1D, Attention, Input
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras import Model

import re
import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras import backend as K
from keras.layers import Layer
from nltk.tokenize import word_tokenize
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score

'''
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
'''

tweetData = pd.read_csv('featureEngineeredFinal.csv', index_col=False)

'''
enc = OneHotEncoder(handle_unknown='ignore')
labels = np.array(tweetData['tweettype'])
labels = enc.fit_transform(labels.reshape(-1, 1))

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
'''

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

keras.backend.clear_session()
embed_dim = 8

inputs=Input(shape=(28,), dtype='int32', name='main_input')
x=Embedding(27608,embed_dim,input_length = X.shape[1],\
            embeddings_regularizer=keras.regularizers.l2(.001))(inputs)
x = Dropout(0.3)(x)
x = Conv1D(64, 5, activation='relu')(x)
x = MaxPooling1D(pool_size=4)(x)
x = LSTM(100)(x)
x = Dropout(0.3)(x)

output_columns_binary = ['sadness', 'neutral', 'joy', 'love', 'enthusiasm', 'anger', 'surprise', 'relief', 'fear']
for label in output_columns_binary:
    tweetData[label] = 0
for i, typeTweet in enumerate(tweetData['tweettype']):
    tweetData.loc[i, typeTweet] = 1

tweetData.head()
outputs = tweetData[output_columns_binary]

X_train, X_test, Y_train, Y_test = train_test_split(X, outputs, test_size=0.3, random_state=42)

output_array = []
metrics_array = {}
loss_array = {}
for i, dense_layer in enumerate(output_columns_binary):
    name = f'binary_output_{i}'
    binary_output = Dense(1, activation='sigmoid', name=name)(x)
    output_array.append(binary_output)
    metrics_array[name] = 'binary_accuracy'
    loss_array[name] = 'binary_crossentropy'

model = Model(inputs=inputs, outputs=output_array)

model.compile(optimizer='adadelta',
              loss=loss_array,
              metrics = metrics_array)
model.summary()

weight_binary = {0: 0.5, 1: 7}

classes_weights = {}
for i, dense_layer in enumerate(output_columns_binary):
    name = f'binary_output_{i}'
    classes_weights[name] = weight_binary

y_train_output = []
for col in output_columns_binary:
    y_train_output.append(Y_train[col])

y_test_output = []
for col in output_columns_binary:
    y_test_output.append(Y_test[col])

history = model.fit(X_train, y_train_output, epochs = 50, batch_size=512)

y_pred = model.predict(X_test)
THRESHOLD = 0.5 # threshold between classes
f1_score_results = []
# Binary Outputs
for col_idx, col in enumerate(output_columns_binary):
    print(f'{col} accuracy \n')
    
    # Transform array of probabilities to class: 0 or 1
    y_pred[col_idx][y_pred[col_idx]>=THRESHOLD] = 1
    y_pred[col_idx][y_pred[col_idx]<THRESHOLD] = 0
    f1_score_results.append(f1_score(Y_test[col], y_pred[col_idx], average='macro'))
    print(classification_report(Y_test[col], y_pred[col_idx]))

print('Total :',np.sum(f1_score_results))


'''
def build_lstm():
    embed_dim = 8
    keras.backend.clear_session()
    model_dropout = Sequential()
    model_dropout.add(Embedding(input_dim=128, output_dim=embed_dim, input_length=X.shape[1]))
    model_dropout.add(Dropout(rate=0.4))
    model_dropout.add(Bidirectional(LSTM(units=256, return_sequences=True)))
    model_dropout.add(Dropout(rate=0.4))
    model_dropout.add(Bidirectional(LSTM(units=128, return_sequences=False)))
    model_dropout.add(Dense(9, activation='softmax'))
    model_dropout.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model_dropout

    model=KerasClassifier(build_fn=build_lstm, verbose = -1)
batch_size = [512, 256, 128, 64]
epochs = [25, 50, 100, 150, 200]
param_grid = dict(batch_size=batch_size, epochs=epochs)
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid_result = grid.fit(X_train, Y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

'''

