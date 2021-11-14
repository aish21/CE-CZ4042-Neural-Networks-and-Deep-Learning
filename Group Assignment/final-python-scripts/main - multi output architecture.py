import nltk
import pandas as pd
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Embedding, LSTM, MaxPooling1D, Dropout, Conv1D, Input
from sklearn.model_selection import train_test_split
from keras import Model

import re
from tensorflow import keras
import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.metrics import classification_report, accuracy_score, f1_score

tweetData = pd.read_csv('featureEngineeredFinal.csv', index_col=False)

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