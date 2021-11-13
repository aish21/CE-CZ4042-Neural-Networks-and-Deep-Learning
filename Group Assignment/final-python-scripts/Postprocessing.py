#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sb
import matplotlib as plt
import numpy as np
import sklearn
import matplotlib.pyplot as mtplt


# In[ ]:


from nltk.corpus import stopwords
from collections import  Counter
from sklearn.feature_extraction.text import CountVectorizer
import gensim
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.tokenize import word_tokenize
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from wordcloud import WordCloud, STOPWORDS
pyLDAvis.enable_notebook()


# In[4]:


tweetData = pd.read_csv('combined_tweettypes.csv', index_col=False)
tweetData


# In[6]:


tweetData.index.name = 'index'
tweetData = tweetData.set_index('index')


# > Similar to Exploratory Data Analysis, further combining categories to have a total of 3 categories in the end.

# In[7]:


tweetData.loc[tweetData['tweettype'] == 'anger', 'tweettype'] = 'negative'
tweetData.loc[tweetData['tweettype'] == 'fear', 'tweettype'] = 'negative'
tweetData.loc[tweetData['tweettype'] == 'joy', 'tweettype'] = 'positive'
tweetData.loc[tweetData['tweettype'] == 'sadness', 'tweettype'] = 'negative'
tweetData.loc[tweetData['tweettype'] == 'enthusiasm', 'tweettype'] = 'positive'
tweetData.loc[tweetData['tweettype'] == 'surprise', 'tweettype'] = 'positive'
tweetData.loc[tweetData['tweettype'] == 'love', 'tweettype'] = 'positive'
tweetData.loc[tweetData['tweettype'] == 'relief', 'tweettype'] = 'positive'


# In[8]:


tweetData['tweettype'].value_counts()


# In[9]:


print(tweetData["tweettype"].value_counts())
sb.catplot(y = "tweettype", data = tweetData, kind = "count")


# In[11]:


tweetData.to_csv('Postporcessed-Output.csv')


# In[ ]:




