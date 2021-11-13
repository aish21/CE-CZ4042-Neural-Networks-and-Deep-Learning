#!/usr/bin/env python
# coding: utf-8

# ### Necessary Imports

# In[ ]:


import pandas as pd
import seaborn as sb
import matplotlib as plt
import numpy as np
import sklearn
import matplotlib.pyplot as mtplt
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


# ### Cleaned Data Retrieval

# In[2]:


tweetData = pd.read_csv('text_emotion_recognition_updated.csv', index_col=False)
tweetData


# > We begin the analysis by understanding the rough estimation of the tweet length, by finding out the number of characters present in each sentence of a tweet.

# In[4]:


tweetData['tweet'].str.len().hist()


# > To get a rough estimation of dimensions needed for the neural network architecture - by counting the number of words in each tweet

# In[5]:


tweetData['tweet'].str.split().    map(lambda x: len(x)).    hist()


# > Now, we check the average word length in each sentence

# In[6]:


tweetData['tweet'].str.split().   apply(lambda x : [len(i) for i in x]).    map(lambda x: np.mean(x)).hist()


# > From the above analysis, one may assume that people are generally using really short words in tweets, and, although this conjecture may ne true in some cases, it is not a correct conclusion/explaination to explaing the plots obtained. One reason for this is the presence of stopwords, words that are most commonly used in any language (a, an, the, etc.). This can explain why the above plot could be left skewed by short words.

# In[7]:


import nltk
stop=set(stopwords.words('english'))


# In[8]:


# Plot the occurances of the most common stopwords in the dataset
corpus=[]
check= tweetData['tweet'].str.split()
check=check.values.tolist()
corpus=[word for i in check for word in i]

from collections import defaultdict
dic=defaultdict(int)
for word in corpus:
    if word in stop:
        dic[word]+=1
        
top=sorted(dic.items(), key=lambda x:x[1],reverse=True)[:10] 
x,y=zip(*top)
mtplt.bar(x,y)
mtplt.show()


# > Now we know which stopwords occur frequently in the dataset. The next step is to inspect which words other than these stopwords occur just as frequently, if not more, or less. This is achieved using the collections library function. 

# In[9]:


counter=Counter(corpus)
most=counter.most_common()

x, y= [], []
for word,count in most[:60]:
    if (word not in stop):
        x.append(word)
        y.append(count)
        
sb.barplot(x=y,y=x)


# > Since these are tweets (hence, unfortunatelly, the poor grammar) it will include quite a number stopwords as well. Therefore, next, we will use N-gram analysis - N-grams are simply contiguous sequences of n words. Looking at most frequent n-grams can give a better understanding of the context in which the word was used. To build a representation of the vocabulary Countvectorizer has been used. Countvectorizer is a simple method used to tokenize, vectorize and represent the corpus in an appropriate form. A function that combines everything above hsa be depicted below - 

# In[10]:


def plot_top_ngrams_barchart(text, n=2):
    stop=set(stopwords.words('english'))

    new= text.str.split()
    new=new.values.tolist()
    corpus=[word for i in new for word in i]

    def _get_top_ngram(corpus, n=None):
        vec = CountVectorizer(ngram_range=(n, n)).fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) 
                      for word, idx in vec.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
        return words_freq[:10]

    top_n_bigrams=_get_top_ngram(text,n)[:10]
    x,y=map(list,zip(*top_n_bigrams))
    sb.barplot(x=y,y=x)


# In[11]:


plot_top_ngrams_barchart(tweetData['tweet'],2)


# In[12]:


plot_top_ngrams_barchart(tweetData['tweet'],3)


# In[13]:


plot_top_ngrams_barchart(tweetData['tweet'],4)


# In[14]:


plot_top_ngrams_barchart(tweetData['tweet'],5)


# > Topic modeling is the process of using unsupervised learning techniques to extract the main topics that occur in a collection of documents. Latent Dirichlet Allocation (LDA) is an easy to use and efficient model for topic modeling. Each document is represented by the distribution of topics and each topic is represented by the distribution of words. This has been used in practice below to better analyse the dataset.

# In[15]:


# The output of the following cells are interactable - to give a better idea of the dataset
def get_lda_objects(text):
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    stop=set(stopwords.words('english'))

    
    def _preprocess_text(text):
        corpus=[]
        stem=PorterStemmer()
        lem=WordNetLemmatizer()
        for news in text:
            words=[w for w in word_tokenize(news) if (w not in stop)]

            words=[lem.lemmatize(w) for w in words if len(w)>2]

            corpus.append(words)
        return corpus
    
    corpus=_preprocess_text(text)
    
    dic=gensim.corpora.Dictionary(corpus)
    bow_corpus = [dic.doc2bow(doc) for doc in corpus]
    
    lda_model =  gensim.models.LdaMulticore(bow_corpus, 
                                   num_topics = 10, 
                                   id2word = dic,                                    
                                   passes = 10,
                                   workers = 2)
    
    return lda_model, bow_corpus, dic

def plot_lda_vis(lda_model, bow_corpus, dic):
    vis = gensimvis.prepare(lda_model, bow_corpus, dic)
    return vis


# In[16]:


lda_model, bow_corpus, dic = get_lda_objects(tweetData['tweet'])


# In[17]:


lda_model.show_topics()


# In[18]:


# Can adjust the metric value, as well as the topics
plot_lda_vis(lda_model, bow_corpus, dic)


# > Wordcloud is a great way to represent text data. The size and color of each word that appears in the wordcloud indicate it’s frequency or importance.

# In[19]:


def plot_wordcloud(text):
    nltk.download('stopwords')
    stop=set(stopwords.words('english'))

    def _preprocess_text(text):
        corpus=[]
        stem=PorterStemmer()
        lem=WordNetLemmatizer()
        for news in text:
            words=[w for w in word_tokenize(news) if (w not in stop)]

            words=[lem.lemmatize(w) for w in words if len(w)>2]

            corpus.append(words)
        return corpus
    
    corpus=_preprocess_text(text)
    
    wordcloud = WordCloud(
        background_color='white',
        stopwords=set(STOPWORDS),
        max_words=100,
        max_font_size=30, 
        scale=3,
        random_state=1)
    
    wordcloud=wordcloud.generate(str(corpus))

    fig = mtplt.figure(1, figsize=(12, 12))
    mtplt.axis('off')
 
    mtplt.imshow(wordcloud)
    mtplt.show()


# In[20]:


plot_wordcloud(tweetData['tweet'])


# In[21]:


# Analysing the Target Variable - count plot
print(tweetData["tweettype"].value_counts())
sb.catplot(y = "tweettype", data = tweetData, kind = "count")


# In[22]:


tweetData.set_index('tweet_id')


# In[24]:


tweetData.index.name = 'index'


# In[25]:


tweetData = tweetData.set_index('tweet_id')


# In[26]:


# Combining some categories due to lack of data points and improve distribution of target variable
tweetData.loc[tweetData['tweettype'] == 'hate', 'tweettype'] = 'anger'
tweetData.loc[tweetData['tweettype'] == 'happiness', 'tweettype'] = 'joy'
tweetData.loc[tweetData['tweettype'] == 'fun', 'tweettype'] = 'enthusiasm'
tweetData.loc[tweetData['tweettype'] == 'worry', 'tweettype'] = 'sadness'
tweetData.loc[tweetData['tweettype'] == 'empty', 'tweettype'] = 'neutral'


# In[27]:


tweetData['tweettype'].value_counts()


# In[29]:


tweetData.loc[tweetData['tweettype'] == 'boredom', 'tweettype'] = 'neutral'


# In[30]:


print(tweetData["tweettype"].value_counts())
sb.catplot(y = "tweettype", data = tweetData, kind = "count")


# In[ ]:


# Save the modified target variables
tweetData.to_csv('Exploratory-Data-Analysis-Output.csv')

