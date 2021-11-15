#Import necessary libraries
import pandas as pd
import seaborn as sb

#Read dataset
tweetData = pd.read_csv('data\Original Data\combined_tweettypes.csv', index_col=False)

#Set Index
tweetData.index.name = 'index'
tweetData = tweetData.set_index('index')

# > Similar to Exploratory Data Analysis, further combining categories to have a total of 3 categories in the end.
tweetData.loc[tweetData['tweettype'] == 'anger', 'tweettype'] = 'negative'
tweetData.loc[tweetData['tweettype'] == 'fear', 'tweettype'] = 'negative'
tweetData.loc[tweetData['tweettype'] == 'joy', 'tweettype'] = 'positive'
tweetData.loc[tweetData['tweettype'] == 'sadness', 'tweettype'] = 'negative'
tweetData.loc[tweetData['tweettype'] == 'enthusiasm', 'tweettype'] = 'positive'
tweetData.loc[tweetData['tweettype'] == 'surprise', 'tweettype'] = 'positive'
tweetData.loc[tweetData['tweettype'] == 'love', 'tweettype'] = 'positive'
tweetData.loc[tweetData['tweettype'] == 'relief', 'tweettype'] = 'positive'

#View the value_counts of each of the three categories
print(tweetData["tweettype"].value_counts())
sb.catplot(y = "tweettype", data = tweetData, kind = "count")

#Export to csv
tweetData.to_csv('data\Postprocessed-Output.csv')