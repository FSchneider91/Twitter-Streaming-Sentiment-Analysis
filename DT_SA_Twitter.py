#######################################################################################################################
#######################################################################################################################
#                                                # NOTES #
#
#                                              Date: 05.11.2019
#                                              Creator: Florian Schneider
#
#                                              Important: To download packages exit the Internal EY Network (e.g.
#                                                         use your Mobile to create a HotSpot
#
#######################################################################################################################
#######################################################################################################################

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

df_tweets = pd.read_csv("DT_Tweets_0516_0219 - SMALL.csv")
df_tweets.tail()
df_tweets.head()

# Combining two datasets
# df_combined = pd.merge(dataset1, dataset2, left_index=True, right_index=True)


df_tweets['Month'] = pd.to_datetime(df_tweets['created_at']).dt.month
df_tweets['Weekday'] = pd.to_datetime(df_tweets['created_at']).dt.weekday
df_tweets['Hour'] = pd.to_datetime(df_tweets['created_at']).dt.hour
df_tweets.head()

# Sentiment Analysis

## Step 1: Identification of the  most favourite tweet
fav_tweets = np.max(df_tweets['favorite_count'])
print(fav_tweets)

fav = df_tweets[df_tweets.favorite_count == fav_tweets].index[0]
#print(fav)
print("The tweet with most likes/favourite counts is: \n{}".format(df_tweets['text'][fav]))

# Identification of the top 3 tweets (according to the number of likes)
fav3_tweets = df_tweets['favorite_count'].nlargest(3)
#print(fav3_tweets)
print('The three most likes tweets of Trump are: \n{}'.format(df_tweets['text'][fav3_tweets]))

## Step 2: Identification of the most retweeted tweet
rt_max = np.max(df_tweets['retweet_count'])
print(rt_max)

ID_rt_max = df_tweets[df_tweets.retweet_count == rt_max].index[0]
print("The tweet with the most retweets is: \n{}".format(df_tweets['text'][ID_rt_max]))

## Step 3: Sentiment Analysis with the Natural Language Toolkit (NLTK)
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

# For analyzing A SINGLE STRING the following code can be used:
nltk.sentiment.util.demo_vader_instance('''RT @cvpayne: While the media is fixated on tweets and fear mongering here's'
                                            'are some employment stats for Black American in July''')
nltk.sentiment.util.demo_vader_instance('It was great being with Luther Strange last night in Alabama. What great '
                                        'people what a crowd! Vote Luther on Tuesday.')

# For analyzing A WHOLE ARRAY the following code can be used:
# Yet first the Dataframe must be transformed into an Array
from nltk.corpus import twitter_samples

#text_array = df_tweets[['text']].as_matrix()
#print(len(text_array))

#strings = str(text_array)
#print(strings)
#print(len(strings))
#nltk.sentiment.util.demo_vader_instance(strings)

# Performing the sentiment analysis for all of the rows in the twitter post dataframe
import nltk
nltk.download('stopwords')
nltk.download('opinion_lexicon')
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
from nltk import sentiment
############################!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
############################!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
############################!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
## !!! The execution of the following line take app. 1.5 hours for the full dataset

# Schleife aufbauen, welche über die tweets geht pro position und es ins entsprechende Feld schreibt+++++++++++++++++++++++++++++++++ SPACY auch testen ANSTELLE VON  NLTK!!! BeautifulSoup (Victoria sendet einen link)

df_tweets['SentimentsAnalyzed'] = (df_tweets['text'].apply(nltk.sentiment.util.demo_liu_hu_lexicon))
df_tweets['SentimentsAnalyzed'].to_csv('DT_Tweets_0516_0219_SA_Done.csv')

SA = pd.read_csv('DT_Tweets_0516_0219_SA_Done.csv')

df_tweets.head()

############################!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
############################!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
############################!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

input_file = 'DT_Tweets_0516_0219 - SMALL.csv'
SA = pd.read_csv(input_file)
SA.head()



# Combining the tweet dataframe with the sentiment dataframe
tweets_SA = pd.merge(df_tweets, SA, left_index=True, right_index=True)
tweets_SA.head()



# Definition of custom colour palettes for charts in future
plt.rcdefaults()
own_palette_1= ["g", "#FFD700", "#FF6347", "#1E90FF"]
sns.set_palette(own_palette_1)
sns.palplot(sns.color_palette())
own_palette_2= [ "#FFD700", "#FF6347", "g", "#1E90FF"]
sns.set_palette(own_palette_2)
sns.palplot(sns.color_palette())
own_palette_3= [ "#FF6347", "#FFD700",  "g", "#1E90FF"]
sns.set_palette(own_palette_3)
sns.palplot(sns.color_palette())
plt.show()


# How many tweets have been neutral, positive oder negative?
tweets_SA.Sentiment.value_counts()


pd.Series(tweets_SA["Sentiment"]).value_counts().plot(kind = "bar", width=0.7,
                        figsize=(16,4),fontsize=15, color='#1E90FF', title = "Sentiments in Donald Trump's tweets: May 2016 - May2018" )
#plt.xlabel('Sentiment', fontsize=10)
plt.ylabel('Nr. of tweets', fontsize=15);
plt.show()


# Plot the percentage of the sentiments into a pie chart
pd.Series(tweets_SA["Sentiment"]).value_counts().plot(kind="pie", colors=sns.color_palette(own_palette_1, 10),
    labels=["Positive", "Neutral",  "Negative", "Positive"],
    shadow=True,autopct='%.1f%%', fontsize=15,figsize=(6, 6))
plt.title("Percentage of tweets of each sentiment", fontsize=20);
plt.show()


# Hour of day vs sentiment - Plotted as Crosstable
hour_of_day =pd.crosstab(tweets_SA.Hour, tweets_SA.Sentiment)
hour_of_day
# Hour of day vs sentiment - Plotted as Bar chart
pd.crosstab(index = tweets_SA["Hour"],columns = tweets_SA["Sentiment"]).plot(kind='bar',fontsize=15, color=sns.color_palette(own_palette_3, 10),
                figsize=(16, 6),alpha=0.5,rot=0,width=0.7, stacked=True,title="Tweets per hour of the day")
plt.title("Tweets per hour of the day", fontsize=20)
plt.xlabel('Hour', fontsize=15)
plt.ylabel('Nr of tweets', fontsize=15)
plt.legend(fontsize=15);
plt.show()

# todo Tokenization
# todo Lemmatization
# todo Map the sentiments to the top 10 european DAX shares (Condition: economically involved in the USA) [Maybe Automotive OEMs, Pharmaceutical Companies]
