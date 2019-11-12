#######################################################################################################################
#######################################################################################################################
#                                                # NOTES #
#
#                                              Date: 06.11.2019
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
import re
#import nltk
from nltk.corpus import stopwords
#nltk.download('stopwords') # Already Downloaded - this download must be performed outside of the EY network

################################################################################################################
# Creation of a custom Sentiment Analysis algorithm (following: https://python.gotrained.com/tf-idf-twitter-sentiment-
# analysis/#Installing_Required_Libraries)
# Plan: Training a classification algorithm on a dataset with existing sentiments (
################################################################################################################
TRAINING_tweets = pd.read_csv(
    "https://raw.githubusercontent.com/kolaveridi/kaggle-Twitter-US-Airline-Sentiment-/master/Tweets.csv")
TRAINING_tweets.head()
################################################################################################################


################################################################################################################
# 1. Data Preprocessing
################################################################################################################
X = TRAINING_tweets.iloc[:, 10].values
y = TRAINING_tweets.iloc[:, 1].values
################################################################################################################


################################################################################################################
#2. Cleaning the Dataset
################################################################################################################
# Remove all the double-spaces, special characters, single start characters & transform to lowercase
processed_tweets = []
for tweet in range(0, len(X)):
    # Remove all the special characters
    processed_tweet = re.sub(r'\W', ' ', str(X[tweet]))
    # remove all single characters
    processed_tweet = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_tweet)
    # Remove single characters from the start
    processed_tweet = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_tweet)
    # Substituting multiple spaces with single space
    processed_tweet = re.sub(r'\s+', ' ', processed_tweet, flags=re.I)
    # Removing prefixed 'b'
    processed_tweet = re.sub(r'^b\s+', '', processed_tweet)
    # Converting to Lowercase
    processed_tweet = processed_tweet.lower()
    processed_tweets.append(processed_tweet)

# Transformation to TF-IDF Scheme
from sklearn.feature_extraction.text import TfidfVectorizer
tfidfconverter = TfidfVectorizer(max_features=2000, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
X = tfidfconverter.fit_transform(processed_tweets).toarray()
################################################################################################################
# Splitting the Data into a training and a test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)



#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#COMPARISON OF DIFFERENZ ALGORITHMS WITH AN CROSS VALIDATION SCORE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sklearn.model_selection import cross_val_score
# Define the hyperparameters in a model
models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state=0),
]
CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []
for model in models:
    model_name = model.__class__.__name__
    accuracies = cross_val_score(model, X, y, scoring='accuracy', cv=CV)                #!! Features not defined
    for fold_idx, accuracy in enumerate(accuracies):
        entries.append((model_name, fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

# Seaborn is used for data visualization in statistics (BoxPlots, Scatterplots, etc). It is based on Matplotlib
import seaborn as sns
sns.boxplot(x='model_name', y='accuracy', data=cv_df)
sns.stripplot(x='model_name', y='accuracy', data=cv_df,
              size=8, jitter=True, edgecolor="gray", linewidth=2)
plt.show()

print('The means of the different algorithms are:\n{}'.format(cv_df.groupby('model_name').accuracy.mean()))
#cv_df.groupby('model_name').accuracy.mean()
print('The maximum accuracies of the different algorithms are:\n{}'.format(cv_df.groupby('model_name').accuracy.max()))


# Training and evaluating the text classification model a the Random Forest
from sklearn.ensemble import RandomForestClassifier                                                                         # This is a first try with a Random Forest Classifier
text_classifier = RandomForestClassifier(n_estimators=100, random_state=0)
text_classifier.fit(X_train, y_train)
################################################################################################################
# Predicting (with the test data)
################################################################################################################
predictions_RFC = text_classifier.predict(X_test)
################################################################################################################


################################################################################################################
# Evaluation of the results (Random Forset Classifier)
################################################################################################################
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(y_test, predictions_RFC))
print(classification_report(y_test, predictions_RFC))
print(accuracy_score(y_test, predictions_RFC))
################################################################################################################


# According to the Crossvalidation Score the Average & Maximum of Linear SVC gains the best results -->
# lets test the prediction quality with an Linear SVM
# Training and evaluating the text classification model
from sklearn.svm import LinearSVC
text_classifier = LinearSVC()
text_classifier.fit(X_train, y_train)
################################################################################################################
# Predicting (with the test data) (Linear Support Vector Machine)
################################################################################################################
predictions = text_classifier.predict(X_test)
################################################################################################################


################################################################################################################
# Evaluation of the results
################################################################################################################
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
print(accuracy_score(y_test, predictions))
################################################################################################################


################################################################################################################
################################## Using the trained classifier on the TRUMP Tweets ############################
################################################################################################################
# After the training of the classification algorithm it can be applied on the preprocessed Trump Tweets
# The output of the classification is going to be saved in an additional row called "Sentiment Analysed" and they will
# be copy/pasted into a new file called "DT_Tweets_0516_0219_SA_Done"

df_DT_tweets = pd.read_csv("DT_Tweets_0516_0219 - SMALL.csv")
print(len(df_DT_tweets))

X2 = df_DT_tweets.iloc[:, 0].values
# todo delete these next two lines?
# Combining two datasets
# df_combined = pd.merge(dataset1, dataset2, left_index=True, right_index=True)

################################################################################################################
# Preprocessing of the Trump Tweets
################################################################################################################
for DT_tweet in range(0, len(X2)):
    processed_DT_tweets = []

    for tweet in range(0, len(X2)):
        # Remove all the special characters
        processed_DT_tweet = re.sub(r'\W', ' ', str(X2[tweet]))
        # remove all single characters
        processed_DT_tweet = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_DT_tweet)
        # Remove single characters from the start
        processed_DT_tweet = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_DT_tweet)
        # Substituting multiple spaces with single space
        processed_DT_tweet = re.sub(r'\s+', ' ', processed_DT_tweet, flags=re.I)
        # Removing prefixed 'b' (This is required when the string is provided in a byte format)
        processed_DT_tweet = re.sub(r'^b\s+', '', processed_DT_tweet)
        # Converting to Lowercase
        processed_DT_tweet = processed_DT_tweet.lower()
        processed_DT_tweets.append(processed_DT_tweet)


################################################################################################################
# Vectorize the Trump Tweets using the tdifconverter that we fitted the Airline review data to.
################################################################################################################
vectorized_DT_tweets = tfidfconverter.transform(processed_DT_tweets).toarray()
################################################################################################################


################################################################################################################
# Predict on the transformed (vectorized) DT_Tweets
################################################################################################################
Sentiments = text_classifier.predict(vectorized_DT_tweets)
################################################################################################################


################################################################################################################
# Save the results to a new row in the original DT_Tweets dataframe
################################################################################################################
df_Sentiments = pd.DataFrame(Sentiments)
df_DT_tweets.head()
df_DT_tweets['Sentiments'] = pd.DataFrame(Sentiments)
df_DT_tweets.head()
df_DT_tweets.to_csv('DT_Tweets_0516_0219 - SMALL_DONE.csv')
################################################################################################################
################################################################################################################



################################################################################################################
#################################### Data Exploration & Vizusalization##########################################
################################################################################################################
# General Data Exploration
df_DT_tweets['Month'] = pd.to_datetime(df_DT_tweets['created_at']).dt.month
df_DT_tweets['Weekday'] = pd.to_datetime(df_DT_tweets['created_at']).dt.weekday
df_DT_tweets['Hour'] = pd.to_datetime(df_DT_tweets['created_at']).dt.hour
df_DT_tweets.head()


################################################################################################################
## Step 1: Identification of the  most favourite tweet
################################################################################################################
fav_tweets = np.max(df_DT_tweets['favorite_count'])
print(fav_tweets)
fav = df_DT_tweets[df_DT_tweets.favorite_count == fav_tweets].index[0]
print("The tweet with most likes/favourite counts is: \n{}".format(df_DT_tweets['text'][fav]))
################################################################################################################


################################################################################################################
# Identification of the top 3 tweets (according to the number of likes)
################################################################################################################
fav3_tweets = df_DT_tweets['favorite_count'].nlargest(3)
#print(fav3_tweets)
print('The three most likes tweets of Trump are: \n{}'.format(df_DT_tweets['text'][fav3_tweets]))
################################################################################################################


################################################################################################################
## Step 2: Identification of the most retweeted tweet
################################################################################################################
rt_max = np.max(df_DT_tweets['retweet_count'])
print(rt_max)
################################################################################################################
ID_rt_max = df_DT_tweets[df_DT_tweets.retweet_count == rt_max].index[0]
print("The tweet with the most retweets is: \n{}".format(df_DT_tweets['text'][ID_rt_max]))
################################################################################################################
################################################################################################################


################################################################################################################
# Definition of custom colour palettes for charts in future
################################################################################################################
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
################################################################################################################


################################################################################################################
# How many tweets have been neutral, positive oder negative?
################################################################################################################
df_DT_tweets.Sentiments.value_counts()


pd.Series(df_DT_tweets["Sentiments"]).value_counts().plot(kind = "bar", width=0.7,
                        figsize=(16,4),fontsize=15, color='#1E90FF', title = "Sentiments in Donald Trump's tweets: DATESCOPE" )
#plt.xlabel('Sentiment', fontsize=10)
plt.ylabel('Nr. of tweets', fontsize=15);
plt.show()
################################################################################################################


################################################################################################################
# Plot the percentage of the sentiments into a pie chart
################################################################################################################
pd.Series(df_DT_tweets["Sentiments"]).value_counts().plot(kind="pie", colors=sns.color_palette(own_palette_1, 10),
    labels=["Positive", "Neutral",  "Negative", "Positive"],
    shadow=True,autopct='%.1f%%', fontsize=15,figsize=(6, 6))
plt.title("Percentage of tweets of each sentiment", fontsize=20);
plt.show()
################################################################################################################


################################################################################################################
# Hour of day vs sentiment - Plotted as Crosstable
################################################################################################################
hour_of_day =pd.crosstab(df_DT_tweets.Hour, df_DT_tweets.Sentiments)
# Hour of day vs sentiment - Plotted as Bar chart
pd.crosstab(index = df_DT_tweets["Hour"], columns = df_DT_tweets["Sentiments"]).plot(kind='bar', fontsize=15, color=sns.color_palette(own_palette_3, 10),
                figsize=(16, 6), alpha=0.5, rot=0, width=0.7, stacked=True, title="Tweets per hour of the day")
plt.title("Tweets per hour of the day", fontsize=20)
plt.xlabel('Hour', fontsize=15)
plt.ylabel('Nr of tweets', fontsize=15)
plt.legend(fontsize=15)
plt.show()
################################################################################################################



# todo Tokenization
# todo Lemmatization
# todo Map the sentiments to the top 10 european DAX shares (Condition: economically involved in the USA) [Maybe Automotive OEMs, Pharmaceutical Companies]


################################################################################################################
# Merging DataFrames (JOINING)
################################################################################################################
#df_DT_tweets = pd.merge(df_DT_tweets, SA, left_index=True, right_index=True)
#df_DT_tweets.head()
################################################################################################################