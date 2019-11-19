#######################################################################################################################
#######################################################################################################################
#                                                # NOTES #
#
#                                              Date: 07.11.2019
#                                              Creator: Florian Schneider
#
#                                              Important: To download packages exit the Internal EY Network (e.g.
#                                                         use your Mobile to create a HotSpot
#
#
#                                              Orignial Source: https://python.gotrained.com/scraping-tweets-sentiment-
#                                              analysis/#Scraping_Tweets_from_Twitter
#######################################################################################################################
#######################################################################################################################

import pandas as pd
from pandas import DataFrame
import tweepy
import re
from tweepy import OAuthHandler


consumer_api_key = 'consumer_api_key'
consumer_api_secret = 'consumer_api_secret'
access_token = 'access_token'
access_token_secret = 'access_token_secret'


authorizer = OAuthHandler(consumer_api_key, consumer_api_secret)
authorizer.set_access_token(access_token, access_token_secret)

## Testing the authentication
api = tweepy.API(authorizer)
try:
    api.verify_credentials()
    print("Authentication OK")
except:
    print("Error during authentication")

# Scraping Tweets
api = tweepy.API(authorizer, timeout=15)
all_tweets = []
search_query = 'Trump'
for tweet_object in tweepy.Cursor(api.search, q=search_query + " -filter:retweets", lang='en',
                                  result_type='recent').items(200):
    all_tweets.append(tweet_object.text)


df_tweets_scraped = DataFrame(all_tweets)
df_tweets_scraped .to_csv('tweets_scraped.csv')


# Create a Stream Listener with Tweepy

#override tweepy.StreamListener to add logic to on_status
class MyStreamListener(tweepy.StreamListener):
    def on_status(self, status):
        print(status.text)
