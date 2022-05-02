#!/usr/bin/env python3

import sys

from twython import Twython, TwythonError #https://github.com/ryanmcgrath/twython

from twitter_auth import *
import tweets_json

def get_user_tweets(twitter, screen_name, **kwargs):
    """uses twitter (Twython object) to collect tweets from user with screen_name (no @), can include extra search parameters for get_user_timeline, returns list of tweets"""

    #initialize a list to hold all the tweets
    user_tweets = []
    try:
        #make initial request for most recent tweets (200 is the maximum allowed count).
        #We normally don't want retweets, so we set include_rts to false.
        #tweet_mode="extended" allows for full text tweets, rather than truncated (i.e. over 140 chars)
        #https://developer.twitter.com/en/docs/tweets/timelines/api-reference/get-statuses-user_timeline.html
        new_tweets = twitter.get_user_timeline(screen_name=screen_name,count=200,include_rts=False,tweet_mode="extended", **kwargs)

        #add to list
        user_tweets.extend(new_tweets)

        #save the id of the oldest tweet less one, this is the starting point for collecting further tweets.
        oldest = user_tweets[-1]['id'] - 1
        #keep grabbing tweets until there are no tweets left to grab. Twitter limits us to 3,200 (including retweets)
        while len(new_tweets) > 0:
            try:
                #all subsequent requests use the max_id param to prevent duplicates
                new_tweets = twitter.get_user_timeline(screen_name=screen_name,count=200,include_rts=False,tweet_mode="extended",max_id=oldest, **kwargs)
                user_tweets.extend(new_tweets)
                oldest = user_tweets[-1]['id'] - 1
                print("...%s tweets downloaded so far" % (len(user_tweets)))
            except TwythonRateLimitError as e:
                #We have hit the rate limit, so we need to take a break.
                #find how much time need to sleep for.
                remainder = float(twitter.get_lastfunction_header(header='x-rate-limit-reset')) - time.time()
                print("sleeping for %d seconds" % remainder)
                #Pause until we can go again.
                time.sleep(remainder)
                continue

    except TwythonRateLimitError as e:
        #We have hit the rate limit on first call, so we need to take a break, and start again.
        #find how much time need to sleep for.
        remainder = float(twitter.get_lastfunction_header(header='x-rate-limit-reset')) - time.time()
        print("sleeping for %d seconds" % remainder)
        #Pause until we can go again.
        time.sleep(remainder)
        #start again
        return get_user_tweets(twitter, screen_name, **kwargs)

    except TwythonError as e:
        print(e)

    return user_tweets

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: %s screen_name" % sys.argv[0])
        sys.exit (1)

    #pass in the screen name of the account you want to download
    screen_name = sys.argv[1]

    #Authorize use of Twitter API with supplied credentials (from twitter_auth).
    twitter = Twython(consumer_key, consumer_secret, access_token, access_secret)

    tweets = get_user_tweets(twitter, screen_name)
    tweets_json.to_full_json(tweets, filepath="%s_tweets.json" % screen_name)
