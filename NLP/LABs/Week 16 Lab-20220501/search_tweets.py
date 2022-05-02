#!/usr/bin/env python3

import sys
import time

from twython import Twython, TwythonError, TwythonRateLimitError #https://github.com/ryanmcgrath/twython

from twitter_auth import *
import tweets_json

def search_twitter(twitter, search_term, limit, **kwargs):
    """uses twitter (Twython object) to collect tweets returned from given search_term, up to limit, , can include extra search parameters, returns list of tweets"""

    #initialise list of tweets
    tweets = []

    try:
        #count=100 is the maximum allowed
        #tweet_mode="extended" allows for full text tweets, rather than truncated (i.e. over 140 chars)
        #https://developer.twitter.com/en/docs/tweets/search/api-reference/get-search-tweets.html
        search_results = twitter.search(q=search_term,tweet_mode="extended",count=100, **kwargs)
        tweets.extend(search_results['statuses'])

        if len(tweets) == 0:
            print("No results found")
            return tweets

        #save the id of the oldest tweet less one, this is the starting point for collecting further tweets.
        oldest = tweets[-1]['id'] - 1
        #keep grabbing tweets until there are no tweets left to grab.
        while len(search_results['statuses']) > 0 and len(tweets) < limit:
            try:
                #all subsequent requests use the max_id param to prevent duplicates
                search_results = twitter.search(q=search_term,tweet_mode="extended",count=100,max_id=oldest, **kwargs)
                tweets.extend(search_results['statuses'])
                oldest = tweets[-1]['id'] - 1
                print("...%s tweets downloaded so far" % (len(tweets)))
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
        return search_twitter(twitter, search_term, limit, **kwargs)

    except TwythonError as e:
        print(e)

    return tweets[:limit]

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: %s search_term limit" % sys.argv[0])
        sys.exit (1)

    #Authorize use of Twitter API with supplied credentials (from twitter_auth).
    twitter = Twython(consumer_key, consumer_secret, access_token, access_secret)

    search_term = sys.argv[1]
    limit = int(sys.argv[2])
    tweets = search_twitter(twitter, search_term, limit)
    tweets_json.to_just_text(tweets, filepath="%s_tweets.txt" % search_term)
