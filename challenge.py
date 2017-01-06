# SemEval 2017 Challenge - Task 6
# HashTagWars
#  --------------------------------------------------------------------------------
# imports

import re
import nltk
import numpy
import sklearn
import os
# --------------------------------------------------------------------------------
# functions

def readFileByLineAndTokenize(file):
    tweets = [line.rstrip('\n') for line in open("test/"+file, 'r')]
    tokenized = [re.split('\s| ', tweet) for tweet in tweets]
    [tweet.remove('') if tweet.__contains__('') else tweet for tweet in tokenized]
    return tokenized

def filterText(text, hashtag):
    tweets = []
    for tweet in text:
        filtered = []
        for token in tweet:
            if(not(token.startswith('@') | token.startswith('http'))):
                filtered.append(token)
        tweets.append(filtered)
    return tweets

def getData(text):
    regex = ""
    pattern = re.compile(regex)
    match = pattern.search(text)
    return match
# --------------------------------------------------------------------------------
# main

for f in os.listdir(os.getcwd()+"/test"):
    hashtag = "#"+str(os.path.basename(f)[:-4].replace("_",""))
    print(hashtag)
    text = readFileByLineAndTokenize(f)
    tweets = filterText(text, hashtag)
    print(tweets)

