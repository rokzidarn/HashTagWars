# SemEval 2017 Challenge - Task 6
# HashTagWars
#  --------------------------------------------------------------------------------
# imports
import re
import nltk
import numpy
import sklearn
import os

#  --------------------------------------------------------------------------------
# classes
class Tweet:
    def __init__(self, id, hashtag, text, score):
        self.id = id
        self.hashtag = hashtag
        self.text = text
        self.score = score

    def __str__(self):
        return 'TWEET ID: %d | HASHTAG: %s | TEXT: %s | SCORE: %d' % (self.id, self.hashtag, self.text, self.score)

# --------------------------------------------------------------------------------
# functions
def readFileByLineAndTokenize(file, subdirectory):
    tweets = [line.rstrip('\n') for line in open(subdirectory+file, 'r', encoding="utf8")]
    tokenized = [re.split('\s| ', tweet) for tweet in tweets]
    return tokenized

def filterText(text):
    tweets = []
    for tweet in text:
        filtered = []
        for token in tweet:
            if(not(token.startswith('#') or token.startswith('@') or token.startswith('.@') or token.startswith('http'))):
                filtered.append(token)
            if tweet.__contains__(''):
                tweet.remove('')
        tweets.append(filtered)
        #print(filtered)
    return tweets

def createData(tweets, hashtag):
    data = []
    for tweet in tweets:
        obj = Tweet(int(tweet[0]), hashtag, " ".join(tweet[1:-1]), int(tweet[-1]))
        print(obj)
        data.append(obj)
    return data

# --------------------------------------------------------------------------------
# main
dataList = []
subdirectory = "train_data/"
for f in os.listdir(os.getcwd()+"/"+subdirectory):
    hashtag = "#"+str(os.path.basename(f)[:-4].replace("_", ""))
    text = readFileByLineAndTokenize(f, subdirectory)
    tweets = filterText(text)
    hashtagList = createData(tweets, hashtag)
    dataList.append(hashtagList)


