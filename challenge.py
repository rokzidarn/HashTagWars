# SemEval 2017 Challenge - Task 6
# HashTagWars

#  --------------------------------------------------------------------------------
# imports
import re
import nltk
import numpy
import sklearn
import os
import string
from bs4 import BeautifulSoup
import requests

#  --------------------------------------------------------------------------------
# classes
class Tweet:
    def __init__(self, id, hashtag, text, score, tokens):
        self.id = id
        self.hashtag = hashtag
        self.text = text
        self.score = score
        self.tokens = tokens

    def __str__(self):
        return 'TWEET ID: %d | HASHTAG: %s | TEXT: %s | SCORE: %d' % (self.id, self.hashtag, self.text, self.score)

    def tweetTokens(self):
        return self.tokens

# --------------------------------------------------------------------------------
# functions
def readFileByLineAndTokenize(file, subdirectory):
    tweets = [line.rstrip('\n') for line in open(subdirectory+file, 'r', encoding="utf8")]
    tokenized = [re.split('\s| ', tweet) for tweet in tweets]
    return tokenized

def filterText(text):  # remove unnecesary data from tweet, such as extra hashtags, links
    tweets = []
    for tweet in text:
        filtered = []
        for token in tweet:
            if(not(token.startswith('#') or token.startswith('@') or token.startswith('.@') or token.startswith('http'))):
                filtered.append(token)
        while filtered.__contains__(''):
            filtered.remove('')
        tweets.append(filtered)
        #print(filtered)
    return tweets

def createData(tweets, hashtag): # remove puncuation & stopwords, transform to lowercase, lemmas, create objects
    data = []
    for tweet in tweets:
        text = " ".join(tweet[1:-1]).lower()
        table = text.maketrans({key: None for key in string.punctuation})
        text = text.translate(table)

        tokens = nltk.word_tokenize(text)
        tokens = [token for token in tokens if token not in nltk.corpus.stopwords.words('english')]
        lemmatizer = nltk.stem.WordNetLemmatizer()
        lemmas = [lemmatizer.lemmatize(token) for token in tokens]

        obj = Tweet(int(tweet[0]), hashtag, text, int(tweet[-1]), lemmas)
        #print(obj)
        data.append(obj)
    return data

def printData(hashtagTweets):
    for j in range(len(hashtagTweets)):
        print(hashtagTweets[j])
        print("  TOKENS: " + str(hashtagTweets[j].tweetTokens()))

def getAllTokensFromHashtag(hashtagTweets):
    all = []
    for tweet in hashtagTweets:
        for token in tweet.tokens:
            all.append(token)
    return all

def getTweetsInHashtagByScore(hashtagTweets): # score > 0
    funny = []
    notfunny = []
    for tweet in hashtagTweets:
        if tweet.score > 0:
            funny.append(tweet)
        else:
            notfunny.append(tweet)
    return [notfunny, funny]

def processTweets(hashtagTweets):  # basic data, frequency, important words, synsets
    print(hashtagTweets[0].hashtag)
    list = getAllTokensFromHashtag(hashtagTweets)
    freq = nltk.FreqDist(list)
    mostCommonWords = sorted(freq.items(), key = lambda x: x[1], reverse = True)[:2]
    print("2 most common word: {}, {}".format(mostCommonWords[0][0], mostCommonWords[1][0]))

    synset1 = nltk.corpus.wordnet.synsets(mostCommonWords[0][0])
    synset2 = nltk.corpus.wordnet.synsets(mostCommonWords[1][0])
    if (len(synset1) > 0 and len(synset2) > 0):
        synonyms = []
        for lemma in synset1[0].lemmas():  # first synset only
            synonyms.append(lemma.name())
        print("{} synonyms:  '{}'".format(mostCommonWords[0][0],set(synonyms)))
        for lemma in synset2[0].lemmas():  # first synset only
            synonyms.append(lemma.name())
        print("{} synonyms:  '{}'".format(mostCommonWords[1][0], set(synonyms)))

        sim = synset1[0].wup_similarity(synset2[0])
        if(sim is not None):
            sim = float(sim)
            print("2 most common words similarity: {} vs. {}: {:.2}"
                  .format(mostCommonWords[0][0], mostCommonWords[1][0],sim))
    else:
        print("Similarity:  / - word not found in synset")

# returns a list of puns from url
# url = "http://www.punoftheday.com/cgi-bin/findpuns.pl?q=dog"
def getPuns(url):
    puns = []
    r = requests.get(url)
    data = r.text
    soup = BeautifulSoup(data, "lxml")
    for link in soup.find_all('td')[1::2]:
        puns.append(link.get_text())
    return puns

# --------------------------------------------------------------------------------
# main
dataList = [] # 2D list, [hashtag][tweet]
hashtags = [] # list of all hashtags
subdirectory = "train_data/"
for f in os.listdir(os.getcwd()+"/"+subdirectory):  # preprocessing
    hashtag = "#"+str(os.path.basename(f)[:-4].replace("_", ""))
    hashtags.append(hashtag)
    text = readFileByLineAndTokenize(f, subdirectory)
    tweets = filterText(text)
    hashtagList = createData(tweets, hashtag)
    dataList.append(hashtagList)

#for h in range(len(hashtags)):
    #hashtagTweets = dataList[h]
    #printData(hashtagTweets)

for i in range(len(dataList)):  # process each category (hashtag) separately
    hashtagTweets = dataList[i] # tweets from the same hashtag
    #processTweets(hashtagTweets)
    tweetsByScore = getTweetsInHashtagByScore(hashtagTweets)  # 0 == not funny, 1 == funnny
    #printData(tweetsByScore)
    processTweets(tweetsByScore[0])
    print("-----------------------------")