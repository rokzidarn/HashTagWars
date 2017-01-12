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
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import TweetTokenizer

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

def getMostCommonWords(hashtagTweets):  # basic data, frequency, common words
    list = getAllTokensFromHashtag(hashtagTweets)
    freq = nltk.FreqDist(list)
    mostCommonWords = sorted(freq.items(), key = lambda x: x[1], reverse = True)[:2]
    #print("2 most common word: {}, {}".format(mostCommonWords[0][0], mostCommonWords[1][0]))
    return mostCommonWords[0][0]

def getPuns(url):  # returns a list of puns from url
    puns = []
    r = requests.get(url)
    data = r.text
    soup = BeautifulSoup(data, "lxml")
    for link in soup.find_all('td')[1::2]:
        puns.append(link.get_text())
    return puns

def processPuns(mostCommonWord):
    punsOfMostCommonWord = getPuns("http://www.punoftheday.com/cgi-bin/findpuns.pl?q=" + mostCommonWord + "&opt=text&submit=+Go%21+")
    if (len(punsOfMostCommonWord) > 0):
        #print("Pun example ({}): {}".format(mostCommonWord, punsOfMostCommonWord[0]))
        return punsOfMostCommonWord[0]
    else:
        #print("No puns found!")
        return None

# CLASSIFICATION
def numOfCapitalLettersFF(text):
    tokens = nltk.word_tokenize(text)
    up = 0
    for token in tokens:
        for char in token:
            if(char.isupper()):
                up += 1
    return up

def numOfStopWordsFF(text):
    tokens = nltk.word_tokenize(text)
    stop = 0
    stopWords = nltk.corpus.stopwords.words('english')
    for token in tokens:
        if token in stopWords:
            stop += 1
    return stop

def containsPunctuationFF(text):
    tokens = nltk.word_tokenize(text)
    for token in tokens:
        if token in string.punctuation:
            return True
    return False

def numOfLemmasFF(tokens):
    return len(tokens)

def containsMostCommonWordFF(tokens, mostCommonWord):
    if(mostCommonWord in tokens):
        return True
    return False

def cosineSimilarityToPunFF(text, pun):
    if(pun is not None):
        vect = sklearn.feature_extraction.text.TfidfVectorizer()
        tfidf = vect.fit_transform([text, pun])
        cosine = (tfidf * tfidf.T).A # if cosine > 0.7: return True else: return false
        return cosine[0][1]
    else:
        return 0

def getSentimentScores(text):   # returns scores for positive, negative sentiment
    sid = SentimentIntensityAnalyzer()
    ss = sid.polarity_scores(text)
    positive = ss["pos"]
    negative = ss["neg"]
    return [positive, negative]

def containsProfanity(text, profanityWords):
    tokens = nltk.word_tokenize(text)
    for token in tokens:
        if token in profanityWords:
            return True
    return False

def containsNegativeWords(text, negativeWords):
    tokens = nltk.word_tokenize(text)
    for token in tokens:
        if token in negativeWords:
            return True
    return False

def positiveToNegativeWordRatio(text, negativeWords, positiveWords):
    negative = 0
    positive = 0
    tokens = nltk.word_tokenize(text)
    for token in tokens:
        if token in negativeWords:
            negative += 1
        elif token in positiveWords:
            positive += 1
    if(positive == 0 or negative == 0):
        return 1  # non humour, equal
    else:
        return positive/negative

def calculatePerplexity(text, tokens):  # not working
    ngram = nltk.model.ngram.NgramModel(2, tokens)
    perplexity = ngram.perplexity(text)
    return perplexity

def calculateLexicalDiversity(text):
    textT = nltk.Text(text)
    if(len(textT) == 0):
        return 0
    return round(len(set(textT)) / len(textT) * 100, 2)

def maxPOSTags(tokens):  # not working, want to get all possible for token
    for token in tokens:
        postag = nltk.tag.pos_tag([token])
        print(postag)

def calculateVerbToNounRatio(tokens):
    nouns = 0
    verbs = 0
    for token in tokens:
        postag = nltk.tag.pos_tag([token])[0][1]
        if(postag == 'NN'):
            nouns += 1
        elif(postag == 'VB'):
            verbs += 1
    if(verbs == 0 or nouns == 0):
        return 0
    else:
        return verbs/nouns

def containsEmoticons(text, emoticonList):
    tweetTokenizer = TweetTokenizer()
    tokens = tweetTokenizer.tokenize(text)
    for token in tokens:
        if token in emoticonList:
            return True
    return False

def calculateSemanticRelatedness(hashtagTokens, textTokens):  # get tokens from hashtag + test
    maxSR = 0
    for htoken in hashtagTokens:
        for token in textTokens:
            synset1 = nltk.corpus.wordnet.synsets(htoken)
            synset2 = nltk.corpus.wordnet.synsets(token)
            if (len(synset1) > 0 and len(synset2) > 0):
                sim = synset1[0].wup_similarity(synset2[0])
                if (sim is not None and sim > maxSR):
                    maxSR = sim
    return maxSR

def classifyTweetsByHashtag(hashtagTweets, mostCommonWord, profanityWords, negativeWords, positiveWords, emoticons):
    features = []
    classes = []
    pun = processPuns(mostCommonWord)

    for tweet in hashtagTweets:
        curr = []  # features of current tweet, FF - feature functions
        curr.append(numOfCapitalLettersFF(tweet.text))
        curr.append(numOfStopWordsFF(tweet.text))
        curr.append(containsPunctuationFF(tweet.text))
        curr.append(numOfLemmasFF(tweet.tokens))
        curr.append(containsMostCommonWordFF(tweet.tokens, mostCommonWord))
        curr.append(cosineSimilarityToPunFF(tweet.text, pun))
        sentimentScores = getSentimentScores(tweet.text)
        curr.append(sentimentScores[0])
        curr.append(sentimentScores[1])
        curr.append(containsProfanity(tweet.text, profanityWords))
        curr.append(containsNegativeWords(tweet.text, negativeWords))
        curr.append(containsEmoticons(tweet.text, emoticons))
        #curr.append(calculatePerplexity(tweet.text, tweet.tokens))
        curr.append(calculateLexicalDiversity(tweet.text))
        #curr.append(maxPOSTags(tweet.tokens))
        curr.append(calculateVerbToNounRatio(tweet.tokens))
        curr.append(positiveToNegativeWordRatio(tweet.text, negativeWords, positiveWords))

        features.append(curr)
        score = tweet.score
        if(score == 2):  # problem, beacuse there is only one representative of "super" funny
            score = 1
        classes.append(score)

    (X, y) = (numpy.array(features), numpy.array(classes))
    print(("Dataset shape: {}".format((X.shape, y.shape))))
    from sklearn.naive_bayes import MultinomialNB
    clf = MultinomialNB(alpha=.01)
    clf.fit(X, y)

    scorings = ["accuracy"]  # "precision_weighted", "recall_weighted", "f1_weighted"
    for scoring in scorings:
        scores = sklearn.model_selection.cross_val_score(clf, X, y, cv=5, scoring=scoring)
        print(scores)

# CLUSTERING
def tokenizeHashtag(rawHashtags):
    all = []
    for hashtag in rawHashtags:
        tokens = hashtag.split("_")
        for token in tokens:
            all.append(token)

    from nltk.stem.snowball import SnowballStemmer
    stemmer = SnowballStemmer("english")
    stopwords = nltk.corpus.stopwords.words('english')
    stems = [stemmer.stem(t) for t in all if t not in stopwords]
    return stems

def hashtagClustering(rawHashtags, numClusters):
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer(
        max_df=0.8,
        max_features=200,  # max number of words
        min_df=0.2,
        stop_words='english',
        use_idf=True,
        tokenizer=tokenizeHashtag,
        ngram_range=(1, 3))

    tfidf_matrix = tfidf_vectorizer.fit_transform(rawHashtags)
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=numClusters)  # test different number of clusters
    km.fit(tfidf_matrix)

    clusters = km.labels_.tolist()
    #print("Clusters: {}".format(clusters))
    visualizeDendrogram(tfidf_matrix)

def visualizeDendrogram(tfidf_matrix):
    from sklearn.metrics.pairwise import cosine_similarity
    dist = 1 - cosine_similarity(tfidf_matrix)

    from scipy.cluster.hierarchy import ward, dendrogram
    linkage_matrix = ward(dist)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(30, 40))  # set size
    ax = dendrogram(linkage_matrix, orientation="right", labels=rawHashtags)

    plt.tick_params(
        axis='x',
        which='both',
        bottom='off',
        top='off',
        labelbottom='off')

    #plt.tight_layout()
    plt.show()
    # plt.savefig('ward_clusters.png', dpi=200) # save figure as ward_clusters
    plt.close()

def getWordData():
    with open("word_data/profanity-words.txt") as f:
        profanityWords = f.readlines()
    profanityWords = [line.strip('\n') for line in profanityWords]

    with open("word_data/negative-words.txt") as f:
        negativeWords = f.readlines()
    negativeWords = [line.strip('\n') for line in negativeWords]

    with open("word_data/positive-words.txt") as f:
        positiveWords = f.readlines()
    positiveWords = [line.strip('\n') for line in positiveWords]

    with open("word_data/emoticons.txt") as f:
        emoticons = f.readlines()
    emoticons = [line.strip('\n') for line in emoticons]

    return [profanityWords, negativeWords, positiveWords, emoticons]

# --------------------------------------------------------------------------------
# main
dataList = []  # 2D list, [hashtag][tweet]
hashtags = []  # list of all hashtags
rawHashtags = []  # raw hashtags, easier to tokenize in clustering
subdirectory = "test_data/"
for f in os.listdir(os.getcwd()+"/"+subdirectory):  # preprocessing
    hashtag = "#"+str(os.path.basename(f)[:-4].replace("_", ""))
    rawHashtags.append(os.path.basename(f)[:-4])
    hashtags.append(hashtag)
    text = readFileByLineAndTokenize(f, subdirectory)
    tweets = filterText(text)
    hashtagList = createData(tweets, hashtag)
    dataList.append(hashtagList)

wordData = getWordData()
#hashtagClustering(rawHashtags, 6)

for i in range(len(dataList)):  # process each category (hashtag) separately
    hashtagTweets = dataList[i]  # tweets from the same hashtag
    #tweetsByScore = getTweetsInHashtagByScore(hashtagTweets)  # 0 == not funny, 1 == funnny
    #printData(tweetsByScore)
    mostCommonWord = getMostCommonWords(hashtagTweets)
    classifyTweetsByHashtag(hashtagTweets, mostCommonWord, wordData[0], wordData[1], wordData[2], wordData[3])
    #print("-----------------------------")
