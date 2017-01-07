SemEval 2017 Challenge - Task 6
#HashTagWars (http://alt.qcri.org/semeval2017/task6/)

 - Current AI technique for humour -> binary classification (0/1 - funny/not funny)
 - Humour has a continous nature and is subjective (wit, puns, cleverness)
 - @midnight
 - DATA: tweet - #hashtag - score (0,1,2)
 - GOAL: characterize the sense of humour, predict which tweet will be funny within the hashtag (theme)
    predictive model, pairwise comparison, external knowledge

OBSERVATIONS:
- sex, drugs, stereotypes related == funny (2)
- stemming, tfidf, sequence tagging -> useless?

TODO:
    1. categorization & classification
    2. wordnet.synsets
    3. nltk.Text
        3.1 concordance # every occurance of the most common word in context
        3.2 lexical diversity # round(len(set(textT))/len(textT)*100, 2)
        3.3 collocations # frequent bigrams
    4.

IDEAS:
    1. tweet topic clustering
    2. find funniest words for each cluster
    3. count funny words in tweets
    4. analysis of the funniest tweets
    5. sarcasm detector (?)
    6. pun database (http://www.punoftheday.com/cgi-bin/findpuns.pl?q=dog&opt=text&submit=+Go%21+)
