SemEval 2017 Challenge - Task 6
#HashTagWars (http://alt.qcri.org/semeval2017/task6/)

 - Current AI technique for humour -> binary classification (0/1 - funny/not funny)
 - Humour has a continous nature and is subjective (wit, puns, cleverness)
 - @midnight
 - DATA: tweet - #hashtag - score (0,1,2)
 - GOAL: characterize the sense of humour, predict which tweet will be funny within the hashtag (theme)
    predictive model, pairwise comparison, external knowledge


    twython not installed?? this is thrown by vader import, it doesn't seem to affect performance

TODO:
    # clustering hashtags, sysnsets to define similiar hashtags -> classify only with them
    # feature functions:
        !dataset - slang, contains -> true/false
        !dataset - emoticons, contains -> true/false

        *perplexity of a tweet -> bigger == humour
        *sense combination: t = count tags per token; s = multiply tag number (t1*t2*t3); log(s) -> all combinations of tags
        *max number of POS tags a token (word) can have -> more == ambiguity (sarcasm)

        hypernym distance between synsets (2 verbs/nouns -> get hypernym -> distance between synsets) -> large distance == humour
        tokens from hashtag ->  synset(token, tweet) -> wup_similarity -> max -> semantic relatedness -> lower == humour
        use of emoticons
        repetition: the minimum meaning distance of word pairs in a sentence.
