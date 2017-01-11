SemEval 2017 Challenge - Task 6
#HashTagWars (http://alt.qcri.org/semeval2017/task6/)

 - Current AI technique for humour -> binary classification (0/1 - funny/not funny)
 - Humour has a continous nature and is subjective (wit, puns, cleverness)
 - @midnight
 - DATA: tweet - #hashtag - score (0,1,2)
 - GOAL: characterize the sense of humour, predict which tweet will be funny within the hashtag (theme)
    predictive model, pairwise comparison, external knowledge

TODO:
    # clustering hashtags, sysnsets to define similiar hashtags -> classify only with them
    # feature functions:
        dataset - slang
        dataset - profanity
        dataset - gazetter (lastna imena)
        dataset - number of positive, negative words -> polarity factor (pos/neg) -> ~1 == non humour
        perplexity of a tweet -> bigger == humour
        number of POS tags a token (word) can have -> more == ambiguity (sarcasm)
        hypernym distance between synsets (2 verbs/nouns -> get hypernym -> distance between synsets) -> large distance == humour
        lexical diversity per tweet -> larger == humour (nltk.Text; round(len(set(textT))/len(textT)*100, 2))
        tokens from hashtag ->  synset(token, tweet) -> wup_similarity -> max -> semantic relatedness -> lower == humour
        pos tagging of tweet, counting tags -> calculate sentence complexity -> more == humour, irony, sarcasm
        alliteration: "Infants don’t enjoy infancy like adults do adultery" -> tokens, stems
        antonymy: "A clean desk is a sign of a cluttered desk drawer"
        pos taging: verb vs. noun ratio
        use of emoticons
        disconnection: the maximum meaning distance of word pairs in a sentence.
        repetition: the minimum meaning distance of word pairs in a sentence.
        sense combination: t = count tags per token; s = multiply tag number (t1*t2*t3); log(s) -> all combinations of tags