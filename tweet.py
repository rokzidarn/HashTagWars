class Tweet:
    def __init__(self, id, text, hashtag, score):
        self.id = id
        self.hashtag = hashtag
        self.text = text
        self.score = score

    def __str__(self):
        return "Tweet ID: %d | HASHTAG: %s | TEXT: %s | SCORE: %d", self.id, self.hashtag, self.text, self.score