from config import *

# 自定義class, 計算標題與文章內容出現的詞
class et_nlp_class:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # 一開始已有sos及eos

    def addSentence(self, sentence):
        for word in sentence.split(' '): #對每個型態 "我 很帥"，將'我'跟'很帥'加入字典中
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1