from glove import Glove
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np

class Preprocessor():
    def __init__(self, context_n=3):
        self.glove = Glove()
        self.grammar = r"""
            NP: {<DT|PP\$>?<JJ>*<NN>+}   # chunk determiner/possessive, adjectives and noun
                {<NNP>+}        # chunk sequences of proper nouns
        """
        self.rexParser = nltk.RegexpParser(self.grammar)
        self.context_n = context_n

    def concat(self, phrase):
        phrase = phrase.split()
        ret_array = self.glove.vec(phrase[0])        
        if len(phrase)>1:
            for word in phrase[1:]:
                ret_array = np.concatenate((ret_array, self.glove.vec(word)))
        return ret_array

    def npExtraction(self, sentence):
        tagged = self.rexParser.parse(nltk.pos_tag(nltk.word_tokenize(sentence)))
        ret_list = []
        for tree in tagged.subtrees(filter=lambda t: t.label() == 'NP'):
            ret_list.append(' '.join([child[0] for child in tree.leaves()]))
        return ret_list

    def contextExtraction(self, phrase, sentence, n):
        return "nadodaj metodu"

    def preprocessY(self, cv, skill):
        sentences = sent_tokenize(cv)
        noun_phrases = []
        phrases_vec = []
        context = []
        context_vec = []
        phr_cox_vec = []
        y=[]
        
        for sentence in sentences:
            current_phrases = self.npExtraction(sentence)
            noun_phrases += current_phrases

            for phrase in current_phrases:
                context += self.contextExtraction(phrase,sentence, self.context_n)
            
        for noun_phrase in noun_phrases:
            phrases_vec.append(self.concat(noun_phrase))
            if noun_phrase in skill:
                y.append(1)
            else:
                y.append(0)

        return noun_phrases, context_vec, phr_cox_vec, y

'''
pp = Preprocessor()
nounp= "Software engineer"
s = 'Software engineer on an educational game for schoolers. The game was based on the story of "Tom Sawyer". The game was developed on Delphi and Java.'

x= pp.preprocessY(s,nounp.split())
print(x)
'''
