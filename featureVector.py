import re
import numpy as np
from glove import Glove
'''
Word vectorising based on representing presence or
absence of multiple binary features and GloVe
'''
class FeatureVector:

    def __init__(self):
        self.glove = Glove()
        
    def features(self, word):
        vector = []
        symbols = re.compile('[@_!#$%^&*()<>?/\|}{~:]')
        numbers = re.compile('\d')
        
        vector.append(1 if symbols.search(word) != None else 0)   #contains symbol
        vector.append(1 if numbers.search(word) != None else 0)   #contains number
        vector.append(1 if word[0].isupper() else 0)              #first letter capital
        vector.append(1 if word.isupper() else 0)                 #whole word capital
        return np.asarray(vector)

    def vectorise(self, word):
        vector = self.features(word)
        vector = np.concatenate((vector,self.glove.vec(word)))
        return vector

    def dim(self):
        return len(self.vectorise("test"))
