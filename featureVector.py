import re
import numpy as np

class FeatureVector:
    '''
    Word vectorising based on representing presence or
    absence of multiple binary features
    '''
    def vectorise(self, phrase):
        vector = []
        symbols = re.compile('[@_!#$%^&*()<>?/\|}{~:]')
        numbers = re.compile('\d')
        
        vector.append(1 if symbols.search(phrase) == None else 0)   #contains symbol
        vector.append(1 if numbers.search(phrase) == None else 0)   #contains number
        vector.append(1 if phrase[0].isupper() else 0)              #first letter capital
        vector.append(1 if phrase.isupper() else 0)                 #whole word capital
        return np.asarray(vector)
