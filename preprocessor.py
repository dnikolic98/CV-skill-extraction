from glove import Glove
from featureVector import FeatureVector
import numpy as np

'''
Preprocesses string data(phrases, context and/or skills)
into vectors for nerual network.
'''
class Preprocessor():
    def __init__(self, context_n=3):
        self.glove = Glove()
        self.featureVector = FeatureVector()

    '''
    Transforms words into vectors using GloVe and concatenates vectors on x asxis
    '''
    def concat(self, phrase):
        phrase = phrase.split()
        ret_array = self.glove.vec(phrase[0])
        if len(phrase)>1:
            for word in phrase[1:]:
                ret_array = np.concatenate((ret_array, self.glove.vec(word)))
        return ret_array

    '''
    Main logic that for phrases, context and/or skills returns vector values.
    '''
    def preprocess(self, noun_phrases, context, skills=False):
        phrases_vec = []
        context_vec = []
        phr_cox_vec = []
        y=[]
        
        for i in range(len(noun_phrases)):
            phrases_vec.append(self.concat(noun_phrases[i]))
            context_vec.append(self.concat(context[i]))
            phr_cox_vec.append(self.featureVector.vectorise(noun_phrases[i]))
            if skills != False:
                if noun_phrases[i] in skills:
                    y.append(1)
                else:
                    y.append(0)
        if skills != False:
            return phrases_vec, context_vec, phr_cox_vec, y
        return phrases_vec, context_vec, phr_cox_vec
