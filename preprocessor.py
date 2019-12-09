from featureVector import FeatureVector
import numpy as np

'''
Preprocesses string data(phrases, context and/or skills)
into vectors for nerual network input.
'''
class Preprocessor():
    def __init__(self, context_n=3):
        self.featureVector = FeatureVector()
        self.phrase_dim = self.featureVector.dim()
        self.phr_cox_dim = self.phrase_dim*4

    def getDim(self):
        return self.phrase_dim, self.phr_cox_dim

    '''
    Transforms words into vectors using FeatureVector and stacks vectors into matrix
    '''
    def concat(self, phrase):
        phrase = phrase.split()
        ret_array = np.reshape(self.featureVector.vectorise(phrase[0]), (1,-1))
        if len(phrase)>1:
            for word in phrase[1:]:
                ret_array = np.vstack([ret_array, self.featureVector.vectorise(word)])
        return ret_array
    
    '''
    Returns concatenated vector of maximal and minimal features for given phrase vectors and context vectors 
    '''
    def minMaxVector(self, phrase_vec, context_vec):
        phrase_max_min = np.concatenate((phrase_vec.max(axis=0),phrase_vec.min(axis=0)))
        context_max_min = np.concatenate((context_vec.max(axis=0),context_vec.min(axis=0)))
        vector = np.concatenate((phrase_max_min,context_max_min))                        
        return vector

    '''
    Main logic that for phrases, context and/or skills returns vector values.
    '''
    def preprocess(self, noun_phrases, context, skills=False):
        phrases_vec = []
        context_vec = []
        phr_cox_vec = []
        y=[]
        
        if skills != False:
            skills = [x.lower() for x in skills]
            
        for i in range(len(noun_phrases)):
            current_phrase_vec = self.concat(noun_phrases[i])
            phrases_vec.append(current_phrase_vec)
            current_context_vec = self.concat(context[i])
            context_vec.append(current_context_vec)
            phr_cox_vec.append(self.minMaxVector(current_phrase_vec, current_context_vec))
            if skills != False:
                if noun_phrases[i].lower() in skills:
                    y.append(1)
                else:
                    y.append(0)

        if skills != False:
            return np.array(phrases_vec), np.array(context_vec), np.array(phr_cox_vec), np.array(y)
        return np.array(phrases_vec), np.array(context_vec), np.array(phr_cox_vec)
