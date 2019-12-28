import pandas as pd
import numpy as np
import csv

'''
Pretrained global vectors used for text vectorisation
'''
class Glove:
    def __init__(self):
        self.words = pd.read_table("GloVe/glove50dtwitter.txt", sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)

    '''
    Returns vector values of a word
    '''
    def vec(self, w):
        w = w.lower()
        try:
            ret = self.words.loc[w].to_numpy()
        except:
            ret = np.full((50,), 9999)
        return ret
