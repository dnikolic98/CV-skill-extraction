import pandas as pd
import numpy as np
import csv

class Glove:
    def __init__(self):
        self.words = pd.read_table("GloVe/glove50dtwitter.txt", sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)

    
    def vec(self, w):
        w = w.lower()
        return self.words.loc[w].to_numpy()

glove = Glove()
x = glove.vec("software")
y = glove.words.loc["."].to_numpy()
