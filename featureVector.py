import re
import numpy as np
import pandas as pd
from glove import Glove
'''
Word vectorising based on representing presence or
absence of multiple binary features and GloVe
'''
class FeatureVector:

    def __init__(self):
        self.glove = Glove()
        self.initThemes()
        
        
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
        vector = np.concatenate((vector,self.thematicVector(word)))
        vector = np.concatenate((vector,self.glove.vec(word)))
        return vector

    
    def thematicVector(self,word):
        vector = []
        
        #thematic word lists
        for column in self.df:
            if word in self.df[column].values:
                vector.append(1)
            else:
                vector.append(0)
                
        #english vocabulary        
        if 1 in vector:
            vector.append(1)
        else:
            vector.append(0)
            
        return np.asarray(vector)
    
    def dim(self):
        return len(self.vectorise("test"))

    def initThemes(self):
        path = "dataset/theme_words/"
        df = pd.read_csv(path + "0.csv", header = 0, encoding='latin-1')
        for i in range(1, 171):
            file_path = path + str(i) + ".csv"
            df_temp = pd.read_csv(file_path, header = 0, encoding='latin-1')
            df = pd.merge(df,df_temp, right_index=True, left_index=True, how="outer")
        self.df = df.replace(np.nan, '', regex=True)

fv = FeatureVector()
x = fv.vectorise("test")
