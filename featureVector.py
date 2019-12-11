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
        self.initPreSuf()
        self.initPos()
        
    def features(self, word):
        vector = []
        symbols = re.compile('[@_!#$%^&*()<>?/\|}{~:]')
        numbers = re.compile('\d')
        
        vector.append(1 if symbols.search(word) != None else 0)   #contains symbol
        vector.append(1 if numbers.search(word) != None else 0)   #contains number
        vector.append(1 if word[0].isupper() else 0)              #first letter capital
        vector.append(1 if word.isupper() else 0)                 #whole word capital
        return np.asarray(vector)

    

    def vectorise(self, word, tag):
        vector = self.features(word)
        vector = np.concatenate((vector,self.thematicVector(word)))
        vector = np.concatenate((vector,self.preSufVector(word)))
        vector = np.concatenate((vector,self.posVector(tag)))
        vector = np.concatenate((vector,self.glove.vec(word)))
        return vector

    
    def thematicVector(self,word):
        vector = []
        
        #thematic word lists
        for column in self.df_themes:
            if word in self.df_themes[column].values:
                vector.append(1)
            else:
                vector.append(0)
                
        #english vocabulary        
        if 1 in vector:
            vector.append(1)
        else:
            vector.append(0)
            
        return np.asarray(vector)

    def posVector(self,tag):
        vector = np.zeros(self.pos_vector_size)
        for index, row in self.df_pos.iterrows():
            if(tag == row[0]):
                vector[index] = 1
                break
        return vector

    def preSufVector(self,word):
        vector = np.zeros(self.pre_vector_size + self.suf_vector_size)
        
        for index, row in self.df_prefix.iterrows():
            if(word.startswith(row[0])):
                vector[index] = 1
                break
            
        for index, row in self.df_suffix.iterrows():
            if(word.startswith(row[0])):
                vector[index + self.pre_vector_size] = 1
                break
        return vector
    
    def dim(self):
        return len(self.vectorise("test", "NN"))

    def initThemes(self):
        path = "dataset/theme_words/"
        df = pd.read_csv(path + "0.csv", header = 0, encoding='latin-1')
        for i in range(1, 171):
            file_path = path + str(i) + ".csv"
            df_temp = pd.read_csv(file_path, header = 0, encoding='latin-1')
            df = pd.merge(df,df_temp, right_index=True, left_index=True, how="outer")
        self.df_themes = df.replace(np.nan, '', regex=True)

    def initPreSuf(self):
        path = "dataset/pre_suf/"
        pre_file_path = path + "prefix.csv"
        suf_file_path = path + "suffix.csv"

        self.df_prefix = pd.read_csv(pre_file_path, header = 0, encoding='latin-1').sort_values(by='prefix', ascending=False)
        self.df_suffix = pd.read_csv(suf_file_path, header = 0, encoding='latin-1').sort_values(by='suffix', ascending=False)
        self.pre_vector_size = self.df_prefix.size
        self.suf_vector_size = self.df_suffix.size

    def initPos(self):
        path = "dataset/pos/"
        pre_file_path = path + "POStag.csv"
        self.df_pos = pd.read_csv(pre_file_path, header = 0, encoding='latin-1')
        self.pos_vector_size = self.df_pos.size
