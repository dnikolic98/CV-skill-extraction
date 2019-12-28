from preprocessor import Preprocessor
from inputExtractor import InputExtractor
from skillExtractNN import SkillsExtractorNN
from textFormater import TextFormater
import pandas as pd
import numpy as np

def predict(cv):
    phrases, context, np_tags, context_tags = in_extractor.extract(cv)
    phr_vec, cox_vec, phr_cox_vec = pp.preprocess(phrases,context, np_tags, context_tags)
    print(np.array(phr_vec)[0], np.array(cox_vec)[0], np.array(phr_cox_vec)[0])
    clf.predict(np.array(phr_vec)[0], np.array(cox_vec)[0], np.array(phr_cox_vec)[0])

in_extractor = InputExtractor()
pp = Preprocessor()
tf = TextFormater()

word_features_dim, dense_features_dim = pp.getDim()
clf = SkillsExtractorNN(word_features_dim, dense_features_dim)

df_skills = pd.read_csv("dataset/training/skills.csv")
df_cv = pd.read_csv("dataset/training/cv.csv")
df = pd.merge(df_cv,df_skills, right_index=True, left_index=True, how="outer")

every_phrase_vec = []
every_context_vec = []
every_phr_cox_vec = []
every_y = []

s1 = 'Software engineer on an educational game for schoolers The game was based on the story of Tom Sawyer The game was developed on Delphi and Java'
s2 = 'Software engineer on an educational game for schoolers The game was based on the story of Tom Sawyer The game was developed on Delphi and Java'
s = [s1,s2]
c = [["software", "EngiNEer"],["an educational game", "The game"]]
    
for i in range(1):
    print(df["cv"][i])
    cv = tf.format(df["cv"][i])
    print(cv)
    phrases, context, np_tags, context_tags = in_extractor.extract(cv)
    #for i in range(len(phrases)):
        #print(phrases[i], np_tags[i])
    
    phr_vec, cox_vec, phr_cox_vec, y = pp.preprocess(phrases,context, np_tags, context_tags, df["skill"][i].split())
    every_phrase_vec += phr_vec
    every_context_vec += cox_vec
    every_phr_cox_vec += phr_cox_vec
    every_y += y

'''
for index, row in df.iterrows():
    cv = tf.format(row[0])
    print(cv)
    phrases, context, np_tags, context_tags = in_extractor.extract(cv)
    #for i in range(len(phrases)):
        #print(phrases[i], np_tags[i])
    
    phr_vec, cox_vec, phr_cox_vec, y = pp.preprocess(phrases,context, np_tags, context_tags, row[1].split())
    every_phrase_vec += phr_vec
    every_context_vec += cox_vec
    every_phr_cox_vec += phr_cox_vec
    every_y += y
'''    

clf.fit(np.array(every_phrase_vec), np.array(every_context_vec), np.array(every_phr_cox_vec), np.array(every_y))
predict(s1)
