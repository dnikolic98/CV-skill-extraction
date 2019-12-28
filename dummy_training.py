import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize
from textFormater import TextFormater

df_skills = pd.read_csv("dataset/training/skills.csv")
df_cv = pd.read_csv("dataset/training/cv.csv")
df = pd.merge(df_cv,df_skills, right_index=True, left_index=True, how="outer")


tf = TextFormater()

x = df["cv"][0].lower()
x = x.split("\n\n")

xx = []
for part in x:
    xx.append(tf.format(part).strip())
    #print(tf.format(part).strip())

xx = list(filter(None, xx))
#print(xx)
for part in xx:
    print(part)
