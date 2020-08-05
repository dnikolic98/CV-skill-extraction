from keras.models import model_from_json
from preprocessor import Preprocessor
from inputExtractor import InputExtractor
from skillExtractNN import SkillsExtractorNN
from textFormater import TextFormater
import pandas as pd
import numpy as np
from itertools import islice

in_extractor = InputExtractor(1)
pp = Preprocessor()
tf = TextFormater()

word_features_dim, dense_features_dim = pp.getDim()
clf = SkillsExtractorNN(word_features_dim, dense_features_dim)

df = pd.read_excel("dataset/training/resumes.xlsx", sheet_name=0)
df = df.replace(np.nan, '', regex=True)

every_phrase_vec = []
every_context_vec = []
every_phr_cox_vec = []
every_y = []

for index, row in df.iterrows():
    cv = tf.format(row[0])
    phrases, context, np_tags, context_tags = in_extractor.extract(cv)
    phr_vec, cox_vec, phr_cox_vec, y = pp.preprocess(
        phrases, context, np_tags, context_tags, row[1].strip().split("|"))
    every_phrase_vec += phr_vec
    every_context_vec += cox_vec
    every_phr_cox_vec += phr_cox_vec
    every_y += y

hist = clf.fit(np.array(every_phrase_vec), np.array(
    every_context_vec), np.array(every_phr_cox_vec), np.array(every_y))
acc = hist.history['accuracy'][-1]

# save model
model_json = clf.model.to_json()
with open("saved_model/model(" + str(acc).replace(".", "_") + ").json", "w") as json_file:
    json_file.write(model_json)
clf.model.save_weights(
    "saved_model/model(" + str(acc).replace(".", "_") + ").h5")
print("Saved model to disk")
