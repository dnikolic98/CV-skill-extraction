from preprocessor import Preprocessor
from inputExtractor import InputExtractor
from skillExtractNN import SkillsExtractorNN
from textFormater import TextFormater
import pandas as pd
import numpy as np

in_extractor = InputExtractor()
pp = Preprocessor()
tf = TextFormater()

word_features_dim, dense_features_dim = pp.getDim()
clf = SkillsExtractorNN(word_features_dim, dense_features_dim)

path = "saved/model(0_9711111).h5"
clf.load(path)

#prepare cv and return predictions
cv = open('predict.txt', 'r').read()
cv = tf.format(cv)
phrases, context, np_tags, context_tags = in_extractor.extract(cv)
phr_vec, cox_vec, phr_cox_vec = pp.preprocess(phrases,context, np_tags, context_tags)
predicted = clf.predict(np.array(phr_vec), np.array(cox_vec), np.array(phr_cox_vec))

for i in range(len(predicted)):
    if(np.argmax(predicted[i]) == 1):
        print(phrases[i])
    
