from preprocessor import Preprocessor
from inputExtractor import InputExtractor
from skillExtractNN import SkillsExtractorNN
import numpy as np

in_extractor = InputExtractor()
pp = Preprocessor()

word_features_dim, dense_features_dim = pp.getDim()
clf = SkillsExtractorNN(word_features_dim, dense_features_dim)#(x,y) x=duzina jednog retka(glove[50] + feature[0]); y=duzina za dense sloj(min max features)

#trenutno ne radi za sve ulaze
#sentence = 'Software engineer on an educational game for schoolers. The game was based onthe story of "Tom Sawyer". The game was developed on Delphi and Java.'

sentence = 'Software engineer on an educational game for schoolers The game was based on the story of Tom Sawyer The game was developed on Delphi and Java'
skills = ["software", "EngiNEer"]

phrases, context, np_tags, context_tags = in_extractor.extract(sentence)

phr_vec, cox_vec, phr_cox_vec, y = pp.preprocess(phrases,context, np_tags, context_tags, skills)
clf.fit(np.array(phr_vec), np.array(cox_vec), np.array(phr_cox_vec), np.array(y))

print(y)
