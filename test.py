from preprocessor import Preprocessor
from inputExtractor import InputExtractor
from skillExtractNN import SkillsExtractorNN
import numpy as np

in_extractor = InputExtractor()
pp = Preprocessor()
clf = SkillsExtractorNN(1,1)

#trenutno ne radi za sve ulaze
#sentence = 'Software engineer on an educational game for schoolers. The game was based on the story of "Tom Sawyer". The game was developed on Delphi and Java.'

sentence = 'Software engineer on an educational game for schoolers The game was based on the story of Tom Sawyer The game was developed on Delphi and Java'
skills = ["Software", "Engineer"]

phrases, context = in_extractor.extract(sentence)
phr_vec, cox_vec, phr_cox_vec, y = pp.preprocess(phrases,context, skills)
x=np.array(phr_vec)
p=np.array(cox_vec)
z=np.array(phr_cox_vec)
clf.fit(x,p,z,y)
#clf.fit(phr_vec,cox_vec, phr_cox_vec, y)

