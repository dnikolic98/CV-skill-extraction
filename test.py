from preprocessor import Preprocessor
from inputExtractor import InputExtractor
#from skillExtractNN import SkillsExtractorNN

in_extractor = InputExtractor()
pp = Preprocessor()

#trenutno ne radi za sve ulaze
#sentence = 'Software engineer on an educational game for schoolers. The game was based on the story of "Tom Sawyer". The game was developed on Delphi and Java.'

sentence = 'Software engineer on an educational game for schoolers The game was based on the story of Tom Sawyer The game was developed on Delphi and Java'
skills = ["Software", "Engineer"]

phrases, context = in_extractor.extract(sentence)
phr_vec, cox_vec, phr_cox_vec, y = pp.preprocess(phrases,context, skills)
print(len(phr_vec), len(cox_vec), len(y))
print(phr_cox_vec)

