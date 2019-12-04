import nltk
from nltk.tokenize import sent_tokenize

class InputExtractor:
    def __init__(self, context_n=3):
        self.grammar = r"""
            NP: {<DT|PP\$>?<JJ>*<NN>+}   # chunk determiner/possessive, adjectives and noun
                {<NNP>+}        # chunk sequences of proper nouns
        """
        self.rexParser = nltk.RegexpParser(self.grammar)
        self.context_n = context_n

    def npExtraction(self, sentence):
        tagged = self.rexParser.parse(nltk.pos_tag(nltk.word_tokenize(sentence)))
        ret_list = []
        for tree in tagged.subtrees(filter=lambda t: t.label() == 'NP'):
            ret_list.append(' '.join([child[0] for child in tree.leaves()]))
        return ret_list

    def contextExtractionSingle(self, phrase, cv, n):
        index_phrase_char = cv.index(phrase)
        cv_words = cv.split()
        index_phrase = 0
        x=0
        
        while(x<index_phrase_char):
            x += len(cv_words[index_phrase])+1
            index_phrase +=1
            
        start = index_phrase - n
        finish = index_phrase + len(phrase.split()) + n
        if start < 0:
            start = 0
        if finish > (len(cv_words)-1):
            finish = len(cv_words)
        ret_list= []
        ret_list.append(' '.join(cv_words[start:finish]))
        return ret_list, index_phrase_char+len(phrase)

    def contextExtraction(self, noun_phrases, cv, n):
        ret_list = []
        last = 0
        for phrase in noun_phrases:
            result, x = self.contextExtractionSingle(phrase,cv[last:],n)
            last += x
            ret_list += result
            
        return ret_list

    def extract(self, cv):
        noun_phrases = self.npExtraction(cv)
        context = self.contextExtraction(noun_phrases ,cv ,self.context_n)
        return noun_phrases, context


