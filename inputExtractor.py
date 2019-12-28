import nltk
from nltk.tokenize import sent_tokenize

class InputExtractor:
    def __init__(self, context_n=3):
        self.grammar = r"""
            NP: {<DT|PP\$>?<JJ>*<NN>}   # chunk determiner/possessive, adjectives and noun
                {<NNP>+}        # chunk sequences of proper nouns
        """
        self.rexParser = nltk.RegexpParser(self.grammar)
        self.context_n = context_n

    '''
    Extracts all noun phrases from text(CV)
    '''
    def npExtraction(self, cv):
        tagged = self.rexParser.parse(nltk.pos_tag(nltk.word_tokenize(cv)))
        ret_np = []
        
        for tree in tagged.subtrees(filter=lambda t: t.label() == 'NP'):
            ret_np.append(' '.join([child[0] for child in tree.leaves()]))
        for tree in tagged.subtrees(filter=lambda t: t.label() == 'S'):
            #print(tree.leaves())
            ret_tag = [child[1] for child in tree.leaves()]
        return ret_np, ret_tag
    
    '''
    For given phrase extracts n-words from left and right.
    Returns list of context (single string) and
    int representing index of last given phrase
    '''
    def contextExtractionSingle(self, phrase, cv, n, tags):
        #print(phrase)
        index_phrase_char = cv.index(phrase)
        cv_words = cv.split()
        #print(len(cv_words))
        #print(cv_words)
        index_phrase = 0
        #print(len(tags))
        #print(tags)
    
        tags = tags[-len(cv_words):]
        x=0
        while(x<index_phrase_char):
            x += len(cv_words[index_phrase])+1
            index_phrase +=1
            #print(cv_words[index_phrase])
        
        start = index_phrase - n
        finish = index_phrase + len(phrase.split()) + n
        if start < 0:
            start = 0
        if finish > (len(cv_words)-1):
            finish = len(cv_words)

        
        ret_context= []
        ret_np_tag = []
        ret_cox_tag = []
        ret_context.append(' '.join(cv_words[start:finish]))
        ret_np_tag.append(' '.join(tags[index_phrase:index_phrase + len(phrase.split())]))
        ret_cox_tag.append(' '.join(tags[start:finish]))
        
        return ret_context, ret_np_tag, ret_cox_tag, index_phrase_char+len(phrase)+1, len(cv_words)
    
    '''
    Extracts context for each noun phrase and returns list of contexts
    '''
    def contextExtraction(self, noun_phrases, cv, n, tags):
        context_list = []
        np_tags = []
        cox_tags = []
        last = 0
        for phrase in noun_phrases:
            single_context, np_tag, cox_tag, x, cv_words = self.contextExtractionSingle(phrase,cv[last:], n, tags)
            last += x
            context_list += single_context
            np_tags += np_tag
            cox_tags += cox_tag
        return context_list, np_tags, cox_tags
    
    '''
    Extracts noun phrases and context of phrases for given text(CV)
    '''
    def extract(self, cv):
        noun_phrases, tagged = self.npExtraction(cv)
        context, np_tags, cox_tags = self.contextExtraction(noun_phrases ,cv ,self.context_n, tagged)
        return noun_phrases, context, np_tags, cox_tags
    
