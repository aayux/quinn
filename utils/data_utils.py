import numpy as np
import pickle as pckl

from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

porter = PorterStemmer()
wnl = WordNetLemmatizer()

class Process(object):
    def __init__(self, line, is_test=False):        
        self.sentence = line[1]
        target_text = line[4]
        
        self.tokens = self.process_line(self.sentence)
        
        self.targets = {}
        target_text = self.process_line(target_text)
        for idx,_ in enumerate(self.tokens):
            if self.tokens[idx] in target_text:
                self.targets[idx] = True
        self.targets = list(self.targets.keys())
        self.targets.sort()
        
        self.label = None
        if not is_test:
            self.label = [float(line[9]), float(line[10])]
        
    def process_line(self, sentence, tokenizer=word_tokenize):
        lemmas = []

        for idx, word in enumerate(tokenizer(sentence)):
            lemmas.append(self.lemmatize(word.lower()))

        lemmas = [lemma for lemma in lemmas if lemma.isalpha()]
        return lemmas
    
    def lemmatize(self, word, lemmatizer=wnl, stemmer=porter):
        lemma = lemmatizer.lemmatize(word)
        stem = stemmer.stem(word)

        if not wordnet.synsets(lemma):
            if not wordnet.synsets(stem):
                return word
            else:
                return stem
        else:
            return lemma

class DataLoader(object):
    def __init__(self, is_test=False):
        self.data = []        
        self.is_test = is_test

    def load(self, file_path):
        with open(file_path) as file:
            lines = [line.split('\t') for line in file.read().splitlines()]
                        
        for line in lines:
            self.data.append(Process(line, is_test=self.is_test))
        
        _x = np.array([data.tokens for data in self.data])
        _x_attend = np.array([data.targets for data in self.data])
            
        if not self.is_test:
            _y = np.array([data.label[0] for data in self.data])
            _y_prob = np.array([data.label[1] for data in self.data])
        
        return _x, _x_attend, _y, _y_prob

def load(filename):
    """
    Load the preprocessed data
    """
    x, x_attend, y, y_prob = pckl.load(open(filename, mode="rb"))
    return x, x_attend, y, y_prob


def create_dump(file_path, write_filename):
    loader = DataLoader()
    x, x_attend, y, y_prob = loader.load(file_path)
    pckl.dump((x, x_attend, y, y_prob), open(write_filename, "wb"))
    return