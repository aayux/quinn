import os

import numpy as np
import pickle as pckl

from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

porter = PorterStemmer()
wnl = WordNetLemmatizer()

def process_line(sentence, tokenizer=word_tokenize):
    lemmas = []

    for idx, word in enumerate(tokenizer(sentence)):
        lemmas.append(lemmatize(word.lower()))

    lemmas = [lemma for lemma in lemmas if lemma.isalpha()]
    return lemmas

def lemmatize(word, lemmatizer=wnl, stemmer=porter):
    lemma = lemmatizer.lemmatize(word)
    stem = stemmer.stem(word)

    if not wordnet.synsets(lemma):
        if not wordnet.synsets(stem):
            return word
        else:
            return stem
    else:
        return lemma

def generate_vocab(filenames, write_filename='./data/embeddings/vocab.txt'):
    vocabulary = []
    for filename in filenames:
        with open(filename) as data_file:
            lines = [line.split('\t') for line in data_file.read().splitlines()]
                        
        for line in lines:
            words = process_line(line[1])
            vocabulary += [word for word in words if word not in vocabulary]
    
    vocabulary.sort()

    with open(write_filename, 'w') as write_file:
        for word in vocabulary:
            write_file.write("%s\n" % word)
    return

def zero_pad(data, maxlen=600):
    # sequence length is approx. equal to
    # the max length of sequence in train set
    return np.array([np.pad(seq, (0, maxlen-len(seq)), mode='constant') \
        if len(seq) < maxlen else np.array(seq[:maxlen]) for seq in data])

class Process(object):
    def __init__(self, line, is_test=False):        
        target_text = line[4]
        
        self.tokens = process_line(line[1])
        
        self.targets = {}
        target_text = process_line(target_text)
        for idx,_ in enumerate(self.tokens):
            if self.tokens[idx] in target_text:
                self.targets[idx] = True
        self.targets = list(self.targets.keys())
        self.targets.sort()
        
        self.label = None
        if not is_test:
            self.label = [float(line[9]), float(line[10])]


class DataLoader(object):
    def __init__(self, is_test=False):
        self.data = []        
        self.is_test = is_test

    def load(self, filename):
        with open(filename) as data_file:
            lines = [line.split('\t') for line in data_file.read().splitlines()]
                        
        for line in lines:
            self.data.append(Process(line, is_test=self.is_test))
        
        _x = np.array([data.tokens for data in self.data])
        _x = zero_pad(_x)

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


def create_dump(filename, write_filename):
    loader = DataLoader()
    x, x_attend, y, y_prob = loader.load(filename)
    pckl.dump((x, x_attend, y, y_prob), open(write_filename, "wb"))
    return

def load_embeddings(path, size, dimensions):
    
    embedding_matrix = np.zeros((size, dimensions), dtype=np.float32)

    size = os.stat(path).st_size
    with open(path, 'rb') as ifile:
        pos = 0
        idx = 0
        while pos < size:
            chunk = np.load(ifile)
            chunk_size = chunk.shape[0]
            embedding_matrix[idx:idx+chunk_size, :] = chunk
            idx += chunk_size
            pos = ifile.tell()
    return embedding_matrix
