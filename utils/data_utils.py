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

def map_to_vocab(sentence, vocab_dict='./data/dumps/vocab.pckl'):
    with open(vocab_dict, 'rb') as vfile:
        v_dict = pckl.load(vfile)
    
    return [v_dict.get(word, 0) for word in sentence]

def zero_pad(sequence, max_len=600):
    # sequence length is approx. equal to
    # the max length of sequence in train set    
    return np.pad(sequence, (0, max_len - len(sequence)), mode='constant') \
            if len(sequence) < max_len else np.array(sequence[:max_len])

class Process(object):
    def __init__(self, line, is_test=False):               
        
        self.tokens = process_line(line[1])
        
        self.target = np.zeros(len(self.tokens))
        target_words = process_line(line[4])
        for idx,_ in enumerate(self.tokens):
            if self.tokens[idx] in target_words:
                self.target[idx] = 1

        self.tokens = zero_pad(map_to_vocab(self.tokens))
        self.target = zero_pad(map_to_vocab(self.target))
        
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
        _x_map = np.array([data.target for data in self.data])
            
        if not self.is_test:
            _y = np.array([data.label[0] for data in self.data])
            _y_prob = np.array([data.label[1] for data in self.data])

        return _x, _x_map, _y, _y_prob

def fetch(filename):
    """
    Fetch the preprocessed data from dump
    """
    x, x_map, y, y_prob = pckl.load(open(filename, mode="rb"))
    return x, x_map, y, y_prob


def create_dump(filename, write_filename):
    loader = DataLoader()
    x, x_map, y, y_prob = loader.load(filename)
    pckl.dump((x, x_map, y, y_prob), open(write_filename, "wb"))
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
            embedding_matrix[idx:idx + chunk_size, :] = chunk
            idx += chunk_size
            pos = ifile.tell()
    return embedding_matrix

def batch_iter(data, batch_size, n_epochs, shuffle=False):
    print ("Generating batch iterator ...")
    data = np.array(data)
    data_size = len(data)
    n_batches_per_epoch = int((data_size - 1) / batch_size) + 1
    
    for epoch in range(n_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        
        for batch_num in range(n_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]