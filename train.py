import os
import utils.data_utils as utils
import numpy as np

train_file = './data/dumps/train.pckl'
val_file = './data/dumps/val.pckl'

if not os.path.exists(train_file):
    print ('Train dump not found. Preparing data ...')
    train_tsv_path = './data/train/english/Wikipedia_Train.tsv'
    utils.create_dump(train_tsv_path, train_file)

if not os.path.exists(val_file):
    print ('Validation dump not found. Preparing data ...')
    val_tsv_path = './data/train/english/Wikipedia_Dev.tsv'
    utils.create_dump(val_tsv_path, val_file)

# Load train data
print ('Loading dataset from ./data/dumps/ ...')
x_train, x_train_att, y_train, y_train_prob = utils.load(train_file)
x_val, x_val_att, y_val, y_val_prob = utils.load(val_file)

vocab_size = 3193
embedding_path = './data/dumps/embeddings.npy'
embedding = utils.load_embeddings(embedding_path, vocab_size, dimensions=300)
print ("Embeddings loaded, Vocabulary Size: {:d}. Starting training ...".format(vocab_size))

np.random.seed(10)

shuff_idx = np.random.permutation(np.arange(len(y_train)))
x_train, x_train_att, y_train, y_train_prob = \
x_train[shuff_idx], x_train_att[shuff_idx], y_train[shuff_idx], y_train_prob[shuff_idx]

shuff_idx = np.random.permutation(np.arange(len(y_val)))
x_val, x_val_att, y_val, y_val_prob = \
x_val[shuff_idx], x_val_att[shuff_idx], y_val[shuff_idx], y_val_prob[shuff_idx]

# TO DO: Load model and train