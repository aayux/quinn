import os
import utils.data_utils as utils

train_file = './data/dumps/train.pckl'
val_file = './data/dumps/val.pckl'if not os.path.exists(train_file):
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

# DO: load and map glove
# https://nlp.stanford.edu/projects/glove/