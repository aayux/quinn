import os

import time
import datetime

import numpy as np
import tensorflow as tf
import utils.data_utils as utils

from models.quinn import Quinn

train_file = './data/dumps/train.pckl'
val_file = './data/dumps/val.pckl'

max_length = 600
vocab_size = 3193
embedding_dims = 300
hidden_layers = 64

batch_size = 64
num_epochs = 3
num_checkpoints = 3

if not os.path.exists(train_file):
    print ("Train dump not found. Preparing data ...")
    train_tsv_path = './data/train/english/Wikipedia_Train.tsv'
    utils.create_dump(train_tsv_path, train_file)

if not os.path.exists(val_file):
    print ("Validation dump not found. Preparing data ...")
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

# Training
with tf.Graph().as_default():
    
    session_conf = tf.ConfigProto(
      allow_soft_placement=True,
      log_device_placement=False)
    
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        quinn = Quinn(max_length=max_length, vocab_size=vocab_size, 
                      embedding_dims=embedding_dims, 
                      hidden_layers=hidden_layers)

        # Define Training procedure
        global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(quinn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, 'runs', timestamp))
        print("Writing to {}\n".format(out_dir))

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, 'checkpoints'))
        checkpoint_prefix = os.path.join(checkpoint_dir, 'model')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=num_checkpoints)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_batch):
            seq_length = np.array([list(x).index(0) + 1 for x in x_batch])
            
            feed_dict = {
                quinn.input_x: x_batch,
                quinn.input_y: y_batch,
                quinn.seq_length: seq_length,
                quinn.embedding_placeholder: embedding,
                quinn.keep_prob: 1.0
            }
            _, _embedding_op, step, loss, accuracy = sess.run(
                [train_op, quinn.embedding_init, global_step, quinn.loss, quinn.accuracy],
                feed_dict)
            
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

        def val_step(x_batch, y_batch):
            seq_length = np.array([list(x).index(0) + 1 for x in x_batch])
            
            feed_dict = {
                quinn.input_x: x_batch,
                quinn.input_y: y_batch,
                quinn.seq_length: seq_length,
                quinn.embedding_placeholder: embedding,
                quinn.dropout_keep_prob: 1.0
            }
            _embedding_op, step, loss, accuracy = sess.run(
                [quinn.embedding_init, global_step, quinn.loss, quinn.accuracy],
                feed_dict)
            
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            
        # Generate batches
        batches = utils.batch_iter(list(zip(x_train, y_train_prob)), batch_size, num_epochs)
        
        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            
            epoch_step = (int((len(x_train[0]) - 1) / batch_size) + 1)
            
            if current_step % epoch_step == 0:
                print("\nEvaluation:")
                val_step(x_val, y_val_prob)
                print("")
            
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
