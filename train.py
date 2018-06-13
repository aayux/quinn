import os

import time
import datetime

import numpy as np
import tensorflow as tf
import utils.data_utils as utils

from models.quinn import Quinn

tf.logging.set_verbosity(tf.logging.ERROR)

train_file = './data/dumps/train.pckl'
val_file = './data/dumps/val.pckl'

# Model Parameters
max_length = 600
vocab_size = 3193
embedding_dims = 300
hidden_layers = 64

# Training Parameters
batch_size = 64
num_epochs = 100
num_checkpoints = 3
checkpoint_every = 10

# Prepare and load training and validation data
if not os.path.exists(train_file):
    print ("Train dump not found. Preparing data ...")
    train_tsv_path = './data/train/english/Wikipedia_Train.tsv'
    utils.create_dump(train_tsv_path, train_file)

if not os.path.exists(val_file):
    print ("Validation dump not found. Preparing data ...")
    val_tsv_path = './data/train/english/Wikipedia_Dev.tsv'
    utils.create_dump(val_tsv_path, val_file)

print ('Loading dataset from ./data/dumps/ ...')
x_train, x_train_map, y_train, y_train_prob = utils.fetch(train_file)
x_val, x_val_map, y_val, y_val_prob = utils.fetch(val_file)

# Load embeddings
embedding_path = './data/dumps/embeddings.npy'
embedding = utils.load_embeddings(embedding_path, vocab_size, dimensions=300)
print ("Embeddings loaded, Vocabulary Size: {:d}.".format(vocab_size))


# Shuffle training data
np.random.seed(10)
shuff_idx = np.random.permutation(np.arange(len(y_train)))
x_train, x_train_map, y_train, y_train_prob = \
x_train[shuff_idx], x_train_map[shuff_idx], y_train[shuff_idx], y_train_prob[shuff_idx]


print ("Generating graph and starting training ...")

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
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

        sess.run(quinn.embedding_init, feed_dict={quinn.embedding_placeholder: embedding})

        def train_step(x_batch, x_map, y_batch):
            # seq_length = np.array([list(x).index(0) + 1 for x in x_batch])
            
            feed_dict = {
                quinn.input_x: x_batch,
                quinn.input_y: y_batch,
                quinn.attention_map: x_map
            }
            _, step, loss, mae, _update_op = sess.run(
                [train_op, global_step, quinn.loss, quinn.mae, quinn.update_op],
                feed_dict)
            
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, mae {:g}".format(time_str, step, loss, mae))

        def val_step(x_batch, x_map, y_batch):            
            feed_dict = {
                quinn.input_x: x_batch,
                quinn.input_y: y_batch,
                quinn.attention_map: x_map
            }
            step, loss, mae, _update_op = sess.run(
                [global_step, quinn.loss, quinn.mae, quinn.update_op],
                feed_dict)
            
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, mae {:g}".format(time_str, step, loss, mae))
            
        # Generate batches
        batches = utils.batch_iter(list(zip(x_train, x_train_map, y_train_prob)), batch_size, num_epochs)
        
        # Training loop. For each batch...
        for batch in batches:
            x_batch, x_map, y_batch = zip(*batch)
            train_step(x_batch, x_map, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            
            epoch_step = (int((len(x_train[0]) - 1) / batch_size) + 1)
            
            if current_step % epoch_step == 0:
                print("\nValidation:")
                
                # Randomly draw a validation batch
                shuff_idx = np.random.permutation(np.arange(batch_size))
                x_batch_val, x_batch_val_map, y_batch_val_prob = \
                x_val[shuff_idx], x_val_map[shuff_idx], y_val_prob[shuff_idx]
                
                val_step(x_batch_val, x_batch_val_map, y_batch_val_prob)
                
                print("")
            
            if current_step % checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))