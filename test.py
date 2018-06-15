import os
import argparse

import numpy as np
import tensorflow as tf
import utils.data_utils as utils

from sklearn.metrics import accuracy_score, \
                            f1_score, precision_score, \
                            recall_score, mean_absolute_error

tf.logging.set_verbosity(tf.logging.WARN)

parser = argparse.ArgumentParser()
parser.add_argument('--ckptdir', type=str, default=None, help='Model checkpoint directory')
parser.add_argument('--dataset', type=str, default=None, help='TSV test file name')
parser.add_argument('--bsize', type=int, default=128, help='Test batch size')
opt = parser.parse_args()

test_file = './data/dumps/test.pckl'

# Prepare and load test data
if not os.path.exists(test_file):
    print ("Test dump not found. Preparing data ...")
    test_tsv_path = './data/english/' + opt.dataset + '.tsv'
    utils.create_dump(test_tsv_path, test_file)

print ("Loading dataset from {} ...".format(test_file))
x_test, x_test_map, y_test, y_test_prob = utils.fetch(test_file)

# Checkpoint directory from training run
load_checkpoint_dir = "./runs/" + opt.ckptdir + "/checkpoints/"
print ("Loading graph from {}".format(load_checkpoint_dir))

batch_size = int(opt.bsize)

# Evaulation
checkpoint_file = tf.train.latest_checkpoint(load_checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=True,
      log_device_placement=False)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        attention_map = graph.get_operation_by_name("attention_map").outputs[0]

        # Tensors we want to evaluate
        scores = graph.get_operation_by_name("output/scores").outputs[0]

        # Generate batches for one epoch
        batches = utils.batch_iter(list(zip(x_test, x_test_map)), batch_size, 1)

        # Collect the prediction scores here
        pred_scores = []

        for batch in batches:
            x_test_batch, x_test_batch_map = zip(*batch)
            batch_scores = sess.run(scores, {input_x: x_test_batch, attention_map: x_test_batch_map})
            pred_scores = np.concatenate([pred_scores, batch_scores])

predictions = np.array([1 if score > 0.05 else 0 for score in pred_scores])

mae = mean_absolute_error(y_test_prob, pred_scores)
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)

print("Test Statistics\nTotal number of test examples: {}\n".format(len(y_test)))
print("Probabilistic Task:\t\tMean Absolute Error {:g}".format(mae))
print("Binary Classification Task:\tF1 Score {:g}\tAccuracy {:g}".format(f1, accuracy))
print("\t\t\t\tPrecision {:g}\tRecall {:g}".format(precision, recall))