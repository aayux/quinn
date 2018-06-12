import tensorflow as tf

class Quinn(object):
    def __init__(self, max_length, vocab_size, embedding_dims, 
                 hidden_layers=128, l2_lambda=0.0):
        """
        Constructor, Quinn (CWI-NN) Model
        
        Args:
            max_length: ...
            num_classes: ...
            vocab_size: ...
            embedding_dims: ...
            l2_reg_lambda: ...
        """
        
        print ("Initialising placeholders ...")

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, max_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None], name='input_y')
        self.seq_length = tf.placeholder(tf.int32, [None], name='seq_length')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        self.word_embedding = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_dims]), 
                                          trainable=False)
        self.embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_dims])
        self.embedding_init = self.word_embedding.assign(self.embedding_placeholder)

        with tf.device('/cpu:0'):
            with tf.name_scope('embedding'):
                self.embedded_sentence = tf.nn.embedding_lookup(self.word_embedding, self.input_x)
        
        # Bidirectional-GRU Units
        with tf.name_scope('bi-directional-gru'):
            self.out, out_states = tf.nn.bidirectional_dynamic_rnn(
                                        tf.contrib.rnn.GRUCell(hidden_layers), 
                                        tf.contrib.rnn.GRUCell(hidden_layers),
                                        inputs=self.embedded_sentence, 
                                        sequence_length=self.seq_length,
                                        dtype=tf.float32)
        
        # TO DO: Attention Module here
        
        with tf.name_scope('output'):
            w = tf.Variable(tf.truncated_normal([..., 1], stddev=0.1), name='weight')
            b = tf.Variable(tf.constant(0., shape=[1]), name='bias')
            self.scores = tf.nn.xw_plus_b(self.out, w, b, name='scores')
        
        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(tf.sigmoid(self.scores), self.input_y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='accuracy')
        