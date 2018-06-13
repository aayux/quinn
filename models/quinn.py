import tensorflow as tf

class Quinn(object):
    def __init__(self, max_length, vocab_size, embedding_dims, 
                 hidden_layers=128, l2_lambda=0.0):
        """
        Constructor, Quinn (CWI-NN) Model
        
        Args
        ----
        max_length: maximum sequence length
        vocab_size: size of vocabulary
        embedding_dims: embedding dimension
        l2_reg_lambda: (not added into graph) L2 regularization strength

        """
        
        print ("Loading model ...")

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, max_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None], name='input_y')
        self.attention_map = tf.placeholder(tf.int32, [None, max_length], name='attention_map')

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
                                        dtype=tf.float32)
        
        with tf.name_scope('soft_attention'):
            self.out = self.soft_attention(tf.concat(self.out, axis=2), hidden_layers * 2)
        
        with tf.name_scope('output'):
            w = tf.Variable(tf.truncated_normal([hidden_layers * 2, 1], stddev=0.1), name='weight')
            b = tf.Variable(tf.constant(0., shape=[1]), name='bias')
            self.out = tf.nn.sigmoid(tf.nn.xw_plus_b(self.out, w, b))
            self.scores = tf.squeeze(self.out, name='scores')
            
        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.losses.mean_squared_error(labels=self.input_y, predictions=self.scores)
            self.loss = tf.reduce_mean(losses) + l2_lambda * l2_loss

        # Mean Absolute Error
        with tf.name_scope("mae"):
            self.mae, self.update_op = tf.metrics.mean_absolute_error(labels=self.input_y, predictions=self.scores)

    def soft_attention(self, att_input, hidden_layers):
        
        w = tf.Variable(tf.random_normal([hidden_layers, 1], stddev=0.1), name='weight')
        b = tf.Variable(tf.random_normal([1], stddev=0.1), name='bias')

        # shape of s: [batch_size, seq_length]
        s = tf.squeeze(tf.tanh(tf.tensordot(att_input, w, axes=1) + b))
        
        # Attention mask for annotator defined context        
        alpha = self.masked_attention(tf.nn.softmax(s, name='alpha'))
        
        # output shape: [batch_size, hidden_layers]
        out = tf.reduce_sum(att_input * tf.expand_dims(alpha, -1), 1)
        
        return out
    
    def masked_attention(self, alpha, epsilon=1e-5):
        mask = epsilon * tf.cast(tf.equal(self.attention_map,
                                          tf.zeros_like(self.attention_map)), dtype=tf.float32)
        return tf.multiply(mask, alpha)
