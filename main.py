import tensorflow as tf
import numpy as np
import sys

flags = tf.app.flags
sys.path.append('/work/cse496dl/shared/hackathon/08')
DATA_DIR = '/work/cse496dl/shared/hackathon/08/ptbdata'
flags.DEFINE_string('save_dir', '/work/cse496dl/ebrahim31/hw4', 'directory where model graph and weights are saved')
FLAGS = flags.FLAGS
import ptb_reader

TIME_STEPS = 10
BATCH_SIZE = 20

class PTBInput(object):
    """The input data.

    Code sourced from https://github.com/tensorflow/models/blob/master/tutorials/rnn/ptb/ptb_word_lm.py
    """

    def __init__(self, data, batch_size, num_steps, k, name=None):
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        self.input_data, self.targets = ptb_reader.ptb_producer(
            data, batch_size, num_steps, k, name=name)

def loss(logits, targets):
    return tf.contrib.seq2seq.sequence_loss(
        logits,
        targets,
        tf.ones([BATCH_SIZE, TIME_STEPS], dtype=tf.float32),
        average_across_timesteps=True,
        average_across_batch=True)

def save(file_name, data):
    np.save(os.path.join(FLAGS.save_dir, file_name), data)

def main(argv):
    k = int(argv[1])
    tf.reset_default_graph()
    raw_data = ptb_reader.ptb_raw_data(DATA_DIR)
    train_data, valid_data, test_data, _ = raw_data
    train_input = PTBInput(train_data, BATCH_SIZE, TIME_STEPS, k, name="TrainInput")
    test_input = PTBInput(test_data, BATCH_SIZE, TIME_STEPS, k, name='TestInput')

    VOCAB_SIZE = 10000
    EMBEDDING_SIZE = 100

    # setup input and embedding
    embedding_matrix = tf.get_variable('embedding_matrix', dtype=tf.float32, shape=[VOCAB_SIZE, EMBEDDING_SIZE], trainable=True)
    word_embeddings = tf.nn.embedding_lookup(embedding_matrix, train_input.input_data)
    print("The output of the word embedding: " + str(word_embeddings))

    LSTM_SIZE = 200 # number of units in the LSTM layer, this number taken from a "small" language model

    lstm_cell = tf.contrib.rnn.BasicLSTMCell(LSTM_SIZE)

    # Initial state of the LSTM memory.
    initial_state = lstm_cell.zero_state(BATCH_SIZE, tf.float32)
    print("Initial state of the LSTM: " + str(initial_state))

    # setup RNN
    outputs, state = tf.nn.dynamic_rnn(lstm_cell, word_embeddings,
                                       initial_state=initial_state,
                                       dtype=tf.float32)

    logits = tf.layers.dense(outputs, VOCAB_SIZE)

    LEARNING_RATE = 1e-2

    # loss = tf.contrib.seq2seq.sequence_loss(
    #     logits,
    #     train_input.targets,
    #     tf.ones([BATCH_SIZE, TIME_STEPS], dtype=tf.float32),
    #     average_across_timesteps=True,
    #     average_across_batch=True)

    optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE) # better than Adam for RNN networks
    train_op = optimizer.minimize(loss(logits, train_input.targets))

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        # start queue runners
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=session, coord=coord)

        # retrieve some data to look at
        # examples = session.run([train_input.input_data, train_input.targets])
        # we can run the train op as usual

        train_loss = loss(logits, train_input.targets)
        test_loss = loss(logits, test_input.targets)
        train_losses = []
        test_losses = []
        for epoch in range(10):
            print('Epoch: ' + str(epoch))
            _, train_loss_val = session.run([train_op, train_loss])
            train_losses.append(train_loss_val)
            print('Training loss for epoch '+ str(epoch)+' is '+
                    str(train_loss_val))
            test_loss_val = session.run(test_loss)
            test_losses.append(test_loss_val)
            print('Testing loss for epoch: ' + str(epoch) + ' is ' +
                    str(test_loss_val))

    save('train_data', train_losses)
if __name__ == '__main__':
    tf.app.run()

