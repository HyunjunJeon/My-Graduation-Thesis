import input_data
mnist = input_data.read_data_sets()


import tensorflow as tf
from tensorflow.python.ops import rnn_cell, rnn
import numpy as np
import scipy.io

# Parameters
learning_rate = 0.001
training_iters = 500000
batch_size = 10
display_step = 100

# Data folder를 보시오 / data folder 내

# Network Parameters
n_input = 4 # data input %csv 파일 개수 (data_1 to data_28)
n_steps = 480 # timesteps  %각 csv 파일 내 개수
n_hidden = 64 # hidden layer num of features
n_classes = 1 # total classes (0-9 digits)

# 여기까지 결론은 28개의 input data file에서 각 파일 당 50개의 수치값들이 Input으로 들어가며,
# 70개의 hidden unit을 거쳐 12개의 class로 분류된다 라고 해석됨

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
istate = tf.placeholder("float", [None, 2*n_hidden]) #state & cell => 2x n_hidden %
y = tf.placeholder("float", [None, n_classes])

# Define weights
weights = {
    'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])), # Hidden layer weights
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'hidden': tf.Variable(tf.random_normal([n_hidden])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def RNN(_X, _istate, _weights, _biases):

    # input shape: (batch_size, n_steps, n_input)
    _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size % 0 1 2 => 1 0 2로 열 순서를 바꾸겠다.
    # Reshape to prepare input to hidden activation
    _X = tf.reshape(_X, [-1, n_input]) # (n_steps*batch_size, n_input) % 데이터를 쫙 피겠다 1행으로
    # Linear activation
    _X = tf.matmul(_X, _weights['hidden']) + _biases['hidden']

    # Define a lstm cell with tensorflow
    lstm_cell = tf.nn.rnn_cell.LSTMCell(n_hidden, state_is_tuple=True)
    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    _X = tf.split(0, n_steps, _X) # n_steps * (batch_size, n_hidden)

    # Get lstm cell output
    outputs, states = tf.nn.rnn(lstm_cell, _X, dtype=tf.float32)

    # Linear activation
    # Get inner loop last output
    return tf.matmul(outputs[-1], _weights['out']) + _biases['out']

pred = RNN(x, istate, weights, biases)

# Define loss and optimizer
cost = tf.sqrt(tf.reduce_mean(tf.square(pred-y))) # Softmax loss
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer

# Evaluate model

# Initializing the variables
init = tf.global_variables_initializer()
saver = tf.train.Saver()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    save_path = saver.restore(sess, "./model_param.ckpt")
    # Calculate accuracy for 256 mnist test images
    test_len = len(mnist.test.datas)
    test_data = mnist.test.datas[:test_len].reshape((-1, n_steps, n_input))
    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy:", sess.run(cost, feed_dict={x: test_data, y: test_label,
                                             istate: np.zeros((test_len, 2*n_hidden))}))

    scipy.io.savemat("train_y_out.mat",
                     {"train_y_out": sess.run(pred, feed_dict={x: mnist.train._datas, y: mnist.train._labels,
                                             istate: np.zeros((len(mnist.train._labels), 2*n_hidden))})})
    scipy.io.savemat("test_y_out.mat",
                     {"test_y_out": sess.run(pred, feed_dict={x: test_data, y: test_label,
                                             istate: np.zeros((test_len, 2*n_hidden))})})


