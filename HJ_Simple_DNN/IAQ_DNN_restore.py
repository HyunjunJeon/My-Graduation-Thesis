# Deep neural network - CYP3A4
#from __future__ import division, print_function, absolute_import

import numpy
import scipy.io
import tensorflow as tf

mat_data1 = scipy.io.loadmat('train_x.mat')
Features_colum1 = mat_data1['train_x']
#Features_colum = Features_colum * 100
train_x = numpy.array(Features_colum1, dtype=numpy.float32)

mat_data2 = scipy.io.loadmat('train_y.mat')
Features_colum2 = mat_data2['train_y']
#Features_colum = Features_colum * 100
train_y = numpy.array(Features_colum2, dtype=numpy.float32)

mat_data3 = scipy.io.loadmat('test_x.mat')
Features_colum3 = mat_data3['test_x']
#Features_colum = Features_colum * 100
test_x = numpy.array(Features_colum3, dtype=numpy.float32)

mat_data4 = scipy.io.loadmat('test_y.mat')
Features_colum4 = mat_data4['test_y']
#Features_colum = Features_colum * 100
test_y = numpy.array(Features_colum4, dtype=numpy.float32)

print(len(train_x))
# Parameter
learning_rate = 0.001
training_epochs = 15000
display_step = 100

# tf input
X = tf.placeholder("float",[None, 3])
Y = tf.placeholder("float",[None,1])

# store layers weight & bias
#W1 = tf.Variable(tf.random_normal([4306, 350]))
#W2 = tf.Variable(tf.random_normal([350, 100]))
#W3 = tf.Variable(tf.random_normal([100, 50]))
#W4 = tf.Variable(tf.random_normal([50, 1]))

W1 = tf.get_variable("W1", shape=[3, 256], dtype="float", initializer=tf.contrib.layers.xavier_initializer(3, 256))
W2 = tf.get_variable("W2", shape=[256, 256], dtype="float", initializer=tf.contrib.layers.xavier_initializer(256, 256))
W3 = tf.get_variable("W3", shape=[256, 256], dtype="float", initializer=tf.contrib.layers.xavier_initializer(256, 256))
W4 = tf.get_variable("W4", shape=[256, 256], dtype="float", initializer=tf.contrib.layers.xavier_initializer(256, 256))
W5 = tf.get_variable("W5", shape=[256, 1], dtype="float", initializer=tf.contrib.layers.xavier_initializer(256, 1))

B1 = tf.Variable(tf.random_normal([256]))
B2 = tf.Variable(tf.random_normal([256]))
B3 = tf.Variable(tf.random_normal([256]))
B4 = tf.Variable(tf.random_normal([256]))
B5 = tf.Variable(tf.random_normal([1]))

## Construct model
dropout_rate = tf.placeholder('float')
_L1 = tf.nn.relu(tf.add(tf.matmul(X, W1), B1))
L1 = tf.nn.dropout(_L1, dropout_rate)
_L2 = tf.nn.relu(tf.add(tf.matmul(L1, W2), B2))
L2= tf.nn.dropout(_L2, dropout_rate)
_L3 = tf.nn.relu(tf.add(tf.matmul(L2, W3), B3))
L3= tf.nn.dropout(_L3, dropout_rate)
_L4 = tf.nn.relu(tf.add(tf.matmul(L3, W4), B4))
L4= tf.nn.dropout(_L4, dropout_rate)
hypothesis = tf.add(tf.matmul(L4,W5),B5)

# Define cost and optimizer
cost = tf.sqrt(tf.reduce_mean(tf.square(hypothesis-Y)))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()
saver = tf.train.Saver()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    save_path = saver.restore(sess, "./model_param.ckpt")
# Peformance checking
    RMSE = tf.sqrt(tf.reduce_mean(tf.square(hypothesis-Y)))
    print("RMSE_train:", sess.run(RMSE, feed_dict={X: train_x, Y: train_y, dropout_rate: 0.6}))
    print("RMSE_test:", sess.run(RMSE, feed_dict={X: test_x, Y: test_y, dropout_rate: 1}))
    answer = hypothesis
    train_pred = sess.run(answer, feed_dict={X: train_x, dropout_rate: 0.6})
    test_pred = sess.run(answer, feed_dict={X: test_x, dropout_rate: 1})
    scipy.io.savemat("train_y_out.mat", {"train_y_out":sess.run(hypothesis, feed_dict={X: train_x, Y: train_y, dropout_rate: 1})})
    scipy.io.savemat("test_y_out.mat", {"test_y_out": sess.run(hypothesis, feed_dict={X: test_x, Y: test_y, dropout_rate: 1})})

test_y1 = numpy.reshape(test_y,(1,len(test_y)))
test_pred1 = numpy.reshape(test_pred,(1,len(test_pred)))
Rsquare_test = numpy.corrcoef(test_y1, test_pred1)
print(Rsquare_test)
