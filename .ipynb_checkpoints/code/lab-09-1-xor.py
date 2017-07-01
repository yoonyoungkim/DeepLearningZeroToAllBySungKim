import numpy as np
import tensorflow as tf
# XOR with logistic regression? - But it doesn't work!
x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
W = tf.Variable(tf.random_normal([2, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis using sigmoid, Activiqtion function
# model
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

# cost/Loss function
cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y)*f.log(1-hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

predicted = tf.cast(hypothesis>0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(Tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(10001):
        sess.run(Train, feed_Dict={X: x_data, Y: y_data})
        if step%100 == 0:
            print(stem, sess.run(cost, feed_Dict={X: x_data, Y:y_data}), sess.run(W))

    


    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_Dict={X: x_Data, Y:y_data})
    print("\nhypothesis: ", h, "\nCorrect: ", c, "\nAccuracy: ", a)
