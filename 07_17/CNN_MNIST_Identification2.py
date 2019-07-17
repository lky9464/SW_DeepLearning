import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

# Data def.
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
is_training = tf.placeholder(tf.bool)

'''
LAYERS
'''

# First Layer
L1 = tf.layers.conv2d(X, 32, [3, 3], activation=tf.nn.relu)
L1 = tf.layers.max_pooling2d(L1, [2, 2], [2, 2])
L1 = tf.layers.dropout(L1, 0.7, is_training)

# Second Layer
L2 = tf.layers.conv2d(L1, 32, [3, 3], activation=tf.nn.relu)
L2 = tf.layers.max_pooling2d(L2, [2, 2], [2, 2])
L2 = tf.layers.dropout(L2, 0.7, is_training)

# Third Layer
L3 = tf.layers.flatten(L2)
L3 = tf.layers.dense(L3, 256, activation=tf.nn.relu)
L3 = tf.layers.dropout(L3, 0.5, is_training)

# NN Model
model = tf.layers.dense(L3, 10, activation=None)

# Optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
opt = tf.train.AdamOptimizer(pow(10, -3)).minimize(cost)

# Training
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)



for epoch in range(15):
    total_cost = 0
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)

        # 이미지를 데이터를 CNN 모델을 위한 자료형태인 [?, 28, 28, 1]로 재구성
        batch_xs = batch_xs.reshape(-1, 28, 28, 1)
        _, cost_val = sess.run([opt, cost], feed_dict={X: batch_xs, Y: batch_ys, is_training: True})
        total_cost += cost_val

    print('Epoch : ', '%04d' % (epoch + 1), 'Avg cost = ', '{:.3f}'.format(total_cost/total_batch))

print('최적화 완료')

is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도 : ', sess.run(accuracy, feed_dict={X: mnist.test.images.reshape(-1, 28, 28, 1), Y: mnist.test.labels, is_training: False}))
