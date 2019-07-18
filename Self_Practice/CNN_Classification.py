import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)


class CNN_MNIST:
    def __init__(self):
        self.X = tf.placeholder(tf.float32, [None, 28, 28, 1])
        self.Y = tf.placeholder(tf.float32, [None, 10])
        self.keep_prob = tf.placeholder(tf.float32)
        self.is_training = tf.placeholder(tf.bool)


    def CNN_DropOut_MNIST(self, lr, f_size, act_func, kp_train):
        L1 = tf.layers.conv2d(self.X, 32, [f_size, f_size], activation=act_func)
        L1 = tf.layers.max_pooling2d(L1, [2, 2], [2, 2])
        L1 = tf.layers.dropout(L1, kp_train, self.is_training)

        L2 = tf.layers.conv2d(L1, 32, [f_size, f_size], activation=act_func)
        L2 = tf.layers.max_pooling2d(L2, [2, 2], [2, 2])
        L2 = tf.layers.dropout(L2, kp_train, self.is_training)

        L3 = tf.layers.flatten(L2)
        L3 = tf.layers.dense(L3, 256, activation=act_func)
        L3 = tf.layers.dropout(L3, kp_train, self.is_training)

        model = tf.layers.dense(L3, 10, activation=None)

        # Optimizer
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=self.Y))
        opt = tf.train.AdamOptimizer(lr).minimize(cost)

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
                _, cost_val = sess.run([opt, cost], feed_dict={self.X: batch_xs, self.Y: batch_ys, self.is_training: True})
                total_cost += cost_val

            print('Epoch : ', '%04d' % (epoch + 1), 'Avg cost = ', '{:.3f}'.format(total_cost / total_batch))

        print('최적화 완료')

        is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(self.Y, 1))
        accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
        print('정확도 : ', sess.run(accuracy, feed_dict={self.X: mnist.test.images.reshape(-1, 28, 28, 1), self.Y: mnist.test.labels,
                                                      self.is_training: False}))


    def CNN_BatchNormalize(self, lr, f_size, act_func):
        L1 = tf.layers.conv2d(self.X, 32, [f_size, f_size], activation=act_func)
        L1 = tf.layers.max_pooling2d(L1, [2, 2], [2, 2])
        tf.layers.batch_normalization(L1, training=False)

        L2 = tf.layers.conv2d(L1, 32, [f_size, f_size], activation=act_func)
        L2 = tf.layers.max_pooling2d(L2, [2, 2], [2, 2])
        tf.layers.batch_normalization(L2, training=False)

        L3 = tf.layers.flatten(L2)
        L3 = tf.layers.dense(L3, 256, activation=act_func)
        tf.layers.batch_normalization(L3, training=False)

        model = tf.layers.dense(L3, 10, activation=None)

        # Optimizer
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=self.Y))
        opt = tf.train.AdamOptimizer(lr).minimize(cost)

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
                _, cost_val = sess.run([opt, cost],
                                       feed_dict={self.X: batch_xs, self.Y: batch_ys, self.is_training: True})
                total_cost += cost_val

            print('Epoch : ', '%04d' % (epoch + 1), 'Avg cost = ', '{:.3f}'.format(total_cost / total_batch))

        print('최적화 완료')

        is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(self.Y, 1))
        accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
        print('정확도 : ',
              sess.run(accuracy, feed_dict={self.X: mnist.test.images.reshape(-1, 28, 28, 1), self.Y: mnist.test.labels,
                                            self.is_training: False}))
