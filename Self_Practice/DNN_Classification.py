import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)


'''
Define in 'DNN_MNIST' Class
'''
## 1. 손글씨 이미지(mnist dataset)는 28x28픽셀, 784개의 특성값으로 정함.
# X = tf.placeholder(tf.float32, [None, 784])
# keep_prob = tf.placeholder(tf.float32)

## 2. 결과는 0~9의 10가지 분류
# Y = tf.placeholder(tf.float32, [None, 10])

# 3-Layers
class DNN_MNIST:

    def __init__(self):
        self.X = tf.placeholder(tf.float32, [None, 784])
        self.Y = tf.placeholder(tf.float32, [None, 10])
        self.keep_prob = tf.placeholder(tf.float32)

    # DNN Drop-Out
    def DNN_DropOut_MNIST(self, lr, act_func, kp_train):

        # 신경망 레이어 구성 (784-256-256-10)
        L1 = tf.layers.dense(inputs=self.X, units=256, activation=act_func)
        L1 = tf.nn.dropout(L1, kp_train)

        L2 = tf.layers.dense(inputs=L1, units=256, activation=act_func)
        L2 = tf.nn.dropout(L2, kp_train)

        model = tf.layers.dense(inputs=L2, units=10)


        # return model

        '''
        최적화
        '''
        # cross-entropy로 error를 계산, AdamOptimizer로 최적화 수행
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=self.Y))
        opt = tf.train.AdamOptimizer(lr).minimize(cost)

        '''
        훈련 데이터를 신경망 모델 학습
        '''
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        batch_size = 100
        total_batch = int(mnist.train.num_examples / batch_size)

        for epoch in range(15):
            total_cost = 0

            for i in range(total_batch):
                # 텐서플로우의 mnist 모델의 next_batch 함수를 이용해
                # 지정한 크기만큼 학습할 데이터를 가져온다.
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)

                _, cost_val = sess.run([opt, cost], feed_dict={self.X: batch_xs, self.Y: batch_ys, self.keep_prob: kp_train})
                total_cost += cost_val

            print('Epoch:', '%04d' % (epoch + 1), 'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))

        print('최적화 완료!')

        is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(self.Y, 1))
        accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
        print('정확도:', sess.run(accuracy, feed_dict={self.X: mnist.test.images, self.Y: mnist.test.labels, self.keep_prob:1}))



    # DNN BatchNormalize
    def DNN_BatchNormalize_MNIST(self, lr, act_func):

        L1 = tf.layers.dense(inputs=self.X, units=256, activation=act_func)
        tf.layers.batch_normalization(L1, training=False)

        L2 = tf.layers.dense(inputs=L1, units=256, activation=act_func)
        tf.layers.batch_normalization(L2, training=False)

        model = tf.layers.dense(inputs=L2, units=10)

        # return model

        '''
        최적화
        '''
        # cross-entropy로 error를 계산, AdamOptimizer로 최적화 수행
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=self.Y))
        opt = tf.train.AdamOptimizer(lr).minimize(cost)

        '''
        훈련 데이터를 신경망 모델 학습
        '''
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        batch_size = 100
        total_batch = int(mnist.train.num_examples / batch_size)

        for epoch in range(15):
            total_cost = 0

            for i in range(total_batch):
                # 텐서플로우의 mnist 모델의 next_batch 함수를 이용해
                # 지정한 크기만큼 학습할 데이터를 가져온다.
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)

                _, cost_val = sess.run([opt, cost], feed_dict={self.X: batch_xs, self.Y: batch_ys})
                total_cost += cost_val

            print('Epoch:', '%04d' % (epoch + 1), 'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))

        print('최적화 완료!')

        '''
        테스트 데이터를 이용한 최종 식별 결과 확인
        '''
        # model 로 예측한 값과 실제 레이블인 Y의 값을 비교합니다.
        # tf.argmax 함수를 이용해 예측한 값에서 가장 큰 값을 예측한 레이블이라고 평가합니다.
        # 예) [0.1 0 0 0.7 0 0.2 0 0 0 0] -> 3

        is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(self.Y, 1))
        accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
        print('정확도:', sess.run(accuracy, feed_dict={self.X: mnist.test.images, self.Y: mnist.test.labels}))

