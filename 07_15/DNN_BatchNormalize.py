'''
라이브러리, train/test 데이터, 레이블 데이터 로드
'''
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)



'''
신경망 모델 구성
'''
# 손글씨 이미지(mnist dataset)는 28x28픽셀, 784개의 특성값으로 정함.
X = tf.placeholder(tf.float32, [None, 784])

# 결과는 0~9의 10가지 분류
Y = tf.placeholder(tf.float32, [None, 10])

# 신경망 레이어 구성 (784-256-256-10)
W1 = tf.Variable(tf.random_normal([784, 256], stddev=0.01))
L1 = tf.nn.relu(tf.matmul(X, W1))
tf.layers.batch_normalization(L1, training=False)

W2 = tf.Variable(tf.random_normal([256, 256], stddev=0.01))
L2 = tf.nn.relu(tf.matmul(L1, W2))
tf.layers.batch_normalization(L2, training=False)

W3 = tf.Variable(tf.random_normal([256, 10], stddev=0.01))
model = tf.matmul(L2, W3)


'''
최적화
'''
# cross-entropy로 error를 계산, AdamOptimizer로 최적화 수행
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
opt = tf.train.AdamOptimizer(pow(10, -3)).minimize(cost)


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

        _, cost_val = sess.run([opt, cost], feed_dict={X: batch_xs, Y: batch_ys})
        total_cost += cost_val

    print('Epoch:', '%04d' % (epoch + 1), 'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))

print('최적화 완료!')

'''
테스트 데이터를 이용한 최종 식별 결과 확인
'''
# model 로 예측한 값과 실제 레이블인 Y의 값을 비교합니다.
# tf.argmax 함수를 이용해 예측한 값에서 가장 큰 값을 예측한 레이블이라고 평가합니다.
# 예) [0.1 0 0 0.7 0 0.2 0 0 0 0] -> 3

is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도:', sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))


