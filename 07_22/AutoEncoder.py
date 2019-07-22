import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

lr = pow(10, -3)
tr_epoch = 20
batch_size = 100

# 신경망 레이어 구성 옵션
n_hidden = 256              # hidden layer의 뉴런 갯수
n_input = 28*28             # 입력 값의 크기 - 이미지 픽셀 수

# No label
X = tf.placeholder(tf.float32, [None, n_input])

# encoder layer와 decoder layer의 가중치와 편향 변수를 설정
# 다음과 같이 이어지는 layer를 구성하기 위한 값들
# input -> encode -> decode -> output
W_encode = tf.Variable(tf.random_normal([n_input, n_hidden]))
b_encode = tf.Variable(tf.random_normal([n_hidden]))
encoder = tf.nn.sigmoid(tf.add(tf.matmul(X, W_encode), b_encode))

W_decode = tf.Variable(tf.random_normal([n_hidden, n_input]))
b_decode = tf.Variable(tf.random_normal([n_input]))
decoder = tf.nn.sigmoid(tf.add(tf.matmul(encoder, W_decode), b_decode))



# decoder는 input과 최대한 같은 결과를 내야 하므로, decoding한 결과를 평가하기 위해 입력 값인 X값을 평가를 위한
# 실측 결과 값으로하여 decodeer와의 차이를 손실값으로 설정
cost = tf.reduce_mean(tf.pow(X - decoder, 2))
opt = tf.train.RMSPropOptimizer(lr).minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
total_batch = int(mnist.train.num_examples/batch_size)


for epoch in range(tr_epoch):
    total_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        _, cost_val = sess.run([opt, cost], feed_dict={X: batch_xs})
        total_cost += cost_val
    print('Epoch: ', '%04d' % (epoch + 1), 'Avg cost =', '{: .4f}'.format(total_cost/total_batch))

print('최적화 완료')


# 입력 값(위쪽)과 모델이 생성한 값(아래쪽)을 시각적으로 비교
sample_size = 10
samples = sess.run(decoder, feed_dict={X: mnist.test.images[:sample_size]})
fig, ax = plt.subplots(2, sample_size, figsize=(sample_size, 2))

for i in range(sample_size):
    ax[0][i].set_axis_off()
    ax[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))

    ax[1][i].set_axis_off()
    ax[1][i].imshow(np.reshape(samples[i], (28, 28)))

plt.show()