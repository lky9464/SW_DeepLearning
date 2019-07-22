import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

lr = pow(10, -2)
tr_epoch = 20
batch_size = 100

# 입력 -> 784
# 전체 레이어 구성 -> (input)784 - (hidden)612 - 256 - 128 - (latent)64 - 128 - 256 - 612(hidden) - 784(output)
# 신경망 레이어 구성 옵션
n_input = 28*28             # 입력 값의 크기 - 이미지 픽셀 수

n_hidden = 612              # hidden layer의 뉴런 갯수
n_hidden2 = 256
n_hidden3 = 128
n_latent = 64


# No label
X = tf.placeholder(tf.float32, [None, n_input])
global_step = tf.Variable(0, trainable=False, name='global_step')

# encoder layer와 decoder layer의 가중치와 편향 변수를 설정
# 다음과 같이 이어지는 layer를 구성하기 위한 값들
# input -> encode -> decode -> output

# Encode part
W_encode = tf.Variable(tf.random_normal([n_input, n_hidden]))       # 784, 612
b_encode = tf.Variable(tf.random_normal([n_hidden]))                # 612
encoder = tf.nn.sigmoid(tf.add(tf.matmul(X, W_encode), b_encode))   # X -> 1st Layer

W_encode2 = tf.Variable(tf.random_normal([n_hidden, n_hidden2]))    # 612, 256
b_encode2 = tf.Variable(tf.random_normal([n_hidden2]))              # 612
encoder2 = tf.nn.sigmoid(tf.add(tf.matmul(encoder, W_encode2), b_encode2))      # 1st Hidden Layer -> 2nd Hidden Layer

W_encode3 = tf.Variable(tf.random_normal([n_hidden2, n_hidden3]))   # 256, 128
b_encode3 = tf.Variable(tf.random_normal([n_hidden3]))              # 256
encoder3 = tf.nn.sigmoid(tf.add(tf.matmul(encoder2, W_encode3), b_encode3))     # 2nd Hidden Layer -> 3rd Hidden Layer

W_encode4 = tf.Variable(tf.random_normal([n_hidden3, n_latent]))      # 128, 64
b_encode4 = tf.Variable(tf.random_normal([n_latent]))                 # 128
encoder4 = tf.nn.sigmoid(tf.add(tf.matmul(encoder3, W_encode4), b_encode4))     # 3rd Hidden Layer -> Latent Layer


# Decode part, Reverse
W_decode = tf.Variable(tf.random_normal([n_latent, n_hidden3]))
b_decode = tf.Variable(tf.random_normal([n_hidden3]))
decoder = tf.nn.sigmoid(tf.add(tf.matmul(encoder4, W_decode), b_decode))

W_decode2 = tf.Variable(tf.random_normal([n_hidden3, n_hidden2]))
b_decode2 = tf.Variable(tf.random_normal([n_hidden2]))
decoder2 = tf.nn.sigmoid(tf.add(tf.matmul(decoder, W_decode2), b_decode2))

W_decode3 = tf.Variable(tf.random_normal([n_hidden2, n_hidden]))
b_decode3 = tf.Variable(tf.random_normal([n_hidden]))
decoder3 = tf.nn.sigmoid(tf.add(tf.matmul(decoder2, W_decode3), b_decode3))

W_decode4 = tf.Variable(tf.random_normal([n_hidden, n_input]))
b_decode4 = tf.Variable(tf.random_normal([n_input]))
decoder4 = tf.nn.sigmoid(tf.add(tf.matmul(decoder3, W_decode4), b_decode4))

# decoder4 -> 최종 모델



# decoder는 input과 최대한 같은 결과를 내야 하므로, decoding한 결과를 평가하기 위해 입력 값인 X값을 평가를 위한
# 실측 결과 값으로하여 decodeer와의 차이를 손실값으로 설정
cost = tf.reduce_mean(tf.pow(X - decoder4, 2))
opt = tf.train.RMSPropOptimizer(lr).minimize(cost, global_step=global_step)

init = tf.global_variables_initializer()
sess = tf.Session()
saver = tf.train.Saver(tf.global_variables())
sess.run(init)
total_batch = int(mnist.train.num_examples/batch_size)

# make check point
ckpt = tf.train.get_checkpoint_state('./models')

if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    saver.restore(sess, ckpt.model_checkpoint_path)

else:       # 'models' 디렉토리 없으면 새로 Session 시작
    sess.run(tf.global_variables_initializer())


for epoch in range(tr_epoch):
    total_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        _, cost_val = sess.run([opt, cost], feed_dict={X: batch_xs})
        total_cost += cost_val
    print('Epoch: ', '%04d' % (epoch + 1), 'Avg cost =', '{: .4f}'.format(total_cost/total_batch))

print('최적화 완료')

saver.save(sess, './models/DeepAE_Model_Reuse_tfSaver.ckpt', global_step=global_step)