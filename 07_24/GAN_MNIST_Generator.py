import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

# # HP 정의
# 옵션 설정
total_epoch = 100
batch_size = 100
lr = 2*pow(10, -4)

# 신경망 레이어 구성 옵션
n_hidden = 256
n_input = 28*28
n_noise = 128       # 생성기(G)의 입력 값으로 사용할 noise의 크기

# 신경망 모델 구성
# 비지도 학습, Y 없음
X = tf.placeholder(tf.float32, [None, n_input])
Z = tf.placeholder(tf.float32, [None, n_noise])

# G 신경망에 사용할 변수
G_W1 = tf.Variable(tf.random_normal([n_noise, n_hidden], stddev=0.01))
G_b1 = tf.Variable(tf.zeros([n_hidden]))
G_W2 = tf.Variable(tf.random_normal([n_hidden, n_input], stddev=0.01))
G_b2 = tf.Variable(tf.zeros([n_input]))

# D 신경망에 사용할 변수
D_W1 = tf.Variable(tf.random_normal([n_input, n_hidden], stddev=0.01))
D_b1 = tf.Variable(tf.zeros([n_hidden]))
D_W2 = tf.Variable(tf.random_normal([n_hidden, 1], stddev=0.01))
D_b2 = tf.Variable(tf.zeros([1]))

# G 신경망 구성
def generator(noise_z):
    hidden = tf.nn.relu(tf.matmul(noise_z, G_W1) + G_b1)
    output = tf.nn.sigmoid(tf.matmul(hidden, G_W2) + G_b2)
    return output

# D 신경망 구성
def discriminator(inputs):
    hidden = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
    output = tf.nn.sigmoid(tf.matmul(hidden, D_W2) + D_b2)
    return output

# 랜덤 노이즈
def get_noise(batch_size, n_noise):
    return np.random.normal(size=(batch_size, n_noise))

# 노이즈를 이용해 랜덤한 이미지 생성(fake image)
G = generator(Z)

# 노이즈를 이용해 생성한 이미지가 진짜 이미지인지 판별한 값
D_gene = discriminator(G)

# 진짜 이미지 이용해 판별한 값 구학
D_real = discriminator(X)

loss_D = tf.reduce_mean(tf.log(D_real) + tf.log(1-D_gene))
loss_G = tf.reduce_mean(tf.log(D_gene))

D_var_list = [D_W1, D_b1, D_W2, D_b2]
# D_var_list = tf.get
G_var_list = [G_W1, G_b1, G_W2, G_b2]

# 최적화 하려는 loss_D와 loss_G에 음수 부호 부가
train_D = tf.train.AdadeltaOptimizer(lr).minimize(-loss_D, var_list=D_var_list)
train_G = tf.train.AdadeltaOptimizer(lr).minimize(-loss_G, var_list=G_var_list)


# 신경망 모델 학습
sess = tf.Session()
sess.run(tf.global_variables_initializer())
total_batch = int(mnist.train.num_examples / batch_size)
loss_val_D, loss_val_G = 0, 0

for epoch in range(total_epoch):
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        noise = get_noise(batch_size, n_noise)

        # 판별기와 성성기 신경망 각각을 학습
        _, loss_val_D = sess.run([train_D, loss_D], feed_dict={X: batch_xs, Z: noise})
        _, loss_val_G = sess.run([train_G, loss_G], feed_dict={Z: noise})

    print("Epoch:", '%04d' % epoch, 'D loss: {:.4}'.format(loss_val_D), 'G loss:{:.4}'.format(loss_val_G))

    # 학습 과정을 보기위해 주기적으로 이미지 생성 & 저장
    if epoch == 0 or (epoch + 1) % 5 == 0:
        sample_size = 10
        noise = get_noise(sample_size, n_noise)
        samples = sess.run(G, feed_dict={Z: noise})

        fig, ax = plt.subplots(1, sample_size, figsize=(sample_size, 1))
        for i in range(sample_size):
            ax[i].set_axis_off()
            ax[i].imshow(np.reshape(samples[i], (28, 28)))

        plt.savefig('samples/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
        plt.close(fig)

print('최적화 완료')