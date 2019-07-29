import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

# # MNIST Number Dataset
mnist = input_data.read_data_sets("mnist/mnist_fashion/", one_hot=True)

# # MNIST Fashion Dataset
# mnist = input_data.read_data_sets("mnist/data/", one_hot=True)

'''

Bidirectional RNN, DropOut and Non-DropOut

'''


# 옵션 설정
lr = pow(10, -3)
total_epoch = 30
batch_size = 128
keep_prob = 0.7

# RNN은 순서가 있는 자료를 다루므로, 한 번에 입력받는 갯수와, 총 몇단계로
# 이루어져있는 데이터를 받을지를 설정
# 가로 픽셀 수를 n_input으로, 세로 픽셀 수를 입력 단계인 n_step으로 설정
n_input = 28
n_step = 28
n_hidden = 128
n_class = 10
num_multi_rnn = 5

X = tf.placeholder(tf.float32, [None, n_step, n_input]) # n_step이 추가(CNN과 차이점)
Y = tf.placeholder(tf.float32, [None,n_class])

###
# #Without Dropout
## Two-way cell
# lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
# lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
#
# outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell, cell_bw=lstm_bw_cell, inputs=X, dtype=tf.float32)
# outputs_fw = tf.transpose(outputs[0], [1, 0, 2])
# outputs_bw = tf.transpose(outputs[1], [1, 0, 2])
# outputs_concat = tf.concat([outputs_fw[-1], outputs_bw[-1]], axis=1)

#With Dropout

## Two-way cell
lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell, output_keep_prob=keep_prob)
lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell, output_keep_prob=keep_prob)

outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell, cell_bw=lstm_bw_cell, inputs=X, dtype=tf.float32)
outputs_fw = tf.transpose(outputs[0], [1, 0, 2])
outputs_bw = tf.transpose(outputs[1], [1, 0, 2])
outputs_concat = tf.concat([outputs_fw[-1], outputs_bw[-1]], axis=1)


###

# 결과를 Y의 다음 형식과 바꿔야 하기 때문에
# outputs 의 형태를 이에 맞춰 변경해야합니다.
# outputs :  [batch_size, n_step, n_hidden]
#           ->  [n_step, batch_size, n_hidden]
#           ->  [batch_size, n_hidden]
outputs = tf.transpose(outputs, [1, 0, 2])
outputs = outputs[-1]

model = tf.layers.dense(outputs, 10)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
opt = tf.train.AdamOptimizer(lr).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

total_batch = int(mnist.train.num_examples / batch_size)

for epoch in range(total_epoch):
    total_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)

        # X 데이터를 RNN 입력 데이터에 맞게 [batch_size, n_step, n_input]형태로 변환
        batch_xs = batch_xs.reshape((batch_size, n_step, n_input))

        _, cost_val = sess.run([opt, cost],
                               feed_dict={X: batch_xs, Y: batch_ys})
        total_cost += cost_val

    print('Epoch:', '%04d' % (epoch + 1), 'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))

print('최적화 완료')

is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

test_batch_size = len(mnist.test.images)
test_xs = mnist.test.images.reshape(test_batch_size,n_step, n_input)
test_ys = mnist.test.labels

print('정확도 :', sess.run(accuracy, feed_dict={X: test_xs, Y: test_ys}))