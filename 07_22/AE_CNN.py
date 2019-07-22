import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

lr = pow(10, -3)

'''
Conv. and Max Pooling (Encoder)
'''

# 1*28*28
inputs_ = tf.placeholder(tf.float32, (None, 28, 28, 1), name='inputs')
targets_ = tf.placeholder(tf.float32, (None, 28, 28, 1), name='targets')

# 32*28*28
conv1 = tf.layers.conv2d(inputs=inputs_, filters=32, kernel_size=(3, 3),
                         padding='SAME', activation=tf.nn.relu)
# 32*14*14
maxpool1 = tf.layers.max_pooling2d(conv1, pool_size=(2, 2), strides=(2, 2),
                                   padding='SAME')

# 32*14*14
conv2 = tf.layers.conv2d(inputs=maxpool1, filters=32, kernel_size=(3, 3),
                         padding='SAME', activation=tf.nn.relu)
# 32*7*7
maxpool2 = tf.layers.max_pooling2d(conv2, pool_size=(2, 2), strides=(2, 2),
                                   padding='SAME')


# 16*7*7
conv3 = tf.layers.conv2d(inputs=maxpool2, filters=16, kernel_size=(3, 3),
                         padding='SAME', activation=tf.nn.relu)
# 16*4*4
maxpool3 = tf.layers.max_pooling2d(conv3, pool_size=(2, 2), strides=(2, 2),
                                   padding='SAME')
### maxpool3 == Latent Layer ###


'''
Upsample and Conv. (Decoder)
'''

# result -> 16*7*7
upsample1 = tf.image.resize_images(maxpool3, size=(7, 7),
                                   method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

# result -> 16*7*7
conv4 = tf.layers.conv2d(inputs=upsample1, filters=16, kernel_size=(3, 3),
                         padding='SAME', activation=tf.nn.relu)

# result -> 16*14*14
upsample2 = tf.image.resize_images(conv4, size=(14, 14),
                                   method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

# result -> 32*14*14
conv5 = tf.layers.conv2d(inputs=upsample2, filters=32, kernel_size=(3, 3),
                         padding='SAME', activation=tf.nn.relu)

# result -> 32*28*28
upsample3 = tf.image.resize_images(conv5, size=(28, 28),
                                   method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# result -> 32*28*28
conv6 = tf.layers.conv2d(inputs=upsample3, filters=32, kernel_size=(3, 3),
                         padding='SAME', activation=tf.nn.relu)

# result -> 1*28*28
logits = tf.layers.conv2d(inputs=conv6, filters=1, kernel_size=(3, 3),
                          padding='SAME', activation=None)

# Pass logits through sigmoid to get reconstructed image
decoded = tf.nn.sigmoid(logits)
# Pass logits through sigmoid and calculate the cross-entropy loss
loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets_, logits=logits)
# Get cost and define the optimizer
cost = tf.reduce_mean(loss)
opt = tf.train.AdamOptimizer(lr).minimize(cost)

sess = tf.Session()
epochs = 20
batch_size = 100

# Set's how much noise we're adding to the MNIST images
noise_factor = 0.5
sess.run(tf.global_variables_initializer())
for e in range(epochs):
    # print(e)
    for ii in range(mnist.train.num_examples // batch_size):
        batch = mnist.train.next_batch(batch_size)
        # Get images from the batch
        imgs = batch[0].reshape((-1, 28, 28, 1))
        # Add random noise to the input images
        noisy_imgs = imgs + noise_factor * np.random.randn(*imgs.shape)
        # Clip the images to be between 0 and 1
        noisy_imgs = np.clip(noisy_imgs, 0., 1.)
        # Noisy images as inputs, original images as targets
        batch_cost, _ = sess.run([cost, opt], feed_dict={inputs_: noisy_imgs, targets_: imgs})

    print("Epoch: {}/{}...".format(e + 1, epochs), "Training loss: {:.4f}".format(batch_cost))


sample_size = 10
tmp = mnist.test.images[:sample_size]
images = tmp.reshape((-1, 28, 28, 1))
result = sess.run(decoded, feed_dict={inputs_: images})
fig, ax = plt.subplots(2, sample_size, figsize=(sample_size, 2))

for i in range(sample_size):
    ax[0][i].set_axis_off()
    ax[1][i].set_axis_off()

    ax[0][i].imshow(np.reshape(images[i], [28, 28]))
    ax[1][i].imshow(np.reshape(result[i], [28, 28]))

plt.show()