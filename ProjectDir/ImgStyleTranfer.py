import time

from IPython.display import display

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import  PIL.Image


# Use 'vgg16' Model (Tensorflow)
from SW_DL.ProjectDir.vgg16 import vgg16

# vgg16.maybe_download()


'''

Helper Funcs.

'''
# Load image func.
def load_image(filename, max_size=None):
    image = PIL.Image.open(filename)

    if max_size is not None:

        factor = max_size / np.max(image.size)
        size = np.array(image.size) * factor
        size = size.astype(int)

        image = image.resize(size, PIL.Image.LANCZOS)

    return np.float32(image)

# Save image func.
def save_image(image, filename):
    image = np.clip(image, 0.0, 255.0)
    image = image.astype(np.uint8)

    # write image file in jepg
    with open(filename, 'wb') as file:
        PIL.Image.fromarray(image).save(file, 'jpeg')

# Plot big image
def plot_image_big(image):
    image = np.clip(image, 0.0, 255.0)
    image = image.astype(np.uint8)
    display(PIL.Image.fromarray(image))

# Plot image func.
def plot_images(cont_image, st_image, mixed_image):
    # create figure with sub-plots
    fig, axes = plt.subplots(1, 3, figsize=(10, 10))
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    smooth = True

    if smooth:
        interpolation = 'sinc'
    else:
        interpolation = 'nearest'

    # Content Plot
    ax = axes.flat[0]
    ax.imshow(cont_image / 255.0, interpolation=interpolation)
    ax.set_xlabel('Content Image')

    # Mixed Plot
    ax = axes.flat[1]
    ax.imshow(mixed_image / 255.0, interpolation=interpolation)
    ax.set_xlabel('Mixed Image')

    # Style Plot
    ax = axes.flat[2]
    ax.imshow(st_image / 255.0, interpolation=interpolation)
    ax.set_xlabel('Style Image')

    # Remove ticks from all the plots
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()


'''

Loss Funcs.

'''
# Mean squared error
def mean_sqrd_error(a, b):
    return tf.reduce_mean(tf.square(a-b))

# Loss of content image
def create_cont_loss(sess, model, cont_image, layer_ids):
    feed_dict = model.create_feed_dict(image=cont_image)
    layers = model.get_layer_tensors(layer_ids)
    values = sess.run(layers, feed_dict=feed_dict)

    with model.graph.as_default():
        # list for loss values
        layer_losses = []

        for value, layer in zip(values, layers):
            value_const = tf.constant(value)
            loss = mean_sqrd_error(layer, value_const)
            layer_losses.append(loss)

        # Combine losses and take avg. value
        total_loss = tf.reduce_mean(layer_losses)

    return total_loss


# Gram matrix
def gram_matrix(tensor):
    shape = tensor.get_shape()
    num_channels = int(shape[3])
    matrix = tf.reshape(tensor, shape=[-1, num_channels])

    # make gram matrix ( pi/2 rotated mat * mat)
    gram = tf.matmul(tf.transpose(matrix), matrix)

    return gram

def create_st_loss(sess, model, st_image, layer_ids):

    feed_dict = model.create_feed_dict(image=st_image)
    layers = model.get_layer_tensors(layer_ids)

    with model.graph.as_default():
        gram_layers = [gram_matrix(layer) for layer in layers]
        values = sess.run(gram_layers, feed_dict=feed_dict)

        # list for loss values
        layer_losses = []

        for value, gram_layer in zip(values, gram_layers):
            value_const = tf.constant(value)

            loss = mean_sqrd_error(gram_layer, value_const)

            layer_losses.append(loss)

        total_loss = tf.reduce_mean(layer_losses)

    return total_loss


# Denoise loss func.
# Shifts input image by 1 pixel on x & y axis
def create_denoise_loss(model):
    loss = tf.reduce_sum(tf.abs(model.input[:, 1:, :, :] - model.input[:, :-1, :, :])) + tf.reduce_sum(tf.abs(model.input[:, :, 1:, :] - model.input[:, :, :-1, :]))
    return loss


'''

Style-Transfer Algorithm

'''
# ★Style transfer func.★
def style_transfer(cont_image, st_image, cont_layer_ids, st_layer_ids,
                   weight_cont=1.5, weight_st=10.0, weight_denoise=0.3,
                   num_iterations=120, step_size=10.0):

    # Get pre-trained vgg16 model
    model = vgg16.VGG16()

    # Create a TF Session
    sess = tf.InteractiveSession(graph=model.graph)

    # Print the names of the  content-layers
    print('Content layers :')
    print(model.get_layer_names(cont_layer_ids))
    print()

    # Print the names of the style-layers
    print('Style layers :')
    print(model.get_layer_names(st_layer_ids))
    print()

    # Create loss-func. for the content-layers and image
    loss_content = create_cont_loss(sess=sess, model=model,
                                    cont_image=cont_image,
                                    layer_ids=st_layer_ids)

    # Create the loss-func. for the style-layers and image
    loss_style = create_st_loss(sess=sess, model=model,
                                st_image=st_image,
                                layer_ids=st_layer_ids)

    # Create the loss-func. for the denoising of the mixed-image
    loss_denoise = create_denoise_loss(model)

    adj_content = tf.Variable(1e-10, name='adj_content')
    adj_style = tf.Variable(1e-10, name='adj_style')
    adj_denoise = tf.Variable(1e-10, name='adj_denoise')

    sess.run([adj_content.initializer, adj_style.initializer, adj_denoise.initializer])

    update_adj_content = adj_content.assign(1.0 / (loss_content + 1e-10))
    update_adj_style = adj_style.assign(1.0 / (loss_style + 1e-10))
    update_adj_denoise = adj_denoise.assign(1.0 / (loss_denoise + 1e-10))

    loss_combined = (weight_cont * adj_content * loss_content) + (weight_st * adj_style * loss_style) + (weight_denoise * adj_denoise * loss_denoise)

    gradient = tf.gradients(loss_combined, model.input)

    run_list = [gradient, update_adj_content, update_adj_style, update_adj_denoise]

    mixed_image = np.random.rand(*cont_image.shape) + 128

    for i in range(num_iterations):
        feed_dict = model.create_feed_dict(image=mixed_image)

        grad, adj_content_val, adj_style_val, adj_denoise_val = sess.run(run_list, feed_dict=feed_dict)

        grad = np.squeeze(grad)

        step_size_scaled = step_size / (np.std(grad) + 1e-8)

        mixed_image -= grad * step_size_scaled

        mixed_image = np.clip(mixed_image, 0.0, 255.0)

        print('. ', end="")

        # Display status once every 10 iterations, and the last
        if (i % 10 == 0) or (i == num_iterations - 1):
            print()
            print('Iteration :', i)

            # Print adjustment weights for loss-funcs.
            msg = 'Weight Adj. for Content : {0: .2e}, Style : {1: .2e}, Denoise : {2: .2e}'
            print(msg.format(adj_content_val, adj_style_val, adj_denoise_val))

            # in larger resolution
            # Plot the content, style, mixed images
            plot_images(cont_image=cont_image, st_image=st_image, mixed_image=mixed_image)

            print()
            print('Final image : ')
            plot_image_big(mixed_image)

            sess.close()

            return mixed_image

# Show Example
content_filename = './images/catImage.jpg'
content_image = load_image(content_filename, max_size=None)

style_filename = './images/style_tree.jpg'
style_image = load_image(style_filename, max_size=300)

content_layer_ids = [4]

style_layer_ids = list(range(13))
step_size = 10.0


img = style_transfer(content_image,
                     style_image,
                     content_layer_ids,
                     style_layer_ids,
                     1.5,
                     10.0,
                     0.3,
                     60,
                     10.0)


