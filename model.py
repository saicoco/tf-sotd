import tensorflow as tf
import numpy as np

from tensorflow.contrib import slim

tf.app.flags.DEFINE_integer('text_scale', 512, '')

from nets import resnet_v1

FLAGS = tf.app.flags.FLAGS


def unpool(inputs):
    return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1]*2,  tf.shape(inputs)[2]*2])


def mean_image_subtraction(images, means=[123.68, 116.78, 103.94]):
    '''
    image normalization
    :param images:
    :param means:
    :return:
    '''
    num_channels = images.get_shape().as_list()[-1]
    if len(means) != num_channels:
      raise ValueError('len(means) must match the number of channels')
    channels = tf.split(axis=3, num_or_size_splits=num_channels, value=images)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=3, values=channels)


def model(images, weight_decay=1e-5, is_training=True):
    '''
    define the model, we use slim's implemention of resnet
    '''
    images = mean_image_subtraction(images)

    with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=weight_decay)):
        logits, end_points = resnet_v1.resnet_v1_50(images, is_training=is_training, scope='resnet_v1_50')

    with tf.variable_scope('feature_fusion', values=[end_points.values]):
        batch_norm_params = {
        'decay': 0.997,
        'epsilon': 1e-5,
        'scale': True,
        'is_training': is_training
        }
        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(weight_decay)):
            f = [end_points['pool5'], end_points['pool4'],
                 end_points['pool3'], end_points['pool2'],
                 ]
            for i in range(4):
                print('Shape of f_{} {}'.format(i, f[i].shape))
            g = [None, None, None, None]
            h = [None, None, None, None]
            num_outputs = [None, 256, 128, 64, 32]
            with tf.variable_scope('decoder'):
                for i in range(4):
                    if i == 0:
                        h[i] = f[i]
                    else:
                        c1_1 = slim.conv2d(tf.concat([g[i-1], f[i]], axis=-1), num_outputs[i], 1)
                        h[i] = slim.conv2d(c1_1, num_outputs[i], 3)
                    if i <= 2:
                        g[i] = unpool(h[i])
                    else:
                        g[i] = slim.conv2d(h[i], num_outputs[i], 3)

                    print('Shape of h_{} {}, g_{} {}'.format(i, h[i].shape, i, g[i].shape))

            # here we use a slightly different way for regression part,
            # we first use a sigmoid to limit the regression range, and also
            # this is do with the angle map

            with tf.variable_scope('sotd_head'):
                # unpool_1
                conv_1x1 = slim.conv2d(g[-1], 16, 1, 1, activation_fn=tf.nn.relu, normalizer_fn=None)
                conv_3x3 = slim.conv2d(conv_1x1, 16, 3, 1, activation_fn=tf.nn.relu, normalizer_fn=None)
                unpool1 = unpool(conv_3x3)

                # unpool_2
                conv_1x1_2 = slim.conv2d(unpool1, 8, 1, 1, activation_fn=tf.nn.relu, normalizer_fn=None)
                conv_3x3_2 = slim.conv2d(conv_1x1_2, 8, 3, 1, activation_fn=tf.nn.relu, normalizer_fn=None)
                unpool2 = unpool(conv_3x3_2)
                sotd_map = slim.conv2d(unpool2, 3, 1, activation_fn=tf.nn.softmax, normalizer_fn=None)

    return sotd_map

def dice_coefficient_sotd(y_true_cls, y_pred_cls,
                     training_mask):
    '''
    dice loss
    :param y_true_cls:
    :param y_pred_cls:
    :param training_mask:
    :return:
    '''
    eps = 1e-5
    intersection = tf.reduce_sum(tf.multiply(tf.multiply(y_true_cls, y_pred_cls), training_mask))
    union = tf.reduce_sum(tf.multiply(y_true_cls, training_mask)) + tf.reduce_sum(tf.multiply(y_pred_cls, training_mask)) + eps
    loss = 1. - (2 * intersection / union)
    tf.summary.scalar('sotd_dice_loss', loss)
    return loss

