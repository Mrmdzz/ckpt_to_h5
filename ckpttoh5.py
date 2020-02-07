# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 20:46:31 2020

@author: zhangzheng
"""

import os

import tensorflow as tf
from keras import Input, Model, layers
from keras import backend as K
from keras.engine import get_source_inputs
from keras.layers import Activation, SeparableConv2D, BatchNormalization, Dropout
from keras.layers import AveragePooling2D, MaxPooling2D, Add, Concatenate
from keras.layers import Convolution2D, GlobalAveragePooling2D, Dense
from keras.layers import ZeroPadding2D, GlobalMaxPooling2D, Cropping2D
from keras.utils import get_file, Progbar
from keras.utils.conv_utils import convert_kernel

import efficientnet.tfkeras as efn
def get_channel_axis():
    return 1 if K.image_data_format() == 'channels_first' else -1


def preprocess(image, size):
    with tf.Session():
        x = preprocess_tf(image, size).eval()

    return x


def preprocess_tf(image, size=224, central_fraction=0.875):
    """Used to train the weights
    From:
    https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/inception_preprocessing.py
    """
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.central_crop(image, central_fraction=central_fraction)

    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, [size, size], align_corners=False)
    image = tf.squeeze(image, [0])

    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)

    return image


def load_weights_from_tf_checkpoint(model, checkpoint_file, background_label):
    print('Load weights from tensorflow checkpoint')
    progbar = Progbar(target=len(model.layers))

    reader = tf.compat.v1.train.NewCheckpointReader(checkpoint_file)
    for index, layer in enumerate(model.layers):
        progbar.update(current=index)

        if isinstance(layer, layers.convolutional.SeparableConv2D):
            depthwise = reader.get_tensor('{}/depthwise_weights'.format(layer.name))
            pointwise = reader.get_tensor('{}/pointwise_weights'.format(layer.name))

            if K.image_data_format() == 'channels_first':
                depthwise = convert_kernel(depthwise)
                pointwise = convert_kernel(pointwise)

            layer.set_weights([depthwise, pointwise])
        elif isinstance(layer, layers.convolutional.Convolution2D):
            weights = reader.get_tensor('{}/weights'.format(layer.name))

            if K.image_data_format() == 'channels_first':
                weights = convert_kernel(weights)

            layer.set_weights([weights])
        elif isinstance(layer, layers.BatchNormalization):
            beta = reader.get_tensor('{}/beta'.format(layer.name))
            gamma = reader.get_tensor('{}/gamma'.format(layer.name))
            moving_mean = reader.get_tensor('{}/moving_mean'.format(layer.name))
            moving_variance = reader.get_tensor('{}/moving_variance'.format(layer.name))

            layer.set_weights([gamma, beta, moving_mean, moving_variance])
        elif isinstance(layer, layers.Dense):
            weights = reader.get_tensor('{}/weights'.format(layer.name))
            biases = reader.get_tensor('{}/biases'.format(layer.name))

            if background_label:
                layer.set_weights([weights, biases])
            else:
                layer.set_weights([weights[:, 1:], biases[1:]])


def load_pretrained_weights(model, fname, origin, md5_hash, background_label=False, cache_dir=None):
    """Download and convert tensorflow checkpoints"""

    if cache_dir is None:
        cache_dir = os.path.expanduser(os.path.join('~', '.keras', 'models'))

    weight_path = os.path.join(cache_dir, '{}_{}_{}.h5'.format(model.name, md5_hash, K.image_data_format()))

    if os.path.exists(weight_path):
        model.load_weights(weight_path)
    else:
        #path = get_file(fname, origin=origin, extract=True, md5_hash=md5_hash, cache_dir=cache_dir)
        path='yourmodelpath'
        checkpoint_file = os.path.join(path, 'model.ckpt')
        print(checkpoint_file)
        load_weights_from_tf_checkpoint(model, checkpoint_file, background_label)

        model.save_weights(weight_path)
model=efn.EfficientNetB2(input_shape=(128,256,3), weights=None, include_top=False)
load_pretrained_weights(model, 'efficientnet-b2', origin='https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/advprop/efficientnet-b2.tar.gz', md5_hash='6fdf8c24a374f0b5fcd8211af623dcd3', background_label=False, cache_dir='C:\\Users\\zhangzheng\\.keras\\models\\datasets\\')




