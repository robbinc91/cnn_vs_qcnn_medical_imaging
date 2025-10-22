#quaternionconv3D.py

import tensorflow as tf
from tensorflow.keras import layers

class QuaternionConv3D(layers.Layer):
    """3D Quaternion Convolutional Layer - Fixed to return 5D tensor"""
        
    def __init__(self, filters, kernel_size, strides=1, padding='same', **kwargs):
        super(QuaternionConv3D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        
    def build(self, input_shape):
        input_channels = input_shape[-1] // 4
            
        # Initialize quaternion kernels
        self.kernel_r = self.add_weight(
            name='kernel_r',
            shape=(*self.kernel_size, input_channels, self.filters),
            initializer='he_uniform',
            trainable=True
        )
        self.kernel_i = self.add_weight(
            name='kernel_i',
            shape=(*self.kernel_size, input_channels, self.filters),
            initializer='he_uniform',
            trainable=True
        )
        self.kernel_j = self.add_weight(
            name='kernel_j',
            shape=(*self.kernel_size, input_channels, self.filters),
            initializer='he_uniform',
            trainable=True
        )
        self.kernel_k = self.add_weight(
            name='kernel_k',
            shape=(*self.kernel_size, input_channels, self.filters),
            initializer='he_uniform',
            trainable=True
        )
            
        self.bias = self.add_weight(
            name='bias',
            shape=(self.filters * 4,),
            initializer='zeros',
            trainable=True
        )
            
    def call(self, inputs):
        # Split input into quaternion components
        input_r = inputs[..., 0::4]
        input_i = inputs[..., 1::4]
        input_j = inputs[..., 2::4]
        input_k = inputs[..., 3::4]
            
        # Hamilton product for quaternion convolution
        conv_r = (tf.nn.conv3d(input_r, self.kernel_r, strides=[1, self.strides, self.strides, self.strides, 1], padding=self.padding.upper()) -
                tf.nn.conv3d(input_i, self.kernel_i, strides=[1, self.strides, self.strides, self.strides, 1], padding=self.padding.upper()) -
                tf.nn.conv3d(input_j, self.kernel_j, strides=[1, self.strides, self.strides, self.strides, 1], padding=self.padding.upper()) -
                tf.nn.conv3d(input_k, self.kernel_k, strides=[1, self.strides, self.strides, self.strides, 1], padding=self.padding.upper()))
            
        conv_i = (tf.nn.conv3d(input_r, self.kernel_i, strides=[1, self.strides, self.strides, self.strides, 1], padding=self.padding.upper()) +
                tf.nn.conv3d(input_i, self.kernel_r, strides=[1, self.strides, self.strides, self.strides, 1], padding=self.padding.upper()) +
                tf.nn.conv3d(input_j, self.kernel_k, strides=[1, self.strides, self.strides, self.strides, 1], padding=self.padding.upper()) -
                tf.nn.conv3d(input_k, self.kernel_j, strides=[1, self.strides, self.strides, self.strides, 1], padding=self.padding.upper()))
            
        conv_j = (tf.nn.conv3d(input_r, self.kernel_j, strides=[1, self.strides, self.strides, self.strides, 1], padding=self.padding.upper()) -
                tf.nn.conv3d(input_i, self.kernel_k, strides=[1, self.strides, self.strides, self.strides, 1], padding=self.padding.upper()) +
                tf.nn.conv3d(input_j, self.kernel_r, strides=[1, self.strides, self.strides, self.strides, 1], padding=self.padding.upper()) +
                tf.nn.conv3d(input_k, self.kernel_i, strides=[1, self.strides, self.strides, self.strides, 1], padding=self.padding.upper()))
            
        conv_k = (tf.nn.conv3d(input_r, self.kernel_k, strides=[1, self.strides, self.strides, self.strides, 1], padding=self.padding.upper()) +
                tf.nn.conv3d(input_i, self.kernel_j, strides=[1, self.strides, self.strides, self.strides, 1], padding=self.padding.upper()) -
                tf.nn.conv3d(input_j, self.kernel_i, strides=[1, self.strides, self.strides, self.strides, 1], padding=self.padding.upper()) +
                tf.nn.conv3d(input_k, self.kernel_r, strides=[1, self.strides, self.strides, self.strides, 1], padding=self.padding.upper()))
            
        # Stack components along the channel dimension (axis=-1) to get 5D tensor
        output = tf.concat([conv_r, conv_i, conv_j, conv_k], axis=-1)
        output = output + self.bias
            
        return output