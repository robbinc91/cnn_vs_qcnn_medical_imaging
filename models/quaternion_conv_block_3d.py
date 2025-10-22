#quaternion_conv_block_3d.py

from instancenormalization import InstanceNormalization
import tensorflow as tf
from tensorflow.keras import layers
from models.quaternionconv3D import QuaternionConv3D

def quaternion_conv_block_3d(inputs, filters, use_instance_norm=True):
    """Quaternion 3D Convolutional Block - Fixed to return 5D tensor"""
    # First quaternion convolution
    x = QuaternionConv3D(filters, (3, 3, 3), padding='same')(inputs)
    if use_instance_norm:
        x = InstanceNormalization()(x)
    x = layers.Activation('relu')(x)
            
    # Second quaternion convolution
    x = QuaternionConv3D(filters, (3, 3, 3), padding='same')(x)
    if use_instance_norm:
        x = InstanceNormalization()(x)
    x = layers.Activation('relu')(x)
            
    return x