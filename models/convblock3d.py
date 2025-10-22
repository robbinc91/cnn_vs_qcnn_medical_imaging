#convblock3d.py

import tensorflow as tf
from tensorflow.keras import layers, Model
from models.instancenormalization import InstanceNormalization

# =============================================================================
# 1. STANDARD 3D UNET IMPLEMENTATION
# =============================================================================
        
def conv_block_3d(inputs, filters, use_instance_norm=True):
    """3D Convolutional Block with Instance Normalization"""
    x = layers.Conv3D(filters, (3, 3, 3), padding='same', kernel_initializer='he_normal')(inputs)
    if use_instance_norm:
        x = InstanceNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv3D(filters, (3, 3, 3), padding='same', kernel_initializer='he_normal')(x)
    if use_instance_norm:
        x = InstanceNormalization()(x)
    x = layers.Activation('relu')(x)
    return x
