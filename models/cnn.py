#cnn.py

import tensorflow as tf
import numpy as np
import os
import glob
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import nibabel as nib
from tqdm import tqdm
import warnings

from models.convblock3d import conv_block_3d
warnings.filterwarnings('ignore')


def build_unet_3d(input_shape=(128, 128, 128, 1), num_classes=1, filters=64):
    """Build 3D U-Net Model"""
    inputs = layers.Input(shape=input_shape)
    
    # Encoder
    c1 = conv_block_3d(inputs, filters)
    p1 = layers.MaxPooling3D((2, 2, 2))(c1)
    
    c2 = conv_block_3d(p1, filters * 2)
    p2 = layers.MaxPooling3D((2, 2, 2))(c2)
    
    c3 = conv_block_3d(p2, filters * 4)
    p3 = layers.MaxPooling3D((2, 2, 2))(c3)
    
    c4 = conv_block_3d(p3, filters * 8)
    p4 = layers.MaxPooling3D((2, 2, 2))(c4)
    
    # Bridge
    b1 = conv_block_3d(p4, filters * 16)
    
    # Decoder
    u1 = layers.Conv3DTranspose(filters * 8, (2, 2, 2), strides=(2, 2, 2), padding='same')(b1)
    u1 = layers.concatenate([u1, c4])
    c5 = conv_block_3d(u1, filters * 8)
    
    u2 = layers.Conv3DTranspose(filters * 4, (2, 2, 2), strides=(2, 2, 2), padding='same')(c5)
    u2 = layers.concatenate([u2, c3])
    c6 = conv_block_3d(u2, filters * 4)
    
    u3 = layers.Conv3DTranspose(filters * 2, (2, 2, 2), strides=(2, 2, 2), padding='same')(c6)
    u3 = layers.concatenate([u3, c2])
    c7 = conv_block_3d(u3, filters * 2)
    
    u4 = layers.Conv3DTranspose(filters, (2, 2, 2), strides=(2, 2, 2), padding='same')(c7)
    u4 = layers.concatenate([u4, c1])
    c8 = conv_block_3d(u4, filters)
    
    # Output
    outputs = layers.Conv3D(num_classes, (1, 1, 1), activation='sigmoid')(c8)
    
    model = Model(inputs, outputs, name='UNet3D')
    return model