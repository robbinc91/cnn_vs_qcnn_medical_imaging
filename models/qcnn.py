#qcnn.py

import tensorflow as tf

import numpy as np
from tensorflow.keras import layers
from models.instancenormalization import InstanceNormalization
from models.quaternion_conv_block_3d import quaternion_conv_block_3d

# Updated QCNN model builder with proper tensor shapes
def build_qcnn_3d(input_shape=(80, 80, 96, 1), num_classes=1, filters=16):
    """Build 3D Quaternion CNN Model with proper 5D tensors"""
    inputs = layers.Input(shape=input_shape)

    # Convert real input to quaternion representation by repeating across 4 channels
    x = layers.Lambda(lambda x: tf.tile(x, [1, 1, 1, 1, 4]))(inputs)
    print(f"After quaternion conversion: {x.shape}")  # Should be (batch, 80, 80, 96, 4)

    # Encoder path
    c1 = quaternion_conv_block_3d(x, filters)
    print(f"After c1: {c1.shape}")  # Should be (batch, 80, 80, 96, filters*4)
    p1 = layers.MaxPooling3D((2, 2, 2))(c1)
    print(f"After p1: {p1.shape}")  # Should be (batch, 40, 40, 48, filters*4)

    c2 = quaternion_conv_block_3d(p1, filters * 2)
    print(f"After c2: {c2.shape}")  # Should be (batch, 40, 40, 48, filters*8)
    p2 = layers.MaxPooling3D((2, 2, 2))(c2)
    print(f"After p2: {p2.shape}")  # Should be (batch, 20, 20, 24, filters*8)

    c3 = quaternion_conv_block_3d(p2, filters * 4)
    print(f"After c3: {c3.shape}")  # Should be (batch, 20, 20, 24, filters*16)
    p3 = layers.MaxPooling3D((2, 2, 2))(c3)
    print(f"After p3: {p3.shape}")  # Should be (batch, 10, 10, 12, filters*16)

    c4 = quaternion_conv_block_3d(p3, filters * 8)
    print(f"After c4: {c4.shape}")  # Should be (batch, 10, 10, 12, filters*32)
    p4 = layers.MaxPooling3D((2, 2, 2))(c4)
    print(f"After p4: {p4.shape}")  # Should be (batch, 5, 5, 6, filters*32)
            
    # Bridge
    b1 = quaternion_conv_block_3d(p4, filters * 16)
    print(f"After b1: {b1.shape}")  # Should be (batch, 5, 5, 6, filters*64)
            
    # Decoder path
    u1 = layers.Conv3DTranspose(filters * 8, (2, 2, 2), strides=(2, 2, 2), padding='same')(b1)
    print(f"After u1 upsample: {u1.shape}")  # Should be (batch, 10, 10, 12, filters*8)
    u1 = layers.concatenate([u1, c4])
    print(f"After u1 concat: {u1.shape}")  # Should be (batch, 10, 10, 12, filters*40)
    c5 = quaternion_conv_block_3d(u1, filters * 8)
    print(f"After c5: {c5.shape}")  # Should be (batch, 10, 10, 12, filters*32)
            
    u2 = layers.Conv3DTranspose(filters * 4, (2, 2, 2), strides=(2, 2, 2), padding='same')(c5)
    print(f"After u2 upsample: {u2.shape}")  # Should be (batch, 20, 20, 24, filters*4)
    u2 = layers.concatenate([u2, c3])
    print(f"After u2 concat: {u2.shape}")  # Should be (batch, 20, 20, 24, filters*20)
    c6 = quaternion_conv_block_3d(u2, filters * 4)
    print(f"After c6: {c6.shape}")  # Should be (batch, 20, 20, 24, filters*16)
            
    u3 = layers.Conv3DTranspose(filters * 2, (2, 2, 2), strides=(2, 2, 2), padding='same')(c6)
    print(f"After u3 upsample: {u3.shape}")  # Should be (batch, 40, 40, 48, filters*2)
    u3 = layers.concatenate([u3, c2])
    print(f"After u3 concat: {u3.shape}")  # Should be (batch, 40, 40, 48, filters*10)
    c7 = quaternion_conv_block_3d(u3, filters * 2)
    print(f"After c7: {c7.shape}")  # Should be (batch, 40, 40, 48, filters*8)
            
    u4 = layers.Conv3DTranspose(filters, (2, 2, 2), strides=(2, 2, 2), padding='same')(c7)
    print(f"After u4 upsample: {u4.shape}")  # Should be (batch, 80, 80, 96, filters)
    u4 = layers.concatenate([u4, c1])
    print(f"After u4 concat: {u4.shape}")  # Should be (batch, 80, 80, 96, filters*5)
    c8 = quaternion_conv_block_3d(u4, filters)
    print(f"After c8: {c8.shape}")  # Should be (batch, 80, 80, 96, filters*4)
            
    # Convert back to real values for output - take only the real components
    c8_real = layers.Lambda(lambda x: x[..., 0:(x.shape[-1]//4)])(c8)  # Take first quarter (real components)
    print(f"After real conversion: {c8_real.shape}")  # Should be (batch, 80, 80, 96, filters)
            
    # Final convolution to get the output
    outputs = layers.Conv3D(num_classes, (1, 1, 1), activation='sigmoid', padding='same')(c8_real)
    print(f"Final output: {outputs.shape}")  # Should be (batch, 80, 80, 96, num_classes)
            
    model = tf.keras.Model(inputs, outputs, name='QCNN3D_Fixed')
    return model
