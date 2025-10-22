import tensorflow as tf
from metrics.metrics import combined_loss, dice_coefficient
from models.cnn import build_unet_3d
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from models.qcnn import build_qcnn_3d

def train_with_generators(train_gen, val_gen, modelname='unet', num_classes=1, filters=8, input_shape=(80, 80, 96, 1), epochs=100):
    """Example of how to integrate with the previous model training code"""

    if modelname == 'unet':

        model_unet = build_unet_3d(input_shape=input_shape, num_classes=num_classes, filters=filters)
        print('unet model built')
        model_unet.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss=combined_loss,
            metrics=['accuracy', dice_coefficient]
        )

        checkpoint_unet = ModelCheckpoint(
            '/content/drive/MyDrive/Colab Notebooks/best_unet_.h5',
            monitor='val_dice_coefficient',  # Monitor DSC for medical imaging
            mode='max',  # We want to maximize DSC
            save_best_only=True,
            verbose=1
        )

        history_unet = model_unet.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10),
                checkpoint_unet
            ]
        )

        return history_unet, model_unet

    model_qcnn = build_qcnn_3d(input_shape=input_shape, num_classes=num_classes, filters=filters)
    print('qcnn model built')

    model_qcnn.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=combined_loss,
        metrics=['accuracy', dice_coefficient]
    )

    checkpoint_qcnn = ModelCheckpoint(
        '/content/drive/MyDrive/Colab Notebooks/best_qcnn_.h5',
        monitor='val_dice_coefficient',  # Monitor DSC for medical imaging
        mode='max',  # We want to maximize DSC
        save_best_only=True,
        verbose=1
    )

    # Train with generators

    history_qcnn = model_qcnn.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10),
            checkpoint_qcnn
        ]
    )
    return history_qcnn, model_qcnn