import numpy as np
from metrics import dice_coefficient
from utils.largest_connected_component import largest_connected_component_scipy
import matplotlib.pyplot as plt

def evaluate_model(model, test_images, test_masks, model_name):
    """Evaluate model performance"""
    print("Evaluating {model_name}...")

    # Predict on test set
    predictions = model.predict(test_images, verbose=0)
    predictions[predictions < 0.5] = 0
    predictions[predictions >= 0.5] = 1
    # Calculate DSC for each sample
    dsc_scores = []
    for i in range(len(test_images)):
        pred = largest_connected_component_scipy(predictions[i])
        dsc = dice_coefficient(test_masks[i], pred)
        dsc_scores.append(dsc.numpy())
    mean_dsc = np.mean(dsc_scores)
    std_dsc = np.std(dsc_scores)
    print("{model_name} - Mean DSC: {mean_dsc:.4f} ± {std_dsc:.4f}")
    return dsc_scores, predictions

def plot_comparison(original_images, true_masks, unet_preds, qcnn_preds, num_samples=3):

    fig, axes = plt.subplots(num_samples, 4, figsize=(15, 4*num_samples))

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    slice_idx = original_images[0].shape[2] // 2  # Middle slice

    for i in range(num_samples):
        # Original image
        axes[i, 0].imshow(original_images[i, :, :, slice_idx, 0], cmap='gray')
        axes[i, 0].set_title(f'Sample {i+1} - Original')
        axes[i, 0].axis('off')

        # True mask
        axes[i, 1].imshow(true_masks[i, :, :, slice_idx, 0], cmap='hot')
        axes[i, 1].set_title('True Mask')
        axes[i, 1].axis('off')

        # UNet prediction

        _unet_preds = largest_connected_component_scipy(unet_preds[i])
        axes[i, 2].imshow(_unet_preds[:, :, slice_idx, 0], cmap='hot')
        axes[i, 2].set_title(f'UNet Prediction\\nDSC: {dice_coefficient(true_masks[i], unet_preds[i]):.3f}')
        axes[i, 2].axis('off')

        # QCNN prediction
        _qcnn_preds = largest_connected_component_scipy(qcnn_preds[i])
        axes[i, 3].imshow(_qcnn_preds[:, :, slice_idx, 0], cmap='hot')
        axes[i, 3].set_title(f'QCNN Prediction\\nDSC: {dice_coefficient(true_masks[i], qcnn_preds[i]):.3f}')
        axes[i, 3].axis('off')

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_training_history(history_unet, history_qcnn):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss
    axes[0, 0].plot(history_unet.history['loss'], label='UNet Train')
    axes[0, 0].plot(history_unet.history['val_loss'], label='UNet Val')
    axes[0, 0].plot(history_qcnn.history['loss'], label='QCNN Train')
    axes[0, 0].plot(history_qcnn.history['val_loss'], label='QCNN Val')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Dice Coefficient
    axes[0, 1].plot(history_unet.history['dice_coefficient'], label='UNet Train')
    axes[0, 1].plot(history_unet.history['val_dice_coefficient'], label='UNet Val')
    axes[0, 1].plot(history_qcnn.history['dice_coefficient'], label='QCNN Train')
    axes[0, 1].plot(history_qcnn.history['val_dice_coefficient'], label='QCNN Val')
    axes[0, 1].set_title('Dice Coefficient')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('DSC')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Accuracy
    axes[1, 0].plot(history_unet.history['accuracy'], label='UNet Train')
    axes[1, 0].plot(history_unet.history['val_accuracy'], label='UNet Val')
    axes[1, 0].plot(history_qcnn.history['accuracy'], label='QCNN Train')
    axes[1, 0].plot(history_qcnn.history['val_accuracy'], label='QCNN Val')
    axes[1, 0].set_title('Accuracy')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Learning Rate
    if 'lr' in history_unet.history:
        axes[1, 1].plot(history_unet.history['lr'], label='UNet')
        axes[1, 1].plot(history_qcnn.history['lr'], label='QCNN')
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('LR')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()