import numpy as np
from train import train_with_generators
from utils.evals import evaluate_model, plot_comparison, plot_training_history
from utils.medical_image_generator_3d import create_data_generators


train_gen, val_gen, test_gen = create_data_generators("/content/drive/MyDrive/datatrain", batch_size=4)
history_unet, model_unet = train_with_generators(train_gen, val_gen, modelname='unet', num_classes=1, filters=32, epochs=100)
model_unet.save('/content/drive/MyDrive/Colab Notebooks/model_unet.keras')
history_qcnn, model_qcnn = train_with_generators(train_gen, val_gen, modelname='qcnn', num_classes=1, filters=16, epochs=100)
model_qcnn.save('/content/drive/MyDrive/Colab Notebooks/model_qcnn.keras')

val_images = []
val_masks = []
for i in range(len(test_gen)):
    batch_images, batch_masks = test_gen[i]
    val_images.append(batch_images)
    val_masks.append(batch_masks)

val_images = np.concatenate(val_images, axis=0)
val_masks = np.concatenate(val_masks, axis=0)

unet_dsc_scores, unet_predictions = evaluate_model(model_unet, val_images, val_masks, "UNet")
qcnn_dsc_scores, qcnn_predictions = evaluate_model(model_qcnn, val_images, val_masks, "QCNN")

# Plot training history
plot_training_history(history_unet, history_qcnn)

# Plot some example predictions for visual comparison
plot_comparison(val_images, val_masks, unet_predictions, qcnn_predictions, num_samples=min(3, len(val_images)))