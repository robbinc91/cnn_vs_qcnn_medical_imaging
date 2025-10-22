import os
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
import nibabel as nib
from sklearn.model_selection import train_test_split

class MedicalDataGenerator3D(Sequence):
    """Data generator for 3D medical images and labels"""

    def __init__(self, image_paths, label_paths, batch_size=2, target_shape=(80, 80, 96),
                 shuffle=True, augment=False):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.batch_size = batch_size
        self.target_shape = target_shape
        self.shuffle = shuffle
        self.augment = augment
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        batch_image_paths = self.image_paths[index * self.batch_size:(index + 1) * self.batch_size]
        batch_label_paths = self.label_paths[index * self.batch_size:(index + 1) * self.batch_size]

        batch_images = []
        batch_labels = []

        for img_path, label_path in zip(batch_image_paths, batch_label_paths):
            # Load 3D NIfTI images
            image = self.load_and_preprocess_image(img_path)
            label = self.load_and_preprocess_label(label_path)

            if self.augment:
                image, label = self.augment_3d_data(image, label)

            batch_images.append(image)
            batch_labels.append(label)

        return np.array(batch_images), np.array(batch_labels)

    def load_and_preprocess_image(self, path):
        """Load and preprocess 3D medical image"""
        img = nib.load(path)
        data = img.get_fdata()

        # Normalize to [0, 1]
        data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)

        # Add channel dimension
        data = np.expand_dims(data, axis=-1)

        return data

    def load_and_preprocess_label(self, path):
        """Load and preprocess 3D label mask"""
        img = nib.load(path)
        data = img.get_fdata()

        # Binarize labels and ensure proper shape
        data = (data > 0).astype(np.float32)

        # Add channel dimension
        data = np.expand_dims(data, axis=-1)

        return data

    def augment_3d_data(self, image, label):
        """Simple 3D data augmentation"""
        # Random flipping
        if np.random.random() > 0.5:
            axis = np.random.randint(0, 3)
            image = np.flip(image, axis=axis)
            label = np.flip(label, axis=axis)

        # Random rotation (90 degree increments)
        if np.random.random() > 0.5:
            k = np.random.randint(1, 4)
            axes = (0, 1)  # Rotate in axial plane
            image = np.rot90(image, k=k, axes=axes)
            label = np.rot90(label, k=k, axes=axes)

        return image, label

    def on_epoch_end(self):
        if self.shuffle:
            indices = np.arange(len(self.image_paths))
            np.random.shuffle(indices)
            self.image_paths = [self.image_paths[i] for i in indices]
            self.label_paths = [self.label_paths[i] for i in indices]

def load_data_from_drive(drive_path):
    """Load and pair 3D MRI images with their segmentation labels"""

    # Get all files in the directory
    all_files = os.listdir(drive_path)

    # Initialize lists
    image_paths = []
    label_paths = []

    # Pattern to extract the number from filenames like a31_n4bfc... and a31-seg
    pattern = re.compile(r'a(\\d+)')

    # Separate image files and label files
    image_files = [f for f in all_files if f.endswith('_n4bfc_mni_cropped_80x80x96.nii.gz')]
    label_files = [f for f in all_files if f.endswith('-seg.nii.gz')]

    print("Found {len(image_files)} image files")
    print("Found {len(label_files)} label files")

    # Create mapping for easy pairing
    image_dict = {}
    label_dict = {}

    # Process image files
    for file in image_files:
        match = pattern.search(file)
        if match:
            file_id = match.group(1)
            image_dict[file_id] = os.path.join(drive_path, file)

    # Process label files
    for file in label_files:
        match = pattern.search(file)
        if match:
            file_id = match.group(1)
            label_dict[file_id] = os.path.join(drive_path, file)

    # Pair images with labels
    paired_image_paths = []
    paired_label_paths = []

    for file_id in image_dict:
        if file_id in label_dict:
            paired_image_paths.append(image_dict[file_id])
            paired_label_paths.append(label_dict[file_id])
        else:
            print("Warning: No label found for image {file_id}")

    print("Successfully paired {len(paired_image_paths)} image-label pairs")

    return paired_image_paths, paired_label_paths

def create_data_generators(drive_path, batch_size=2, validation_split=0.1, test_split=0.2):
    """Create training and validation data generators for 3D data"""

    # Load and pair data
    image_paths, label_paths = load_data_from_drive(drive_path)

    # Split data
    train_img, val_img, train_lbl, val_lbl = train_test_split(
        image_paths, label_paths, test_size=validation_split, random_state=42
    )

    train_img, test_img, train_lbl, test_lbl = train_test_split(
        train_img, train_lbl, test_size=test_split, random_state=42
    )


    print(train_img, val_img)

    print("Training samples: {len(train_img)}")
    print("Validation samples: {len(val_img)}")
    print("Test samples: {len(test_img)}")

    # Create generators
    train_generator = MedicalDataGenerator3D(
        train_img, train_lbl, batch_size=batch_size, shuffle=True, augment=True
    )

    val_generator = MedicalDataGenerator3D(
        val_img, val_lbl, batch_size=batch_size, shuffle=False, augment=False
    )

    test_generator = MedicalDataGenerator3D(
        test_img, test_lbl, batch_size=batch_size, shuffle=False, augment=False
    )

    return train_generator, val_generator, test_generator