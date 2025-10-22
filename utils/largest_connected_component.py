from scipy import ndimage
import numpy as np

def largest_connected_component_scipy(binary_volume):
    """
    Finds the largest connected component in a 3D binary volume.

    Args:
        binary_volume (numpy.ndarray): A 3D binary array (0s and 1s).

    Returns:
        numpy.ndarray: A new 3D binary array containing only the largest connected component.
    """
    # Label connected components in the volume.
    # `structure` defines the connectivity (26-connectivity for 3D).
    labeled_volume, num_features = ndimage.label(binary_volume)

    # If no components found, return an empty array of the same shape
    if num_features == 0:
        return np.zeros_like(binary_volume)

    # Count the size of each component (using the label indices)
    component_sizes = np.bincount(labeled_volume.ravel())

    # Ignore the background (label 0) by setting its size to -1
    component_sizes[0] = -1

    # Find the label of the largest component
    largest_component_label = np.argmax(component_sizes)

    # Create a new volume where only the largest component is 1
    largest_component_volume = (labeled_volume == largest_component_label)

    return largest_component_volume.astype(binary_volume.dtype)