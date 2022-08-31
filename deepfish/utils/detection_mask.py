import numpy as np


def create_detection_mask(height: int = 1080, width: int = 1980) -> np.array:
    """
    Create gaussian-like detection mask for bobber. More likely, that
        bobber will be in the center of screen.

    Args:
        height: Height of screen.
        width: Width of screen.

    Returns: Mask, 2d numpy array in range [0, 1]
    """
    grid_x, grid_y = np.mgrid[-2 : 2 : height * 1j, -2 : 2 : width * 1j]
    x_0, y_0 = -0.2, 0
    sigma_x, sigma_y = 1.0, 1.2
    mask = np.exp(
        -(((grid_x - x_0) ** 2 / (2 * sigma_x**2)) + ((grid_y - y_0) ** 2 / (2 * sigma_y**2)))
    )
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    return mask
