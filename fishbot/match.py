from typing import Optional, Tuple

import cv2
from pathlib import Path

import numpy as np


class TemplateMatchingModel:
    """
    Simple template-matching model, which uses algorithms from cv2.

    Args:
        template_path: Path to template
    """
    def __init__(self, template_path: Path) -> None:
        self._template = cv2.imread(template_path)
        self._template = cv2.cvtColor(self._template, cv2.COLOR_BGR2RGB)
        self._size_x, self._size_y = self._template.shape[:2]

    @staticmethod
    def _mse_loss(image: np.ndarray, template: np.ndarray) -> float:
        return np.float(np.mean((np.mean(image, 2) - np.mean(template, 2)) ** 2))

    def match(self, image: np.ndarray, threshold: float = np.inf) -> Optional[Tuple[int, int]]:
        """
        Match image with template and get coordinates of best match.

        Args:
            image: Image, where template needs to be found.
            threshold: Threshold for mse loss.

        Returns: coordinates of best match (or None, if best match MSE loss is higher than threshold).
        """
        result = cv2.matchTemplate(image, self._template, cv2.TM_CCOEFF_NORMED)
        _, _, _, (min_x, min_y) = cv2.minMaxLoc(result)
        center_position = (int(min_x + self._size_x / 2), int(min_y + self._size_y / 2))

        if threshold is np.inf:
            return center_position

        image_region = image[min_x: min_x + self._size_y, min_y: min_y + self._size_y],
        mse_loss_value = self._mse_loss(image=image_region, template=self._template) # noqa

        if mse_loss_value > threshold:
            return None

        return center_position
