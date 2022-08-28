from typing import Optional, Tuple

import dxcam
import numpy as np


class OutputController:
    def __init__(self):
        self._screen_controller = dxcam.create()
        self._screen_size = self._screen_controller.grab().shape

    def _get_bounding_box(
        self, position: Tuple[int, int], size: int
    ) -> Tuple[int, int, int, int]:
        return (
            max(position[0] - size, 0),
            max(position[1] - size, 0),
            min(position[0] + size, self._screen_size[0]),
            min(position[1] + size, self._screen_size[1]),
        )

    def get_screenshot(
        self,
        position: Optional[Tuple[int, int]] = None,
        size: int = 64,
    ) -> np.ndarray:
        """
        Get screenshot.

        Args:
            position: Center of capturing bbox, optional argument.
            size: Half of size of capturing bbox,
                if size = 64, than screenshot size will be (64 * 2, 64 * 2) pixels.

        Returns: Screenshot in form of numpy array.
        """
        region = None
        if position is not None:
            region = self._get_bounding_box(position, size)
        return self._screen_controller.grab(region)
