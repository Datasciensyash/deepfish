from typing import Optional, Tuple

import dxcam
import numpy as np
from PIL import ImageGrab


class OutputController:
    def __init__(self):
        self._screen_controller = dxcam.create()
        self._screen_size = self._screen_controller.grab().shape

    @property
    def width(self) -> int:
        return self._screen_size[1]

    @property
    def height(self) -> int:
        return self._screen_size[0]

    def _get_bounding_box(self, position: Tuple[int, int], size: int) -> Tuple[int, int, int, int]:
        return (
            max(position[0] - size, 0),
            max(position[1] - size, 0),
            min(position[0] + size, self._screen_size[1]),
            min(position[1] + size, self._screen_size[0]),
        )

    def get_screenshot(
        self,
        position: Optional[Tuple[int, int]] = None,
        size: int = 32,
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

        # Try to use fast backend, but it's unstable
        output = self._screen_controller.grab(region)
        if output is not None:
            return np.array(output)

        # Use slow ImageGrab backend, if dxcam backend unable to take screenshot.
        return np.array(ImageGrab.grab(region))
