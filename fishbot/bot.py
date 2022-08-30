import logging
import time
from pathlib import Path

from fishbot.constants import (LOOT_OFFSET_PIXELS, LOOTING_GAP_TIME,
                               MAX_BOBBER_LIFETIME_SEC, MAX_LOOT_ITEMS,
                               SLEEP_NOT_ACTIVE_SECONDS, MAX_START_WAIT_TIME)
from fishbot.controllers.input import InputController
from fishbot.controllers.output import OutputController
from fishbot.controllers.window import WindowController
from fishbot.match import TemplateMatchingModel
from fishbot.model.classifynet import ClassificationNet
from fishbot.model.detectnet import BobberDetector
from fishbot.utils.detection_mask import create_detection_mask


class FishingBot:
    def __init__(
        self,
        fishing_skill_key: str,
        bobber_checkpoint: Path,
        splash_checkpoint: Path,
        splash_threshold: float = 0.5,
    ):
        self._fishing_skill_key = fishing_skill_key
        self._splash_threshold = splash_threshold

        self._input_controller = InputController()
        self._screen_controller = OutputController()
        self._window_controller = WindowController()

        self._bobber_detector = BobberDetector.from_checkpoint(
            checkpoint_path=bobber_checkpoint,
            mask=create_detection_mask(
                self._screen_controller.height, self._screen_controller.width
            ),
        )
        self._splash_classifier = ClassificationNet.from_checkpoint(splash_checkpoint)
        self._loot_matching = TemplateMatchingModel(self._template_directory / "loot_skull.png")

    @property
    def _template_directory(self) -> Path:
        return Path(__file__).parent / "templates"

    def _wait_and_prepare_start(self) -> None:

        # Wait until World of Warcraft is opened
        start_time = time.time()
        while True:
            if time.time() - start_time > MAX_START_WAIT_TIME:
                raise RuntimeError("Max World of Warcraft opening time exceeded. Please restart.")

            if self._window_controller.is_warcraft_active():
                break

        # Change third-person to first-person
        self._input_controller.tap_end_key()
        for _ in range(5):
            self._input_controller.tap_home_key()
            time.sleep(0.1)

    def run(self) -> None:
        is_fishing = False

        # Starting preparation
        self._wait_and_prepare_start()

        # Main event loop
        while True:

            # Check active window. Don't do anything, if warcraft window
            # is not active.
            if not self._window_controller.is_warcraft_active():
                is_fishing = False
                time.sleep(SLEEP_NOT_ACTIVE_SECONDS)
                continue

            # Try to start fishing
            if not is_fishing:
                # Move the cursor away from bobber
                self._input_controller.move(0, 0)

                # Tap fishing skill key to start fishing
                self._input_controller.tap_key(self._fishing_skill_key)

                # Save fishing start time
                fishing_start_time = time.time()

                # Wait for bobber to show up
                time.sleep(1.5)

                # Take screenshot to detect bobber
                screenshot = self._screen_controller.get_screenshot()

                # Detect position of bobber
                bobber_position = self._bobber_detector.inference(screenshot, threshold=0.2)

                # If bobber is not found
                if bobber_position is None:
                    continue

                is_fishing = True

            # Get screenshot near current fishing position
            screenshot = self._screen_controller.get_screenshot(bobber_position)  # noqa
            splash_probability = self._splash_classifier.inference(screenshot)

            # Increase probability by a time factor
            total_fishing_time = time.time() - fishing_start_time  # noqa
            time_factor = (total_fishing_time / MAX_BOBBER_LIFETIME_SEC) * 0.2
            splash_probability = splash_probability + time_factor

            # If splash detected, than click on bobber and loot
            if splash_probability > self._splash_threshold:
                # Move mouse to current position of bobber
                self._input_controller.move(*bobber_position)

                # Right-click on bobber
                self._input_controller.click(*bobber_position, right_click=True)

                # Make screenshot to find looting skull icon
                screenshot = self._screen_controller.get_screenshot()

                # Find looting skull location
                skull_location = self._loot_matching.match(screenshot)

                # Compute location of loot
                loot_x, loot_y = [
                    skull_location[0],
                    skull_location[1] + LOOT_OFFSET_PIXELS,
                ]

                # Move mouse to loot position
                self._input_controller.move(loot_x, loot_y)

                # Loot all items
                for _ in range(MAX_LOOT_ITEMS):
                    # Sleep for some time
                    time.sleep(LOOTING_GAP_TIME)

                    # Loot item
                    self._input_controller.click(loot_x, loot_y, right_click=True)

            # If time of fishing is exceeded
            elif time.time() - fishing_start_time > MAX_BOBBER_LIFETIME_SEC:
                # Exit fishing state
                is_fishing = False
