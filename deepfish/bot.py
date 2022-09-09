import logging
import time
from pathlib import Path

from deepfish.constants import CHECKPOINTS_DIR
from deepfish.constants import (BOBBER_DETECTOR_CHECKPOINT_NAME,
                                FISHING_WAIT_TIME, LOOT_OFFSET_PIXELS,
                                LOOTING_GAP_TIME, MAX_BOBBER_LIFETIME_SEC,
                                MAX_LOOT_ITEMS, MAX_START_WAIT_TIME,
                                SLEEP_NOT_ACTIVE_SECONDS,
                                SPLASH_DETECTOR_CHECKPOINT_NAME, BOBBER_SHOW_UP_SECONDS)
from deepfish.controllers.input import InputController
from deepfish.controllers.output import OutputController
from deepfish.controllers.window import WindowController
from deepfish.logging import create_logger
from deepfish.match import TemplateMatchingModel
from deepfish.model.classifynet import ClassificationNet
from deepfish.model.detectnet import BobberDetector
from deepfish.utils.detection_mask import create_detection_mask


class FishingBot:
    def __init__(
        self,
        fishing_skill_key: str,
        splash_threshold: float = 0.5,
        need_logging: bool = True
    ):
        self._logger = create_logger("FishingBot", need_logging=need_logging)
        self._logger.info("Logger successfully initialized...")

        self._fishing_skill_key = fishing_skill_key
        self._splash_threshold = splash_threshold

        self._logger.info("Initialization of input, output and window controllers...")
        self._input_controller = InputController()
        self._screen_controller = OutputController()
        self._window_controller = WindowController()

        self._logger.info("Initialization of BobberDetector...")
        self._bobber_detector = BobberDetector.from_checkpoint(
            checkpoint_path=CHECKPOINTS_DIR / BOBBER_DETECTOR_CHECKPOINT_NAME,
            mask=create_detection_mask(
                self._screen_controller.height, self._screen_controller.width
            ),
        )
        self._logger.info("Initialization of ClassificationNet...")
        self._splash_classifier = ClassificationNet.from_checkpoint(
            CHECKPOINTS_DIR / SPLASH_DETECTOR_CHECKPOINT_NAME
        )

        self._logger.info("Initialization of TemplateMatchingModel for loot skull matching...")
        self._loot_matching = TemplateMatchingModel(self._template_directory / "loot_skull.png")

        self._logger.info("FishingBot was successfully initialized")

    @property
    def _template_directory(self) -> Path:
        return Path(__file__).parent / "templates"

    def _wait_and_prepare_start(self) -> None:
        self._logger.info("Start preparation procedure...")

        # Wait until World of Warcraft is opened
        start_time = time.time()
        self._logger.info("Waiting user to open World of Warcraft...")
        while True:
            if time.time() - start_time > MAX_START_WAIT_TIME:
                self._logger.info("Max World of Warcraft window opening time exceeded...")
                raise RuntimeError("Max World of Warcraft opening time exceeded. Please restart.")

            if self._window_controller.is_warcraft_active():
                self._logger.info("World of Warcraft window is found active...")
                break

        # Change third-person to first-person
        self._logger.info("Trying to setup first-person perspective...")
        self._input_controller.tap_end_key()
        for _ in range(5):
            self._input_controller.tap_home_key()
            time.sleep(0.1)

        self._logger.info("Start preparation procedure is done. Ready to fishing.")

    def run(self) -> None:
        is_fishing = False

        # Starting preparation
        self._wait_and_prepare_start()

        # Main event loop
        while True:

            # Check active window. Don't do anything, if warcraft window
            # is not active.
            if not self._window_controller.is_warcraft_active():
                self._logger.info("World of Warcraft window is not active.")
                if is_fishing:
                    self._logger.info("Stopping fishing...")
                    is_fishing = False

                self._logger.info(f"Sleeping for {SLEEP_NOT_ACTIVE_SECONDS} seconds...")
                time.sleep(SLEEP_NOT_ACTIVE_SECONDS)
                continue

            # Try to start fishing
            if not is_fishing:
                # Sleep for some time
                self._logger.info("Trying to start fishing...")
                time.sleep(FISHING_WAIT_TIME)

                # Move the cursor away from bobber
                self._logger.info("Moving cursor to (0, 0)...")
                self._input_controller.move(0, 0)

                # Tap fishing skill key to start fishing
                self._logger.info(f"Tapping fishing skill key ({self._fishing_skill_key})...")
                self._input_controller.tap_key(self._fishing_skill_key)

                # Save fishing start time
                self._logger.info(f"Fishing started...")
                fishing_start_time = time.time()

                # Wait for bobber to show up
                self._logger.info(f"Waiting bobber to show up for {BOBBER_SHOW_UP_SECONDS} seconds...")
                time.sleep(BOBBER_SHOW_UP_SECONDS)

                # Take screenshot to detect bobber
                self._logger.info(f"Making screenshot to detect bobber...")
                screenshot = self._screen_controller.get_screenshot()

                # Detect position of bobber
                self._logger.info(f"Inference of BobberDetector on screenshot...")
                bobber_position = self._bobber_detector.inference(screenshot, threshold=0.2)

                # If bobber is not found
                if bobber_position is None:
                    self._logger.info(f"Bobber is not found. Stopping fishing.")
                    continue

                self._logger.info(f"Bobber found at {bobber_position}. Starting splash detection stage...")
                is_fishing = True

            # Get screenshot near current fishing position
            self._logger.info(f"Making screenshot for splash classification...")
            screenshot = self._screen_controller.get_screenshot(bobber_position)  # noqa

            # Get splash probability by splash classifier inference
            self._logger.info(f"Splash Classifier inference...")
            splash_probability = self._splash_classifier.inference(screenshot)
            self._logger.info(f"Raw splash probability: {round(splash_probability, 2)}")

            # Increase probability by a time factor
            total_fishing_time = time.time() - fishing_start_time  # noqa
            time_factor = (total_fishing_time / MAX_BOBBER_LIFETIME_SEC) * 0.2
            splash_probability = splash_probability + time_factor
            self._logger.info(f"Adding time factor to splash probability: {round(splash_probability, 2)}")

            # If splash detected, than click on bobber and loot
            if splash_probability > self._splash_threshold:
                self._logger.info(f"Splash probability is higher than threshold: {self._splash_threshold}")

                # Right-click on bobber
                self._logger.info(f"Right-clicking at bobber position...")
                self._input_controller.click(*bobber_position, right_click=True)

                # Wait for some time for loot to show up
                time.sleep(0.5)

                # Move mouse out of the screen
                self._input_controller.move(0, 0)

                # Make screenshot to find looting skull icon
                self._logger.info(f"Making screenshot to detect looting skull icon...")
                screenshot = self._screen_controller.get_screenshot()

                # Find looting skull location
                self._logger.info(f"Inference of loot matching model...")
                skull_location = self._loot_matching.match(screenshot)

                # Compute location of loot
                loot_x, loot_y = [
                    skull_location[0],
                    skull_location[1] + LOOT_OFFSET_PIXELS,
                ]
                self._logger.info(f"Found loot position at {loot_x, loot_y}...")

                # Move mouse to loot position
                self._logger.info(f"Moving mouse to loot position...")
                self._input_controller.move(loot_x, loot_y)

                # Loot all items
                self._logger.info(f"Looting procedure...")
                for _ in range(MAX_LOOT_ITEMS):
                    # Sleep for some time
                    self._logger.info(f"Waiting for {LOOTING_GAP_TIME} seconds...")
                    time.sleep(LOOTING_GAP_TIME)

                    # Loot item
                    self._logger.info(f"Right-clicking on loot...")
                    self._input_controller.click(loot_x, loot_y, right_click=True)

                self._logger.info(f"Fishing is done.")
                is_fishing = False
                continue

            # If time of fishing is exceeded
            elif time.time() - fishing_start_time > MAX_BOBBER_LIFETIME_SEC:
                # Exit fishing state
                self._logger.info(f"Max bobber lifetime exceeded, stopping fishing.")
                is_fishing = False
                continue

            self._logger.info(f"Splash probability is lower than threshold: {self._splash_threshold}")
