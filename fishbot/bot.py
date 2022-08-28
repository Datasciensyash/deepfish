from pathlib import Path
import time

from fishbot.constants import LOOT_OFFSET_PIXELS, MAX_LOOT_ITEMS, LOOTING_GAP_TIME, MAX_BOBBER_LIFETIME_SEC
from fishbot.controllers.input import InputController
from fishbot.controllers.output import OutputController
from fishbot.match import TemplateMatchingModel
from fishbot.model.classifynet import ClassificationNet
from fishbot.model.detectnet import BobberDetector


class FishingBot:

    def __init__(
            self,
            fishing_skill_key: str,
            bobber_checkpoint: Path,
            splash_checkpoint: Path,
            splash_threshold: float = 0.52
    ):
        self._fishing_skill_key = fishing_skill_key
        self._splash_threshold = splash_threshold

        self._input_controller = InputController()
        self._screen_controller = OutputController()

        self._bobber_detector = BobberDetector.from_checkpoint(
            checkpoint_path=bobber_checkpoint,
            mask=None  # TODO: Add mask to checkpoint.
        )
        self._splash_classifier = ClassificationNet.from_checkpoint(splash_checkpoint)
        self._loot_matching = TemplateMatchingModel(Path(self._template_directory / "loot_skull.png"))

    @property
    def _template_directory(self) -> Path:
        return Path(__file__).parent / "templates"

    def run(self) -> None:
        is_fishing = False

        while True:

            # Try to start fishing
            if not is_fishing:
                # Move the cursor so as not to interfere
                self._input_controller.move(0, 0)

                # Tap fishing skill key to start fishing
                self._input_controller.tap_key(self._fishing_skill_key)

                # Save fishing start time
                fishing_start_time = time.time()

                # Take screenshot to detect bobber
                screenshot = self._screen_controller.get_screenshot()

                # Detect position of bobber
                bobber_position = self._bobber_detector.inference(screenshot)

                # If bobber is found, than change state to fishing
                if bobber_position is not None:
                    is_fishing = True

            # Get screenshot near current fishing position
            screenshot = self._screen_controller.get_screenshot(bobber_position)  # noqa
            splash_probability = self._splash_classifier.inference(screenshot)

            # Increase probability by a time factor
            time_factor = ((time.time() - fishing_start_time) / MAX_BOBBER_LIFETIME_SEC) * 0.2 # noqa
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
                loot_x, loot_y = [skull_location[0], skull_location[1] + LOOT_OFFSET_PIXELS]

                # Move mouse to loot position
                self._input_controller.move(loot_x, loot_y)

                # Loot all items
                for _ in range(MAX_LOOT_ITEMS):
                    # Sleep for some time
                    time.sleep(LOOTING_GAP_TIME)

                    # Loot item
                    self._input_controller.click(loot_x, loot_y, right_click=True)

            # If time of fishing is left
            elif time.time() - fishing_start_time > MAX_BOBBER_LIFETIME_SEC:
                # Exit fishing state
                is_fishing = False


