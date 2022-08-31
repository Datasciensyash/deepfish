from pynput.keyboard import Controller as KeyboardController
from pynput.keyboard import Key as KeyboardKeys
from pynput.mouse import Button as MouseButtons
from pynput.mouse import Controller as MouseController


class InputController:
    def __init__(self):
        self._mouse_controller = MouseController()
        self._keyboard_controller = KeyboardController()

    def move(self, x: int, y: int) -> None:
        """
        Move cursor to specific position.

        Args:
            x: x-coordinate of position
            y: y-coordinate of position

        Returns: None.
        """
        self._mouse_controller.position = (x, y)

    def click(self, x: int, y: int, right_click: bool = False) -> None:
        """
        Preform mouse click somewhere on screen.

        Args:
            x: x-coordinate of position
            y: y-coordinate of position
            right_click: Whether to preform right-click or not.

        Returns: None
        """
        self.move(x, y)
        self._mouse_controller.click(MouseButtons.left if not right_click else MouseButtons.right)

    def tap_end_key(self) -> None:
        self.tap_key(KeyboardKeys.end)

    def tap_home_key(self) -> None:
        self.tap_key(KeyboardKeys.home)

    def tap_function_key(self, num: int) -> None:
        """
        Preform function key tap.

        Args:
            num: Number of button, e.g. 0.

        Returns: None
        """
        function_key = getattr(KeyboardKeys, f"f{num}")
        self._keyboard_controller.tap_key(function_key)

    def tap_key(self, button: str) -> None:
        """
        Preform key tap.

        Args:
            button: Name of button, e.g. "0".

        Returns: None
        """
        self._keyboard_controller.tap(button)
