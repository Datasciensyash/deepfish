from threading import Thread

import pyautogui as gui
from pykeyboard import PyKeyboard
from pymouse import PyMouse


class InputController:
    def __init__(self):
        self._mouse_controller = PyMouse()
        self._keyboard_controller = PyKeyboard()

    def move(self, x: int, y: int) -> None:
        """
        Move cursor to specific position.

        Args:
            x: x-coordinate of position
            y: y-coordinate of position

        Returns: None.
        """
        self._mouse_controller.move(x, y)

    def click(self, x: int, y: int, right_click: bool = False) -> None:
        """
        Preform mouse click somewhere on screen.

        Args:
            x: x-coordinate of position
            y: y-coordinate of position
            right_click: Whether to preform right-click or not.

        Returns: None
        """
        click_function = gui.rightClick if right_click else gui.click
        thread = Thread(target=click_function, args=[x, y], daemon=True)
        thread.start()
        self._mouse_controller.click(x, y, 2 if right_click else 1)

    def tap_key(self, button: str) -> None:
        """
        Preform key tap.

        Args:
            button: Name of button, e.g. "0".

        Returns: None
        """
        self._keyboard_controller.tap_key(character=button)
