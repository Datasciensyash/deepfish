from win32gui import GetForegroundWindow, GetWindowText

from deepfish.constants import WORLD_OF_WARCRAFT_NAME


class WindowController:
    """
    Controller for game window.

    Note: Im not sure, does this class really need to exists or not.
    """

    @staticmethod
    def is_warcraft_active() -> bool:
        """
        Check active window, and return True, if it's WoW.

        Returns: If WoW window is active - True, else - False.
        """
        return GetWindowText(GetForegroundWindow()) == WORLD_OF_WARCRAFT_NAME
