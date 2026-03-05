import numpy as np
from pynput.keyboard import Key, KeyCode

from interactive_world_sim.utils.control_interface import ControlInterface
from interactive_world_sim.utils.keystroke_counter import KeystrokeCounter


class KeyboardController(ControlInterface):
    def __init__(self, action_dim=4):
        self.action = np.zeros(action_dim)
        self.key_counter = KeystrokeCounter()
        self.key_counter.__enter__()

    def get_action(self, obs):
        # Get recent key press events
        events = self.key_counter.get_press_events()

        # Process each key event
        for event in events:
            if isinstance(event, KeyCode):
                key_char = event.char
                # Left arm control
                if key_char == "d":
                    self.action[3] += 0.05  # Left arm forward (X-)
                elif key_char == "a":
                    self.action[3] -= 0.05  # Left arm backward (X+)
                elif key_char == "w":
                    self.action[2] -= 0.05  # Left arm left (Y+)
                elif key_char == "s":
                    self.action[2] += 0.05  # Left arm right (Y-)

                # Right arm control
                elif key_char == "l":
                    self.action[1] += 0.05  # Right arm forward (X-)
                elif key_char == "j":
                    self.action[1] -= 0.05  # Right arm backward (X+)
                elif key_char == "i":
                    self.action[0] -= 0.05  # Right arm left (Y+)
                elif key_char == "k":
                    self.action[0] += 0.05  # Right arm right (Y-)

                # Reset action to zero
                elif key_char == "r":
                    self.action = np.zeros(4)

            elif isinstance(event, Key):
                # Handle special keys
                if event == Key.esc:
                    # Exit signal - could be handled by the main loop
                    pass

        # Clamp action values
        self.action = np.clip(self.action, -1.0, 1.0)
        return self.action.copy()

    def close(self):
        self.key_counter.__exit__(None, None, None)
