from abc import ABC, abstractmethod


class ControlInterface(ABC):
    @abstractmethod
    def get_action(self, obs):
        """Return the next action given the current observation."""

    def close(self):
        """Clean up resources if needed."""
