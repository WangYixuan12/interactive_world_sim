from typing import Any

import numpy as np


class BaseEnv:
    """Base class for all environments."""

    def __init__(self, *args: Any, **kwargs: Any):
        pass

    def step(self, action: np.ndarray) -> tuple:
        """Run one timestep of the environment's dynamics."""
        raise NotImplementedError

    def render(self, mode: str = "human", **args: Any) -> dict:
        """Render the environment."""
        raise NotImplementedError

    def compute_init_state(self, hdf5_file_path: str) -> np.ndarray:
        """Compute the initial state of the environment from replay buffer."""
        raise NotImplementedError

    def get_state(self) -> np.ndarray:
        """Get the current state of the environment."""
        raise NotImplementedError

    def get_observations(self) -> dict:
        """Get the current observation of the environment."""
        raise NotImplementedError

    def reset(self, state: Any = None) -> None:
        """Reset the environment and return the initial observation."""
        raise NotImplementedError

    def get_render_size(self) -> tuple[int, int]:
        """Return the render size of the environment."""
        raise NotImplementedError

    def get_curr_pos(self) -> np.ndarray:
        """Return the current position of the environment."""
        raise NotImplementedError

    def get_cam_intrinsic(self, name: str, shape: tuple[int, int]) -> np.ndarray:
        """Get the camera intrinsic matrix."""
        raise NotImplementedError

    def get_cam_extrinsic(self, name: str) -> np.ndarray:
        """Get the camera extrinsic matrix."""
        raise NotImplementedError
