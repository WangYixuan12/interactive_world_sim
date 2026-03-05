import math
from typing import Dict, List, Optional, Tuple

import numpy as np


def get_accumulate_timestamp_idxs(
    timestamps: List[float],
    start_time: float,
    dt: float,
    eps: float = 1e-5,
    next_global_idx: Optional[int] = 0,
    allow_negative: bool = False,
) -> Tuple[List[int], List[int], int]:
    """Choose the first timestamp in each dt window.

    Assumes timestamps are sorted. One timestamp might be chosen multiple times
    due to dropped frames.

    next_global_idx should start at 0 normally, and then use the returned
    next_global_idx. However, when overwriting previous values is desired,
    set next_global_idx to None.

    Returns:
        local_idxs: which index in the given timestamps array to choose from.
        global_idxs: the global index of each chosen timestamp.
        next_global_idx: updated value for subsequent calls.
    """
    local_idxs: List[int] = []
    global_idxs: List[int] = []
    for local_idx, ts in enumerate(timestamps):
        # Add eps * dt so that when ts == start_time + k * dt it is
        # always recorded as the kth element (avoiding floating point errors).
        global_idx = math.floor((ts - start_time) / dt + eps)
        if (not allow_negative) and (global_idx < 0):
            continue
        if next_global_idx is None:
            next_global_idx = global_idx

        n_repeats = max(0, global_idx - next_global_idx + 1)
        for _ in range(n_repeats):
            local_idxs.append(local_idx)
            global_idxs.append(next_global_idx)
            next_global_idx += 1
    assert next_global_idx is not None
    return local_idxs, global_idxs, next_global_idx


def align_timestamps(
    timestamps: List[float],
    target_global_idxs: List[int],
    start_time: float,
    dt: float,
    eps: float = 1e-5,
) -> List[int]:
    """Align the given timestamps to the target global indices.

    Args:
        timestamps: List of timestamp values.
        target_global_idxs: Global indices to which timestamps should be aligned.
        start_time: The reference start time.
        dt: The desired time difference between samples.
        eps: A small epsilon to avoid floating point issues.

    Returns:
        A list of local indices corresponding to the target global indices.
    """
    if isinstance(target_global_idxs, np.ndarray):
        target_global_idxs = target_global_idxs.tolist()
    assert len(target_global_idxs) > 0

    local_idxs, global_idxs, _ = get_accumulate_timestamp_idxs(
        timestamps=timestamps,
        start_time=start_time,
        dt=dt,
        eps=eps,
        next_global_idx=target_global_idxs[0],
        allow_negative=True,
    )
    if len(global_idxs) > len(target_global_idxs):
        # If more steps are available, truncate.
        global_idxs = global_idxs[: len(target_global_idxs)]
        local_idxs = local_idxs[: len(target_global_idxs)]

    for _ in range(len(target_global_idxs) - len(global_idxs)):
        # If missing, repeat the last available index.
        local_idxs.append(len(timestamps) - 1)
        global_idxs.append(global_idxs[-1] + 1)
    assert global_idxs == target_global_idxs
    assert len(local_idxs) == len(global_idxs)
    return local_idxs


class TimestampObsAccumulator:
    """Accumulates observation data along with timestamps.

    This class buffers observation data and corresponding timestamps based on a
    fixed time interval (dt).
    """

    def __init__(self, start_time: float, dt: float, eps: float = 1e-5) -> None:
        """Initialize the observation accumulator.

        Args:
            start_time: The reference start time.
            dt: The fixed time interval between samples.
            eps: A small epsilon to mitigate floating point errors.
        """
        self.start_time: float = start_time
        self.dt: float = dt
        self.eps: float = eps
        self.obs_buffer: Dict[str, np.ndarray] = {}  # Buffer for observations.
        self.timestamp_buffer: Optional[np.ndarray] = None
        self.next_global_idx: int = 0

    def __len__(self) -> int:
        return self.next_global_idx

    @property
    def data(self) -> Dict[str, np.ndarray]:
        """Get the accumulated observation data up to the current index."""
        if self.timestamp_buffer is None:
            return dict()
        result: Dict[str, np.ndarray] = {}
        for key, value in self.obs_buffer.items():
            result[key] = value[: len(self)]
        return result

    @property
    def actual_timestamps(self) -> np.ndarray:
        """Return the raw timestamps from the buffer."""
        if self.timestamp_buffer is None:
            return np.array([])
        return self.timestamp_buffer[: len(self)]

    @property
    def timestamps(self) -> np.ndarray:
        """Generate timestamps based on the start time and dt."""
        if self.timestamp_buffer is None:
            return np.array([])
        return self.start_time + np.arange(len(self)) * self.dt

    def put(self, data: Dict[str, np.ndarray], timestamps: np.ndarray) -> None:
        """Accumulate observation data.

        Args:
            data: A dictionary mapping each observation key to a NumPy array. The
                  first dimension of each array corresponds to time.
            timestamps: A NumPy array of timestamp values.
        """
        local_idxs, global_idxs, self.next_global_idx = get_accumulate_timestamp_idxs(
            timestamps=timestamps,
            start_time=self.start_time,
            dt=self.dt,
            eps=self.eps,
            next_global_idx=self.next_global_idx,
        )

        if len(global_idxs) > 0:
            if self.timestamp_buffer is None:
                # First allocation.
                self.obs_buffer = {}
                for key, value in data.items():
                    self.obs_buffer[key] = np.zeros_like(value)
                self.timestamp_buffer = np.zeros((len(timestamps),), dtype=np.float64)

            this_max_size = global_idxs[-1] + 1
            if this_max_size > len(self.timestamp_buffer):
                # Reallocate with a larger buffer size.
                new_size = max(this_max_size, len(self.timestamp_buffer) * 2)
                for key in list(self.obs_buffer.keys()):
                    new_shape = (new_size,) + self.obs_buffer[key].shape[1:]
                    self.obs_buffer[key] = np.resize(self.obs_buffer[key], new_shape)
                self.timestamp_buffer = np.resize(self.timestamp_buffer, (new_size,))

            # Write new observation data.
            for key, value in self.obs_buffer.items():
                value[global_idxs] = data[key][local_idxs]
            self.timestamp_buffer[global_idxs] = timestamps[local_idxs]


class TimestampActionAccumulator:
    """Accumulates action data along with timestamps.

    Unlike the observation accumulator, this accumulator allows overwriting
    previous values.
    """

    def __init__(self, start_time: float, dt: float, eps: float = 1e-5) -> None:
        """Initialize the action accumulator.

        Args:
            start_time: The reference start time.
            dt: The fixed time interval between action samples.
            eps: A small epsilon to mitigate floating point errors.
        """
        self.start_time: float = start_time
        self.dt: float = dt
        self.eps: float = eps
        self.action_buffer: np.ndarray = np.zeros((0,), dtype=np.float64)
        self.timestamp_buffer: Optional[np.ndarray] = None
        self.size: int = 0

    def __len__(self) -> int:
        return self.size

    @property
    def actions(self) -> np.ndarray:
        """Return the accumulated actions."""
        return self.action_buffer[: len(self)]

    @property
    def actual_timestamps(self) -> np.ndarray:
        """Return the raw timestamps from the action buffer."""
        if self.timestamp_buffer is None:
            return np.array([])
        return self.timestamp_buffer[: len(self)]

    @property
    def timestamps(self) -> np.ndarray:
        """Generate timestamps based on the start time and dt."""
        if self.timestamp_buffer is None:
            return np.array([])
        return self.start_time + np.arange(len(self)) * self.dt

    def put(self, actions: np.ndarray, timestamps: np.ndarray) -> None:
        """Accumulate actions along with their timestamps.

        Note:
            The provided timestamps indicate the time when the action will be issued,
            not when the action has been completed.

        Args:
            actions: A NumPy array of actions.
            timestamps: A NumPy array of timestamp values.
        """
        local_idxs, global_idxs, _ = get_accumulate_timestamp_idxs(
            timestamps=timestamps,
            start_time=self.start_time,
            dt=self.dt,
            eps=self.eps,
            # Allows overwriting previous actions.
            next_global_idx=None,
        )

        if len(global_idxs) > 0:
            if self.timestamp_buffer is None:
                # First allocation.
                self.action_buffer = np.zeros_like(actions)
                self.timestamp_buffer = np.zeros((len(actions),), dtype=np.float64)

            this_max_size = global_idxs[-1] + 1
            if this_max_size > len(self.timestamp_buffer):
                # Reallocate to a larger buffer.
                new_size = max(this_max_size, len(self.timestamp_buffer) * 2)
                new_shape = (new_size,) + self.action_buffer.shape[1:]
                self.action_buffer = np.resize(self.action_buffer, new_shape)
                self.timestamp_buffer = np.resize(self.timestamp_buffer, (new_size,))

            # Write (and potentially overwrite) old data.
            self.action_buffer[global_idxs] = actions[local_idxs]
            self.timestamp_buffer[global_idxs] = timestamps[local_idxs]
            self.size = max(self.size, this_max_size)
