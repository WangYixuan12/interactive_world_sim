from typing import Tuple

import numpy as np


def sync_timestamps(
    high_freq_data: np.ndarray, low_freq_data: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Sync timestamps between observations and actions.

    Args:
        high_freq_data: Observation timestamps in high frequency
        low_freq_data: Action timestamps in low frequency

    Returns:
        Tuple of synced observation and action timestamps.

    Details:
        - find the closest timestamp in obs_timestamps to curr_action_t
        - if the difference if larger than 1/30. s, just choose the largest
        obs_timestamp that is smaller than curr_action_t
    """
    # Filter out data that is out of overlap
    mean_low_freq_dt = np.mean(np.diff(low_freq_data))
    min_high_freq_data = high_freq_data[0]
    max_high_freq_data = high_freq_data[-1]
    min_low_freq_data = low_freq_data[0]
    max_low_freq_data = low_freq_data[-1]
    min_data = max(min_high_freq_data, min_low_freq_data)
    max_data = min(max_high_freq_data, max_low_freq_data)
    low_freq_data_valid = (low_freq_data >= min_data - mean_low_freq_dt) & (
        low_freq_data <= max_data + mean_low_freq_dt
    )
    high_freq_data_valid = (high_freq_data >= min_data - mean_low_freq_dt) & (
        high_freq_data <= max_data + mean_low_freq_dt
    )
    low_freq_data_offset = np.nonzero(low_freq_data_valid)[0][0]
    high_freq_data_offset = np.nonzero(high_freq_data_valid)[0][0]
    low_freq_data = low_freq_data[low_freq_data_valid]
    high_freq_data = high_freq_data[high_freq_data_valid]

    # Find insertion points of action timestamps in observation timestamps
    idxs = np.searchsorted(high_freq_data, low_freq_data)

    # Determine left and right candidate indices
    left_idxs = idxs - 1
    right_idxs = idxs

    # Check validity of left and right indices
    left_possible = left_idxs >= 0
    right_possible = right_idxs < len(high_freq_data)

    # Initialize arrays to hold left and right observation timestamps
    left_ts = np.full_like(low_freq_data, np.nan, dtype=np.float64)
    right_ts = np.full_like(low_freq_data, np.nan, dtype=np.float64)

    # Fill valid left and right timestamps
    left_ts[left_possible] = high_freq_data[left_idxs[left_possible]]
    right_ts[right_possible] = high_freq_data[right_idxs[right_possible]]

    # Compute differences; invalid entries (NaN) will be replaced with infinity
    left_diff = low_freq_data - left_ts
    right_diff = right_ts - low_freq_data
    left_diff[~left_possible] = np.inf
    right_diff[~right_possible] = np.inf

    # Determine which side is closer
    left_closer = left_diff <= right_diff
    closest_idx = np.where(left_closer, left_idxs, right_idxs)
    closest_diff = np.where(left_closer, left_diff, right_diff)

    # Check if the closest is within the threshold
    threshold = 1.0 / 30.0
    mask = closest_diff <= threshold

    # Determine valid entries where either mask is True or left is available
    # when mask is False
    else_valid = left_possible
    valid = mask | (~mask & else_valid)

    # Select the observation indices based on the conditions
    selected_high_freq_idx = np.where(mask, closest_idx, left_idxs)

    # Apply the valid mask to get the indices
    high_freq_idx = selected_high_freq_idx[valid].astype(int)
    low_freq_idx = np.flatnonzero(valid).astype(int)
    high_freq_idx += high_freq_data_offset
    low_freq_idx += low_freq_data_offset

    assert len(high_freq_idx) == len(low_freq_idx), "Mismatch in length"
    return high_freq_idx, low_freq_idx


def test_sync_timestamps() -> None:
    high_freq_hz = 30
    low_freq_hz = 10

    high_freq_start_time = 0.0
    low_freq_start_time = 2.0 / 30.0

    high_freq_data = np.arange(
        high_freq_start_time,
        high_freq_start_time + 10.0,
        1.0 / high_freq_hz,
    )

    low_freq_data = np.arange(
        low_freq_start_time,
        low_freq_start_time + 10.0,
        1.0 / low_freq_hz,
    )
    high_freq_data += np.random.uniform(-0.01, 0.01, size=high_freq_data.shape)
    high_freq_idx, low_freq_idx = sync_timestamps(high_freq_data, low_freq_data)

    gt_high_freq_idx = np.arange(2, 10 * 30 + 2, 3).astype(int)
    gt_low_freq_idx = np.arange(0, 10 * 10, 1).astype(int)
    assert np.array_equal(high_freq_idx, gt_high_freq_idx)
    assert np.array_equal(low_freq_idx, gt_low_freq_idx)

    high_freq_hz = 100
    low_freq_hz = 30
    high_freq_start_time = 0.0
    low_freq_start_time = 30.0 / 100.0
    high_freq_data = np.arange(
        high_freq_start_time,
        high_freq_start_time + 1.0,
        1.0 / high_freq_hz,
    )
    low_freq_data = np.arange(
        low_freq_start_time,
        low_freq_start_time + 10.0,
        1.0 / low_freq_hz,
    )
    high_freq_idx, low_freq_idx = sync_timestamps(high_freq_data, low_freq_data)
    gt_high_freq_idx = np.array(
        [
            30,
            33,
            37,
            40,
            43,
            47,
            50,
            53,
            57,
            60,
            63,
            67,
            70,
            73,
            77,
            80,
            83,
            87,
            90,
            93,
            97,
            99,
        ]
    )
    gt_low_freq_idx = np.arange(0, 22).astype(int)
    assert np.array_equal(high_freq_idx, gt_high_freq_idx)
    assert np.array_equal(low_freq_idx, gt_low_freq_idx)


if __name__ == "__main__":
    test_sync_timestamps()
    print("All tests passed!")
