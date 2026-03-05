from typing import Any, Dict, List, Optional, Tuple

import numba
import numpy as np

from .replay_buffer import ReplayBuffer


@numba.jit(nopython=True)
def create_indices(
    episode_ends: np.ndarray,
    sequence_length: int,
    episode_mask: np.ndarray,
    pad_before: int = 0,
    pad_after: int = 0,
    debug: bool = True,
) -> np.ndarray:
    assert episode_mask.shape == episode_ends.shape, "Invalid episode mask"
    pad_before = min(max(pad_before, 0), sequence_length - 1)
    pad_after = min(max(pad_after, 0), sequence_length - 1)

    indices = list()
    for i in range(len(episode_ends)):
        if not episode_mask[i]:
            # skip episode
            continue
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i - 1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            if debug:
                assert start_offset >= 0
                assert end_offset >= 0
                assert (sample_end_idx - sample_start_idx) == (
                    buffer_end_idx - buffer_start_idx
                )
            indices.append(
                [buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx]
            )
    indices = np.array(indices)
    return indices


def get_val_mask(n_episodes: int, val_ratio: float, seed: int = 0) -> np.ndarray:
    val_mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio <= 0:
        return val_mask

    # have at least 1 episode for validation, and at least 1 episode for train
    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes - 1)
    # rng = np.random.default_rng(seed=seed)
    # val_idxs = rng.choice(n_episodes, size=n_val, replace=False)
    # val_mask[val_idxs] = True
    val_mask[-n_val:] = True  # last n_val episodes are validation
    return val_mask


def downsample_mask(
    mask: np.ndarray, max_n: Optional[int], seed: int = 0
) -> np.ndarray:
    # subsample training data
    train_mask = mask
    if (max_n is not None) and (np.sum(train_mask) > max_n):
        n_train = int(max_n)
        curr_train_idxs = np.nonzero(train_mask)[0]
        rng = np.random.default_rng(seed=seed)
        train_idxs_idx = rng.choice(len(curr_train_idxs), size=n_train, replace=False)
        train_idxs = curr_train_idxs[train_idxs_idx]
        train_mask = np.zeros_like(train_mask)
        train_mask[train_idxs] = True
        assert np.sum(train_mask) == n_train
    return train_mask


class SequenceSampler:
    """Sample sequences from replay buffer."""

    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        sequence_length: int,
        pad_before: int = 0,
        pad_after: int = 0,
        keys: Optional[List[str]] = None,
        key_first_k: Dict[str, int] = dict(),  # noqa
        episode_mask: Optional[np.ndarray] = None,
        skip_idx: int = 1,
        goal_sample: str = "final",
        skip_frame: int = 1,
        keys_to_keep_intermediate: Optional[List[str]] = None,
    ):
        """Initialize sequence sampler.

        key_first_k: dict str: int
        Only take first k data from these keys (to improve perf)
        """
        super().__init__()
        assert sequence_length >= 1
        if keys is None:
            keys = list(replay_buffer.keys())

        episode_ends = replay_buffer.episode_ends[:]
        if episode_mask is None:
            episode_mask = np.ones(episode_ends.shape, dtype=bool)

        if np.any(episode_mask):
            indices = create_indices(
                episode_ends,
                sequence_length=sequence_length,
                pad_before=pad_before,
                pad_after=pad_after,
                episode_mask=episode_mask,
            )
        else:
            indices = np.zeros((0, 4), dtype=np.int64)

        # (buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx)
        self.indices = indices
        self.keys = list(keys)  # prevent OmegaConf list performance problem
        self.sequence_length = sequence_length
        self.replay_buffer = replay_buffer
        self.key_first_k = key_first_k
        self.episode_ends = episode_ends
        self.skip_idx = skip_idx
        self.goal_sample = goal_sample
        self.skip_frame = skip_frame
        if keys_to_keep_intermediate is None:
            keys_to_keep_intermediate = list()
        self.keys_to_keep_intermediate = keys_to_keep_intermediate

    def __len__(self) -> int:
        return len(self.indices) // self.skip_idx

    def idx_to_epi_idx(self, idx: int) -> Tuple[int, int]:
        """Get episode index and offset from sequence index."""
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = (
            self.indices[idx]
        )
        return self.buffer_idx_to_epi_idx(buffer_start_idx)

    def buffer_idx_to_epi_idx(self, buffer_idx: int) -> Tuple[int, int]:
        """Get episode index and offset from buffer index."""
        epi_idx = np.searchsorted(self.episode_ends, buffer_idx, side="right")
        epi_offset = (
            buffer_idx - self.episode_ends[epi_idx - 1] if epi_idx > 0 else buffer_idx
        )
        return epi_idx, epi_offset

    def sample_sequence(self, idx: int) -> Dict[str, np.ndarray]:
        """Sample a sequence."""
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = (
            self.indices[idx * self.skip_idx]
        )
        result = dict()

        epi_idx, epi_offset = self.buffer_idx_to_epi_idx(buffer_start_idx)
        episode_end = self.episode_ends[epi_idx]
        is_early_stop = False
        rel_stop_idx = -1
        if self.goal_sample == "final":
            final_idx = episode_end - 1
        elif self.goal_sample == "intermediate":
            intermediate_start = min(buffer_end_idx, episode_end - 1)
            final_idx = np.random.randint(intermediate_start, episode_end)
        elif self.goal_sample == "aggressive":
            early_stop_prob = 0.2
            is_early_stop = np.random.rand() < early_stop_prob
            if is_early_stop:
                rel_stop_idx = np.random.randint(
                    0, self.sequence_length // self.skip_frame
                )
            else:
                intermediate_start = min(buffer_end_idx, episode_end - 1)
                final_idx = np.random.randint(intermediate_start, episode_end)
        else:
            raise ValueError(f"Invalid goal sample: {self.goal_sample}")
        for key in self.keys:
            input_arr = self.replay_buffer[key]
            # performance optimization, avoid small allocation if possible
            if key not in self.key_first_k:
                sample = input_arr[buffer_start_idx:buffer_end_idx]
            else:
                # performance optimization, only load used obs steps
                n_data = buffer_end_idx - buffer_start_idx
                k_data = min(self.key_first_k[key], n_data)
                # fill value with Nan to catch bugs
                # the non-loaded region should never be used
                sample = np.full(
                    (n_data,) + input_arr.shape[1:],
                    fill_value=np.nan,
                    dtype=input_arr.dtype,
                )
                sample[:k_data] = input_arr[
                    buffer_start_idx : buffer_start_idx + k_data
                ]
            data = sample
            # is_pad = np.zeros(self.sequence_length, dtype=bool)
            if (sample_start_idx > 0) or (sample_end_idx < self.sequence_length):
                data = np.zeros(
                    shape=(self.sequence_length,) + data.shape[1:], dtype=data.dtype
                )
                if sample_start_idx > 0:
                    data[:sample_start_idx] = sample[0]
                    # is_pad[:sample_start_idx] = True
                if sample_end_idx < self.sequence_length:
                    data[sample_end_idx:] = sample[-1]
                    # is_pad[sample_end_idx:] = True
                data[sample_start_idx:sample_end_idx] = sample
            if key in self.keys_to_keep_intermediate:
                inter_frames = self.sequence_length // self.skip_frame
                data_shape = list(data.shape[1:])
                data_shape[0] = data_shape[0] * self.skip_frame
                result[key] = data.reshape(
                    inter_frames, self.skip_frame, *data.shape[1:]
                )
                result[key] = result[key].reshape(-1, *data_shape)
                # is_pad_shape = list(is_pad.shape[1:])
                # is_pad_shape[0] = is_pad_shape[0] * self.skip_frame
                # result[f"{key}_is_pad"] = \
                #   is_pad.reshape(inter_frames, self.skip_frame, *is_pad.shape[1:])
                # result[f"{key}_is_pad"] = \
                #   result[f"{key}_is_pad"].reshape(-1, *is_pad_shape)
            else:
                result[key] = data[:: self.skip_frame]
                # result[f"{key}_is_pad"] = is_pad[:: self.skip_frame]
            if self.goal_sample == "aggressive" and is_early_stop:
                # sample[rel_stop_idx:] = np.repeat(
                #     sample[rel_stop_idx][None], len(sample) - rel_stop_idx, axis=0
                # )
                final_obs = result[key][rel_stop_idx]
            else:
                final_obs = input_arr[final_idx]
            result[f"{key}_final"] = final_obs
            result["is_early_stop"] = is_early_stop
            result["rel_stop_idx"] = rel_stop_idx
        return result

    def sample_pair_from_buffer_idx(
        self,
        buffer_idx: int,
        neg_prob: float = 0.8,
        pos_range: int = 5,
        neg_range: int = 50,
    ) -> Dict[str, Any]:
        """Sample another step from the same episode of idx."""
        epi_idx, _ = self.buffer_idx_to_epi_idx(buffer_idx)
        epi_buffer_idx_start = self.episode_ends[epi_idx - 1] if epi_idx > 0 else 0
        epi_buffer_idx_end = self.episode_ends[epi_idx]
        if (
            epi_buffer_idx_end <= buffer_idx + neg_range
            and epi_buffer_idx_start >= buffer_idx - neg_range
        ):
            # not enough data for negative sample
            neg_prob = 0.0
        if np.random.rand() < neg_prob:
            # sample negative pair
            idx_selections = []
            if epi_buffer_idx_end > buffer_idx + neg_range:
                idx_selections.append(
                    np.arange(buffer_idx + neg_range, epi_buffer_idx_end)
                )
            if epi_buffer_idx_start < buffer_idx - neg_range:
                idx_selections.append(
                    np.arange(epi_buffer_idx_start, buffer_idx - neg_range)
                )
            idx_selections = np.concatenate(idx_selections)
            neg_idx = np.random.choice(idx_selections)
            return {
                "idx": neg_idx,
                "is_positive": False,
                "offset": neg_idx - epi_buffer_idx_start,
                "epi_len": epi_buffer_idx_end - epi_buffer_idx_start,
            }
        else:
            # sample positive pair
            pos_idx = np.random.randint(
                max(epi_buffer_idx_start, buffer_idx - pos_range),
                min(epi_buffer_idx_end, buffer_idx + pos_range),
            )
            return {
                "idx": pos_idx,
                "is_positive": True,
                "offset": pos_idx - epi_buffer_idx_start,
                "epi_len": epi_buffer_idx_end - epi_buffer_idx_start,
            }

    def sample_pairs(
        self, idx: int, neg_prob: float = 0.8, pos_range: int = 5, neg_range: int = 50
    ) -> Dict[str, Any]:
        """Sample_pairs."""
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = (
            self.indices[idx * self.skip_idx]
        )
        pair_info = self.sample_pair_from_buffer_idx(
            buffer_start_idx, neg_prob, pos_range, neg_range
        )
        result: dict = dict()
        result["pair"] = dict()
        result["pair"]["is_positive"] = pair_info["is_positive"]
        result["pair"]["offset"] = pair_info["offset"]
        result["pair"]["epi_len"] = pair_info["epi_len"]

        for key in self.keys:
            input_arr = self.replay_buffer[key]
            sample = input_arr[buffer_start_idx:buffer_end_idx]
            data = sample

            # pad the data
            # is_pad = np.zeros(self.sequence_length, dtype=bool)
            if (sample_start_idx > 0) or (sample_end_idx < self.sequence_length):
                data = np.zeros(
                    shape=(self.sequence_length,) + data.shape[1:], dtype=data.dtype
                )
                if sample_start_idx > 0:
                    data[:sample_start_idx] = sample[0]
                    # is_pad[:sample_start_idx] = True
                if sample_end_idx < self.sequence_length:
                    data[sample_end_idx:] = sample[-1]
                    # is_pad[sample_end_idx:] = True
                data[sample_start_idx:sample_end_idx] = sample

            # save the data
            result[key] = data
            # result[f"{key}_is_pad"] = is_pad
            result["offset"] = self.buffer_idx_to_epi_idx(buffer_start_idx)[1]
            result["pair"][key] = input_arr[pair_info["idx"]][None]
            result["epi_len"] = pair_info["epi_len"]

        return result
