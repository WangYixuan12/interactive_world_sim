import concurrent.futures
import copy
import gc
import glob
import multiprocessing
import os
import shutil
from pathlib import Path
from typing import Dict, Optional

import cv2
import h5py
import numpy as np
import psutil
import torch
import zarr
import zarr.storage
from filelock import FileLock
from imgaug import augmenters as iaa
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from yixuan_utilities.draw_utils import center_crop

from interactive_world_sim.utils.imagecodecs_numcodecs import Jpeg2k, register_codecs
from interactive_world_sim.utils.normalizer import (
    LinearNormalizer,
    array_to_stats,
    get_identity_normalizer_from_stat,
    get_image_range_normalizer,
    get_range_normalizer_from_stat,
)
from interactive_world_sim.utils.pytorch_util import dict_apply
from interactive_world_sim.utils.replay_buffer import ReplayBuffer
from interactive_world_sim.utils.sampler import SequenceSampler

from .base_dataset import BaseImageDataset

register_codecs()


# convert raw hdf5 data to replay buffer, which is used for diffusion policy training
def _convert_real_to_dp_replay(
    store: zarr.storage.Store,
    shape_meta: dict,
    dataset_dir: str,
    n_workers: Optional[int] = None,
    max_inflight_tasks: Optional[int] = None,
) -> ReplayBuffer:
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    if max_inflight_tasks is None:
        max_inflight_tasks = n_workers * 5

    # parse shape_meta
    rgb_keys = list()
    depth_keys = list()
    lowdim_keys = list()
    # construct compressors and chunks
    obs_shape_meta = shape_meta["obs"]
    for key, attr in obs_shape_meta.items():
        shape = attr["shape"]
        type = attr.get("type", "low_dim")
        if type == "rgb":
            rgb_keys.append(key)
        if type == "depth":
            depth_keys.append(key)
        elif type == "low_dim":
            lowdim_keys.append(key)

    root = zarr.group(store)
    data_group = root.require_group("data", overwrite=True)
    meta_group = root.require_group("meta", overwrite=True)

    episodes_paths = glob.glob(os.path.join(dataset_dir, "episode_*.hdf5"))
    episodes_stem_name = [Path(path).stem for path in episodes_paths]
    episodes_idx = [int(stem_name.split("_")[-1]) for stem_name in episodes_stem_name]
    episodes_idx = sorted(episodes_idx)

    episode_ends = list()
    prev_end = 0
    lowdim_data_dict: dict = dict()
    rgb_data_dict: dict = dict()
    depth_data_dict: dict = dict()
    for epi_idx in tqdm(episodes_idx, desc="Loading episodes"):
        dataset_path = os.path.join(dataset_dir, f"episode_{epi_idx}.hdf5")
        with h5py.File(dataset_path) as file:
            # count total steps
            episode_length = file["action"].shape[0]
            episode_end = prev_end + episode_length
            prev_end = episode_end
            episode_ends.append(episode_end)

            # save lowdim data to lowedim_data_dict
            if "action" not in lowdim_data_dict:
                lowdim_data_dict["action"] = list()
            this_data = file["obs"]["ee_pos"][()]  # (T, 2, 4, 4)
            action_data = np.concatenate(
                [this_data[:, 0, :2, 3], this_data[:, 1, :2, 3]], axis=1
            )  # (T, 4)
            lowdim_data_dict["action"].append(action_data)

            for key in rgb_keys:
                if key not in rgb_data_dict:
                    rgb_data_dict[key] = list()
                imgs = file["obs"]["images"][key][()]
                shape = tuple(shape_meta["obs"][key]["shape"])
                c, h, w = shape
                crop_imgs = [center_crop(img, (h, w)) for img in imgs]
                resize_imgs = [
                    cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
                    for img in crop_imgs
                ]
                imgs = np.stack(resize_imgs, axis=0)
                assert imgs[0].shape == (h, w, c)
                rgb_data_dict[key].append(imgs)

            for key in depth_keys:
                if key not in depth_data_dict:
                    depth_data_dict[key] = list()
                imgs = file["obs"]["images"][key][()]
                shape = tuple(shape_meta["obs"][key]["shape"])
                c, h, w = shape
                crop_imgs = [center_crop(img, (h, w)) for img in imgs]
                resize_imgs = [
                    cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
                    for img in crop_imgs
                ]
                imgs = np.stack(resize_imgs, axis=0)[..., None]
                imgs = np.clip(imgs, 0, 1000).astype(np.uint16)
                assert imgs[0].shape == (h, w, c)
                depth_data_dict[key].append(imgs)

    def img_copy(
        zarr_arr: zarr.Array, zarr_idx: int, hdf5_arr: np.ndarray, hdf5_idx: int
    ) -> bool:
        try:
            zarr_arr[zarr_idx] = hdf5_arr[hdf5_idx]
            # make sure we can successfully decode
            _ = zarr_arr[zarr_idx]
            return True
        except Exception:
            return False

    # dump data_dict
    print("Dumping meta data")
    n_steps = episode_ends[-1]
    _ = meta_group.array(
        "episode_ends", episode_ends, dtype=np.int64, compressor=None, overwrite=True
    )

    print("Dumping lowdim data")
    for key, data in lowdim_data_dict.items():
        data = np.concatenate(data, axis=0)
        _ = data_group.array(
            name=key,
            data=data,
            shape=data.shape,
            chunks=data.shape,
            compressor=None,
            dtype=data.dtype,
        )

    print("Dumping rgb data")
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures: set = set()
        for key, data in rgb_data_dict.items():
            hdf5_arr = np.concatenate(data, axis=0)
            shape = tuple(shape_meta["obs"][key]["shape"])
            c, h, w = shape
            this_compressor = Jpeg2k(level=50)
            img_arr = data_group.require_dataset(
                name=key,
                shape=(n_steps, h, w, c),
                chunks=(1, h, w, c),
                compressor=this_compressor,
                dtype=np.uint8,
            )
            for hdf5_idx in tqdm(range(hdf5_arr.shape[0])):
                if len(futures) >= max_inflight_tasks:
                    # limit number of inflight tasks
                    completed, futures = concurrent.futures.wait(
                        futures, return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    for f in completed:
                        if not f.result():
                            raise RuntimeError("Failed to encode image!")
                zarr_idx = hdf5_idx
                futures.add(
                    executor.submit(img_copy, img_arr, zarr_idx, hdf5_arr, hdf5_idx)
                )
        completed, futures = concurrent.futures.wait(futures)
        for f in completed:
            if not f.result():
                raise RuntimeError("Failed to encode image!")

    print("Dumping depth data")
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = set()
        for key, data in depth_data_dict.items():
            hdf5_arr = np.concatenate(data, axis=0)
            shape = tuple(shape_meta["obs"][key]["shape"])
            c, h, w = shape
            this_compressor = Jpeg2k(level=50)
            img_arr = data_group.require_dataset(
                name=key,
                shape=(n_steps, h, w, c),
                chunks=(1, h, w, c),
                compressor=this_compressor,
                dtype=np.uint16,
            )
            for hdf5_idx in tqdm(range(hdf5_arr.shape[0])):
                if len(futures) >= max_inflight_tasks:
                    # limit number of inflight tasks
                    completed, futures = concurrent.futures.wait(
                        futures, return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    for f in completed:
                        if not f.result():
                            raise RuntimeError("Failed to encode image!")
                zarr_idx = hdf5_idx
                futures.add(
                    executor.submit(img_copy, img_arr, zarr_idx, hdf5_arr, hdf5_idx)
                )
        completed, futures = concurrent.futures.wait(futures)
        for f in completed:
            if not f.result():
                raise RuntimeError("Failed to encode image!")

    replay_buffer = ReplayBuffer(root)
    return replay_buffer


def load_replay_buffer(
    dataset_dir: str, use_cache: bool, shape_meta: dict
) -> ReplayBuffer:
    replay_buffer = None
    if use_cache:
        cache_info_str = ""
        cache_zarr_path = os.path.join(dataset_dir, f"cache{cache_info_str}.zarr.zip")
        cache_lock_path = cache_zarr_path + ".lock"
        print("Acquiring lock on cache.")
        with FileLock(cache_lock_path):
            if not os.path.exists(cache_zarr_path):
                try:
                    print("Cache does not exist. Creating!")
                    # store = zarr.DirectoryStore(cache_zarr_path)
                    replay_buffer = _convert_real_to_dp_replay(
                        store=zarr.MemoryStore(),
                        shape_meta=shape_meta,
                        dataset_dir=dataset_dir,
                    )
                    print("Saving cache to disk.")
                    with zarr.ZipStore(cache_zarr_path) as zip_store:
                        replay_buffer.save_to_store(store=zip_store)
                except Exception as e:
                    shutil.rmtree(cache_zarr_path)
                    raise e
            else:
                print("Loading cached ReplayBuffer from Disk.")
                with zarr.ZipStore(cache_zarr_path, mode="r") as zip_store:
                    replay_buffer = ReplayBuffer.copy_from_store(
                        src_store=zip_store, store=zarr.MemoryStore()
                    )
                print("Loaded!")
    else:
        replay_buffer = _convert_real_to_dp_replay(
            store=zarr.MemoryStore(),
            shape_meta=shape_meta,
            dataset_dir=dataset_dir,
        )
    return replay_buffer


class SimAlohaDataset(BaseImageDataset):
    """A dataset for the real-world data collected on Aloha robot."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        # assign config
        shape_meta = cfg.shape_meta
        dataset_dir = cfg.dataset_dir
        horizon = cfg.horizon * cfg.skip_frame
        pad_before = cfg.pad_before
        pad_after = cfg.pad_after
        use_cache = cfg.use_cache
        self.val_horizon = (
            cfg.val_horizon * cfg.skip_frame if "val_horizon" in cfg else horizon
        )
        self.skip_idx = cfg.skip_idx if "skip_idx" in cfg else 1
        self.aug_mode = cfg.aug_mode
        if cfg.aug_mode == "img_aug":
            self.aug = iaa.Sequential(
                [
                    iaa.Affine(
                        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                        rotate=(-30, 30),
                        mode="edge",
                    ),
                    iaa.AdditiveGaussianNoise(
                        loc=0, scale=(0.0, 0.05), per_channel=0.5
                    ),
                    iaa.MultiplyHueAndSaturation(
                        mul_hue=(0.8, 1.2), mul_saturation=(0.8, 1.2)
                    ),
                    iaa.MultiplyBrightness(mul=(0.8, 1.2)),
                ]
            )
        elif cfg.aug_mode == "none":
            self.aug = None
        else:
            raise ValueError(f"Invalid augmentation mode: {cfg.aug_mode}")

        train_dir = os.path.join(dataset_dir, "train")
        self.replay_buffer = load_replay_buffer(train_dir, use_cache, shape_meta)

        rgb_keys = list()
        depth_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            type = attr.get("type", "low_dim")
            if type == "rgb":
                rgb_keys.append(key)
            elif type == "depth":
                depth_keys.append(key)
            elif type == "low_dim":
                lowdim_keys.append(key)

        train_mask = np.ones((self.replay_buffer.n_episodes,), dtype=bool)
        all_keys = list(self.replay_buffer.keys())

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            goal_sample=cfg.goal_sample,
            keys=all_keys,
            skip_frame=cfg.skip_frame,
            keys_to_keep_intermediate=["action"],
        )

        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.depth_keys = depth_keys
        self.lowdim_keys = lowdim_keys
        self.train_mask = train_mask
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.dataset_dir = dataset_dir
        self.skip_frame = cfg.skip_frame
        self.goal_sample = cfg.goal_sample
        self.use_cache = use_cache
        self.resolution = cfg.resolution

    def get_normalizer(self, mode: str = "none", **kwargs: dict) -> LinearNormalizer:
        """Return a normalizer for the dataset."""
        normalizer = LinearNormalizer()

        # action
        stat = array_to_stats(self.replay_buffer["action"])
        this_normalizer = get_range_normalizer_from_stat(stat)
        normalizer["action"] = this_normalizer

        # obs
        for key in self.lowdim_keys:
            stat = array_to_stats(self.replay_buffer[key])

            if key.endswith("pos"):
                # this_normalizer = get_range_normalizer_from_stat(stat)
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith("quat"):
                # quaternion is in [-1,1] already
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith("qpos"):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith("vel"):
                this_normalizer = get_identity_normalizer_from_stat(stat)
            else:
                raise RuntimeError("unsupported")
            normalizer[key] = this_normalizer

        # image
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()

        for key in self.depth_keys:
            normalizer[key] = get_image_range_normalizer()

        return normalizer

    def __len__(self) -> int:
        if self.is_val:
            # the number of episodes in the validation set
            return self.replay_buffer.n_episodes // self.skip_idx
        else:
            return len(self.sampler)

    def get_validation_dataset(self) -> "BaseImageDataset":
        """Return a validation dataset."""
        val_set = copy.copy(self)
        val_set.is_val = True
        val_dir = os.path.join(self.dataset_dir, "val")
        shape_meta = self.shape_meta
        use_cache = self.use_cache
        val_set.replay_buffer = load_replay_buffer(val_dir, use_cache, shape_meta)
        val_mask = np.ones((val_set.replay_buffer.n_episodes,), dtype=bool)
        val_set.sampler = SequenceSampler(
            replay_buffer=val_set.replay_buffer,
            sequence_length=self.val_horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=val_mask,
            skip_idx=self.skip_idx,
            goal_sample=self.goal_sample,
            skip_frame=self.skip_frame,
            keys_to_keep_intermediate=["action"],
        )
        val_set.train_mask = val_mask
        return val_set

    def _sample_to_data(self, sample: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        obs_dict = dict()
        final_dict = dict()

        # Apply augmentation with 0.2 probability
        apply_aug = np.random.random() < 0.2 if self.aug_mode == "img_aug" else False

        # skip_start = np.random.randint(0, self.skip_frame) + self.skip_frame
        for key in self.rgb_keys:
            # move channel last to channel first
            # T,H,W,C
            # convert uint8 image to float32
            obs_images = sample[key].astype(np.uint8)
            final_images = sample[f"{key}_final"].astype(np.uint8)

            # Apply augmentation if probability condition is met
            if apply_aug:
                aug_det = self.aug.to_deterministic()
                combined = [*obs_images, final_images]
                # apply the deterministic augmenter separately to each image
                combined_aug = [aug_det.augment_image(img) for img in combined]
                obs_images = np.stack(combined_aug[:-1], axis=0)
                final_images = combined_aug[-1]  # Last image

            obs_dict[key] = np.moveaxis(obs_images, -1, 1).astype(np.float32) / 255.0
            # obs_dict[key] = obs_dict[key][skip_start :: self.skip_frame]
            final_dict[key] = (
                np.moveaxis(final_images, -1, 0).astype(np.float32) / 255.0
            )
            del sample[f"{key}_final"]
            # T,C,H,W
            del sample[key]
        for key in self.depth_keys:
            # move channel last to channel first
            # T,H,W,C
            # convert uint16 image to float32
            obs_dict[key] = np.moveaxis(sample[key], -1, 1).astype(np.float32) / 1000.0
            # obs_dict[key] = obs_dict[key][skip_start :: self.skip_frame]
            final_dict[key] = (
                np.moveaxis(sample[f"{key}_final"], -1, 0).astype(np.float32) / 1000.0
            )
            del sample[f"{key}_final"]
            # T,C,H,W
            del sample[key]
        for key in self.lowdim_keys:
            obs_dict[key] = sample[key].astype(np.float32)
            # obs_dict[key] = obs_dict[key][skip_start :: self.skip_frame]
            final_dict[key] = sample[f"{key}_final"].astype(np.float32)
            del sample[f"{key}_final"]
            del sample[key]

        actions = sample["action"].astype(np.float32)
        # action_dim = actions.shape[-1]
        # downsample_horizon = actions.shape[0] // self.skip_frame - 1
        # action_len = downsample_horizon * self.skip_frame
        # action_start = skip_start - self.skip_frame
        # actions = actions[action_start : action_start + action_len]
        # actions = actions.reshape(downsample_horizon, self.skip_frame, action_dim)
        # actions = actions.reshape(downsample_horizon, self.skip_frame * action_dim)
        data = {
            "obs": dict_apply(obs_dict, torch.from_numpy),
            "goal": dict_apply(final_dict, torch.from_numpy),
            "action": torch.from_numpy(actions),
            "is_early_stop": torch.from_numpy(np.array([sample["is_early_stop"]])),
            "rel_stop_idx": torch.from_numpy(np.array([sample["rel_stop_idx"]])),
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.is_val:
            epi_idx = idx * self.skip_idx
            epi_start = (
                self.replay_buffer.episode_ends[epi_idx - 1] if epi_idx > 0 else 0
            )
            epi_end = self.replay_buffer.episode_ends[epi_idx]
            val_horizon = self.val_horizon
            seq_end = min(epi_end, epi_start + val_horizon)
            sample = dict()
            for key in self.sampler.keys:
                sample[key] = self.replay_buffer[key][epi_start:seq_end]
                if sample[key].shape[0] < val_horizon:
                    pad_len = val_horizon - sample[key].shape[0]
                    pad_shape = (pad_len, *np.ones_like(sample[key].shape[1:]).tolist())
                    sample_pad = np.tile(sample[key][-1:], pad_shape)
                    sample[key] = np.concatenate([sample[key], sample_pad], axis=0)
                if key in self.sampler.keys_to_keep_intermediate:
                    inter_frames = sample[key].shape[0] // self.skip_frame
                    sample_shape = list(sample[key].shape[1:])
                    sample_shape[0] = sample_shape[0] * self.skip_frame
                    sample[key] = sample[key].reshape(
                        inter_frames, self.skip_frame, *sample[key].shape[1:]
                    )
                    sample[key] = sample[key].reshape(-1, *sample_shape)
                else:
                    sample[key] = sample[key][:: self.skip_frame]
                sample[f"{key}_final"] = sample[key][-1]
                sample["is_early_stop"] = False
                sample["rel_stop_idx"] = val_horizon - 1
        else:
            sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        return data


def test_sim_aloha_dataset() -> None:
    config_path = "configurations/dataset/sim_aloha_dataset.yaml"
    cfg = OmegaConf.load(config_path)
    # cfg.dataset_dir = "/media/yixuan/Extreme SSD/projects/diffusion-forcing/data/sim_aloha/single_arm_transfer_cube_0407_v3"  # noqa
    # cfg.dataset_dir = "/media/yixuan/Extreme SSD/projects/diffusion-forcing/data/sim_aloha/pusht_test_0414"  # noqa
    # cfg.dataset_dir = "/media/yixuan/Extreme SSD/projects/diffusion-forcing/data/sim_aloha/pusht_0407"  # noqa
    cfg.dataset_dir = "data/scripted_sim_aloha_10000"
    cfg.horizon = 10
    cfg.shape_meta.action.shape = (4,)
    cfg.skip_frame = 1
    cfg.skip_idx = 4
    cfg.val_horizon = 200
    cfg.goal_sample = "aggressive"
    cfg.resolution = 128
    dataset = SimAlohaDataset(cfg)
    print(len(dataset))

    p = psutil.Process(os.getpid())

    def rss() -> float:
        return p.memory_info().rss / 1e9

    print(f"START RSS: {rss():.3f} GB")
    k = min(2000, len(dataset))  # enough iterations to see creep
    for i in range(k):
        _ = dataset[i]  # exercises __getitem__
        if (i + 1) % 50 == 0:
            gc.collect()
            print(f"i={i+1} RSS={rss():.3f} GB")

    cpu_count = os.cpu_count()
    assert cpu_count is not None
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=16,
        num_workers=min(cpu_count, 16),
        shuffle=False,
        persistent_workers=False,
        pin_memory=False,
        prefetch_factor=1,
    )
    i = 0
    for _ in dataloader:
        i += 1
        if (i + 1) % 50 == 0:
            gc.collect()
            print(f"i={i+1} RSS={rss():.3f} GB")

    val_dataset = dataset.get_validation_dataset()
    print(len(val_dataset))
    for i in range(len(val_dataset)):
        data = val_dataset[i]
    print(data)
    print("validation dataset success!")


if __name__ == "__main__":
    test_sim_aloha_dataset()
