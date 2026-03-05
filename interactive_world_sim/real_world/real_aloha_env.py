import glob
import os
import pathlib
import shutil
import time
from multiprocessing.managers import SharedMemoryManager
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from yixuan_utilities.hdf5_utils import save_dict_to_hdf5
from yixuan_utilities.kinematics_helper import KinHelper

from interactive_world_sim.real_world.aloha_bimanual_puppet import AlohaBimanualPuppet
from interactive_world_sim.real_world.multi_camera_visualizer import (
    MultiCameraVisualizer,
)
from interactive_world_sim.real_world.multi_realsense import (
    MultiRealsense,
    SingleRealsense,
)
from interactive_world_sim.real_world.video_recorder import VideoRecorder
from interactive_world_sim.utils.aloha_conts import (
    START_ARM_POSE,
)
from interactive_world_sim.utils.cv2_util import get_image_transform, optimal_row_cols
from interactive_world_sim.utils.sync_utils import sync_timestamps
from interactive_world_sim.utils.timestamp_accumulator import (
    TimestampActionAccumulator,
)

# Default observation key mapping.
DEFAULT_OBS_KEY_MAP: Dict[str, str] = {
    # robot
    "curr_ee_pose": "ee_pos",
    "curr_joint_pos": "joint_pos",
    "curr_full_joint_pos": "full_joint_pos",
    "robot_base_pose_in_world": "world_t_robot_base",
    # timestamps
    "step_idx": "step_idx",
    "timestamp": "timestamp",
}


class RealAlohaEnv:
    """Real Aloha Environment for robot control and observation."""

    def __init__(
        self,
        # Required parameters
        output_dir: Union[str, pathlib.Path],
        # Environment parameters
        frequency: int = 10,
        n_obs_steps: int = 1,
        render_size: Tuple[int, int] = (128, 128),
        # Observation parameters
        obs_image_resolution: Tuple[int, int] = (640, 480),
        max_obs_buffer_size: int = 30,
        camera_serial_numbers: Optional[List[str]] = None,
        obs_key_map: Dict[str, str] = DEFAULT_OBS_KEY_MAP,
        obs_float32: bool = False,
        # Action parameters
        max_pos_speed: float = 0.25,
        max_rot_speed: float = 0.6,
        # Robot parameters
        robot_sides: List[str] = ["right"],  # noqa
        init_qpos: Optional[Any] = None,
        ctrl_mode: str = "joint",
        # Video capture parameters
        video_capture_fps: int = 30,
        video_capture_resolution: Tuple[int, int] = (1280, 720),
        # Saving parameters
        record_raw_video: bool = True,
        thread_per_video: int = 2,
        video_crf: int = 21,
        # Visualization parameters
        enable_multi_cam_vis: bool = True,
        multi_cam_vis_resolution: Tuple[int, int] = (1280, 720),
        # Shared memory manager (optionally provided)
        shm_manager: Optional[SharedMemoryManager] = None,
    ) -> None:
        # Ensure frequency is not higher than video_capture_fps.
        assert frequency <= video_capture_fps

        output_dir = pathlib.Path(output_dir)
        assert output_dir.parent.is_dir()
        video_dir: pathlib.Path = output_dir.joinpath("videos")
        if not video_dir.exists():
            video_dir.mkdir(parents=True, exist_ok=True)
        self.episode_id: int = len(
            glob.glob(os.path.join(output_dir.absolute().as_posix(), "*.hdf5"))
        )

        if init_qpos is None:
            init_qpos = np.array(START_ARM_POSE)

        if shm_manager is None:
            shm_manager = SharedMemoryManager()
            shm_manager.start()
        if camera_serial_numbers is None:
            camera_serial_numbers = SingleRealsense.get_connected_devices_serial()

        # Create a color transform for observation images.
        # get_image_transform returns a callable that transforms an image
        # (np.ndarray) to another.
        color_tf: Callable[[np.ndarray], np.ndarray] = get_image_transform(
            input_res=video_capture_resolution,
            output_res=obs_image_resolution,
            bgr_to_rgb=True,
        )
        color_transform: Callable[[np.ndarray], np.ndarray] = color_tf
        if obs_float32:
            # Normalize to [0, 1] float32 if requested.
            color_transform = lambda x: color_tf(x).astype(np.float32) / 255  # type: ignore

        # Define a transform function to process incoming data.
        def transform(data: Dict[str, Any]) -> Dict[str, Any]:
            data["color"] = color_transform(data["color"])
            if "depth" in data:
                data["depth"] = cv2.resize(
                    data["depth"], obs_image_resolution, interpolation=cv2.INTER_NEAREST
                )
            return data

        # Determine optimal visualization grid settings.
        rw, rh, col, row = optimal_row_cols(
            n_cameras=len(camera_serial_numbers),
            in_wh_ratio=obs_image_resolution[0] / obs_image_resolution[1],
            max_resolution=multi_cam_vis_resolution,
        )

        # Determine recording transform and settings.
        recording_transfrom: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None
        recording_fps: int = video_capture_fps
        recording_pix_fmt: str = "bgr24"
        if not record_raw_video:
            recording_transfrom = transform
            recording_fps = frequency
            recording_pix_fmt = "rgb24"

        video_recorder: VideoRecorder = VideoRecorder(
            fps=recording_fps,
            codec="mp4v",
            input_pix_fmt=recording_pix_fmt,
        )

        realsense: MultiRealsense = MultiRealsense(
            serial_numbers=camera_serial_numbers,
            shm_manager=shm_manager,
            resolution=video_capture_resolution,
            capture_fps=video_capture_fps,
            put_fps=video_capture_fps,
            put_downsample=False,
            record_fps=recording_fps,
            enable_color=True,
            enable_depth=False,
            enable_infrared=False,
            get_max_k=max_obs_buffer_size,
            transform=transform,
            vis_transform=None,
            recording_transform=recording_transfrom,
            video_recorder=video_recorder,
            verbose=False,
        )
        realsense.set_exposure(exposure=60, gain=64)
        realsense.set_white_balance(white_balance=3800)
        realsense.set_depth_preset("High Density")
        realsense.set_depth_exposure(7000, 16)

        multi_cam_vis: Optional[MultiCameraVisualizer] = None
        if enable_multi_cam_vis:
            multi_cam_vis = MultiCameraVisualizer(
                realsense=realsense, row=row, col=col, rgb_to_bgr=False
            )

        if ctrl_mode == "joint":
            action_dim = 7 * len(robot_sides)
        elif ctrl_mode == "bimanual_control":
            action_dim = 14  # 7 joints for each arm
        elif ctrl_mode == "bimanual_push":
            action_dim = 4
        elif ctrl_mode == "single_ee":
            action_dim = 4
        elif ctrl_mode == "single_push":
            action_dim = 2  # Only x,y position for single arm pushing
        elif ctrl_mode == "single_sweep":
            action_dim = 4
        elif ctrl_mode == "single_wipe":
            action_dim = 4
        elif ctrl_mode == "single_rope":
            action_dim = 5
        elif ctrl_mode == "bimanual_sweep":
            action_dim = 4
        elif ctrl_mode == "bimanual_sweep_v2":
            action_dim = 4
        elif ctrl_mode == "single_grasp":
            action_dim = 4
        elif ctrl_mode == "bimanual_pack":
            action_dim = 6
        elif ctrl_mode == "bimanual_rope":
            action_dim = 8
        elif ctrl_mode == "single_chain_in_box":
            action_dim = 4
        elif ctrl_mode == "bimanual_box":
            action_dim = 14
        else:
            raise ValueError("Invalid control mode.")
        self.puppet_bot = AlohaBimanualPuppet(
            shm_manager=shm_manager,
            frequency=50,
            robot_sides=robot_sides,
            verbose=False,
            init_qpos=init_qpos,
            ctrl_mode=ctrl_mode,
            action_dim=action_dim,
        )

        self.realsense: MultiRealsense = realsense
        self.video_capture_resolution: Tuple[int, int] = video_capture_resolution
        self.multi_cam_vis: Optional[MultiCameraVisualizer] = multi_cam_vis
        self.video_capture_fps: int = video_capture_fps
        self.frequency: int = frequency
        self.render_size: Tuple[int, int] = render_size
        self.n_obs_steps: int = n_obs_steps
        self.max_obs_buffer_size: int = max_obs_buffer_size
        self.max_pos_speed: float = max_pos_speed
        self.max_rot_speed: float = max_rot_speed
        self.obs_key_map: Dict[str, str] = obs_key_map
        self.output_dir: pathlib.Path = output_dir
        self.video_dir: pathlib.Path = video_dir
        self.last_realsense_data: Dict[str, Any] = {}
        # Recording buffers/accumulators.
        self.obs_hist: Any = None
        self.rob_state_hist: Any = None
        self.joint_action_accumulator: Optional[TimestampActionAccumulator] = None
        self.action_accumulator: Optional[TimestampActionAccumulator] = None
        self.init_qpos: np.ndarray = init_qpos
        self.ctrl_mode: str = ctrl_mode
        self.robot_sides: List[str] = robot_sides
        self.kin_helper = KinHelper("trossen_vx300s")
        self.obs_image_resolution = obs_image_resolution

        self.start_time: Optional[float] = None

    @property
    def is_ready(self) -> bool:
        """Check if the environment is ready for operation."""
        return self.realsense.is_ready and self.puppet_bot.is_ready

    def start(self, wait: bool = True) -> None:
        """Start the environment, initializing all components."""
        self.realsense.start(wait=False)
        self.puppet_bot.start(wait=False)
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.start(wait=False)
        if wait:
            self.start_wait()

    def stop(self, wait: bool = True) -> None:
        """Stop the environment, cleaning up all components."""
        self.end_episode()
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.stop(wait=False)
        self.puppet_bot.stop(wait=False)
        self.realsense.stop(wait=False)
        if wait:
            self.stop_wait()

    def start_wait(self) -> None:
        """Start the environment and wait for all components to be ready."""
        self.realsense.start_wait()
        self.puppet_bot.start_wait()
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.start_wait()

    def stop_wait(self) -> None:
        """Stop the environment and wait for all components to finish."""
        self.puppet_bot.stop_wait()
        self.realsense.stop_wait()
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.stop_wait()

    def __enter__(self) -> "RealAlohaEnv":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()

    def get_obs_async(self, last_k: int = 1) -> Dict[str, Any]:
        """Returns the observation dictionary.

        It aligns the observation timestamps between the camera and robot.
        """
        assert self.is_ready
        start_obs_time = time.time()

        k = int(self.n_obs_steps * self.video_capture_fps / self.frequency)
        all_rs_data = self.realsense.get(k)

        # filter out weird data
        rs_keys = list(all_rs_data.keys())
        for key in rs_keys:
            if len(all_rs_data[key]["color"]) == 0:
                all_rs_data.pop(key)
        # Use the maximum timestamp across all cameras.
        rs_t = np.max([x["timestamp"] for x in all_rs_data.values()], axis=0)

        get_robot_time = time.time()
        last_robot_data: Dict[str, np.ndarray] = self.puppet_bot.get_all_state()
        robot_t = last_robot_data["robot_receive_timestamp"]
        get_robot_end_time = time.time()

        # robot_idx, rs_idx = sync_timestamps(robot_t, rs_t)

        # obs_data: Dict[str, Any] = {}
        # obs_data["timestamp"] = rs_t[rs_idx]
        # for camera_idx, value in all_rs_data.items():
        #     obs_data[f"camera_{camera_idx}_color"] = value["color"][rs_idx]
        #     obs_data[f"camera_{camera_idx}_depth"] = value["depth"][rs_idx]
        #     obs_data[f"camera_{camera_idx}_intrinsics"] = value["intrinsics"][rs_idx]
        #     obs_data[f"camera_{camera_idx}_extrinsics"] = value["extrinsics"][rs_idx]

        put_time = time.time()
        if self.obs_hist is not None:
            if "timestamp" not in self.obs_hist:
                self.obs_hist["timestamp"] = []
            self.obs_hist["timestamp"].append(rs_t)
            for camera_idx, value in all_rs_data.items():
                if f"camera_{camera_idx}_color" not in self.obs_hist:
                    self.obs_hist[f"camera_{camera_idx}_color"] = []
                    if "depth" in value:
                        self.obs_hist[f"camera_{camera_idx}_depth"] = []
                    self.obs_hist[f"camera_{camera_idx}_intrinsics"] = []
                    self.obs_hist[f"camera_{camera_idx}_extrinsics"] = []
                self.obs_hist[f"camera_{camera_idx}_color"].append(value["color"])
                if "depth" in value:
                    self.obs_hist[f"camera_{camera_idx}_depth"].append(value["depth"])
                self.obs_hist[f"camera_{camera_idx}_intrinsics"].append(
                    value["intrinsics"]
                )
                self.obs_hist[f"camera_{camera_idx}_extrinsics"].append(
                    value["extrinsics"]
                )

            if "robot_receive_timestamp" not in self.rob_state_hist:
                self.rob_state_hist["robot_receive_timestamp"] = []
            self.rob_state_hist["robot_receive_timestamp"].append(robot_t)
            for k_key, v in last_robot_data.items():
                if k_key in self.obs_key_map:
                    if self.obs_key_map[k_key] not in self.rob_state_hist:
                        self.rob_state_hist[self.obs_key_map[k_key]] = []
                    self.rob_state_hist[self.obs_key_map[k_key]].append(v)
        put_end_time = time.time()

        # for k_key, v in last_robot_data.items():
        #     if k_key in self.obs_key_map:
        #         obs_data[self.obs_key_map[k_key]] = v[robot_idx]

        # put_time = time.time()
        # if self.obs_accumulator is not None:
        #     self.obs_accumulator.put(
        #         obs_data,
        #         rs_t[rs_idx],
        #     )
        #     self.rob_state_accumulator.put(
        #         last_robot_data,
        #         robot_t[robot_idx],
        #     )
        # put_end_time = time.time()

        obs_data: Dict[str, Any] = {}
        for camera_idx, value in all_rs_data.items():
            obs_data[f"camera_{camera_idx}_color"] = value["color"][-last_k:]
        obs_data.update(last_robot_data)

        end_obs_time = time.time()
        if end_obs_time - start_obs_time > 0.2:
            print(
                f"get_obs_async time: {end_obs_time - start_obs_time:.3f} "
                f"get_rs_time: {get_robot_time - start_obs_time:.3f} "
                f"get_robot_time: {get_robot_end_time - get_robot_time:.3f} "
                f"put time: {put_end_time - put_time:.3f}"
            )
        return obs_data

    def exec_actions(
        self,
        joint_actions: np.ndarray,
        actions: np.ndarray,
        timestamps: np.ndarray,
    ) -> None:
        """Execute a batch of actions based on the provided control mode.

        The joint actions (or end-effector actions) are scheduled with timestamps.
        """
        assert self.is_ready

        # Filter for actions whose timestamps are in the future.
        receive_time: float = time.time()
        is_new: np.ndarray = timestamps > receive_time
        new_joint_actions: np.ndarray = joint_actions[is_new]
        new_actions: np.ndarray = actions[is_new]
        new_timestamps: np.ndarray = timestamps[is_new]
        if len(new_joint_actions) == 0:
            return

        for i in range(len(new_actions)):
            self.puppet_bot.set_actions(new_actions[i], new_timestamps[i])

        if self.joint_action_accumulator is not None:
            self.joint_action_accumulator.put(new_joint_actions, new_timestamps)
        if self.action_accumulator is not None:
            self.action_accumulator.put(new_actions, new_timestamps)

    def get_robot_state(self) -> Dict[str, Any]:
        """Get the current state of the robot."""
        return self.puppet_bot.get_state()

    def init_episode_data(
        self, num_cam: int, cam_height: int, cam_width: int
    ) -> Tuple[Dict, Dict, Dict]:
        """Initialize the episode data structure and configuration."""
        episode: Dict[str, Any] = {
            "timestamp": None,
            "obs": {
                "joint_pos": [],
                "full_joint_pos": [],
                "world_t_robot_base": [],
                "ee_pos": [],
                "images": {},
            },
            "joint_action": [],
            "action": [],
        }
        for cam in range(num_cam):
            episode["obs"]["images"][f"camera_{cam}_color"] = []
            if self.realsense.enable_depth:
                episode["obs"]["images"][f"camera_{cam}_depth"] = []
            episode["obs"]["images"][f"camera_{cam}_intrinsics"] = []
            episode["obs"]["images"][f"camera_{cam}_extrinsics"] = []
        attr_dict: Dict[str, Any] = {"sim": True}
        config_dict: Dict[str, Any] = {
            "obs": {"images": {}},
            "timestamp": {"dtype": "float64"},
        }
        for cam in range(num_cam):
            color_save_kwargs: Dict[str, Any] = {
                "chunks": (1, cam_height, cam_width, 3),
                # "compression": "gzip",
                # "compression_opts": 9,
                "dtype": "uint8",
            }
            depth_save_kwargs: Dict[str, Any] = {
                "chunks": (1, cam_height, cam_width),
                # "compression": "gzip",
                # "compression_opts": 9,
                "dtype": "uint16",
            }
            config_dict["obs"]["images"][f"camera_{cam}_color"] = color_save_kwargs
            config_dict["obs"]["images"][f"camera_{cam}_depth"] = depth_save_kwargs
        return episode, config_dict, attr_dict

    def get_render_size(self) -> Tuple[int, int]:
        """Get the render size of the environment."""
        return self.render_size

    def get_cam_intrinsic(self, obs_key: str, shape: tuple[int, int]) -> np.ndarray:
        """Get the camera intrinsic matrix."""
        K = self.realsense.get_intrinsics()
        obs_i = int(obs_key.split("_")[1])
        K_i = K[obs_i].reshape(3, 3)
        fx = K_i[0, 0]
        fy = K_i[1, 1]
        cx = K_i[0, 2]
        cy = K_i[1, 2]
        width, height = self.video_capture_resolution
        cx = cx * shape[1] / width
        cy = cy * shape[0] / height
        f_scale = max(shape[1] / width, shape[0] / height)
        fx = fx * f_scale
        fy = fy * f_scale
        return np.array([cx, cy, fx, fy])

    def get_cam_extrinsic(self, obs_key: str) -> np.ndarray:
        """Get the camera extrinsic matrix."""
        extrinsics = self.realsense.get_extrinsics()
        obs_i = int(obs_key.split("_")[1])
        return extrinsics[obs_i].reshape(4, 4)

    def get_robot_bases(self) -> np.ndarray:
        """Get the robot base positions in world coordinates."""
        return self.puppet_bot.base_pose_in_world

    def get_observations(self) -> Dict[str, Any]:
        """Get the current observation of the environment."""
        raise NotImplementedError("needed for vis only")

    def start_episode(
        self,
        start_time: Optional[float] = None,
        curr_outdir: Optional[Union[str, pathlib.Path]] = None,
    ) -> None:
        """Start recording an episode and return the first observation."""
        if start_time is None:
            start_time = time.time()
        self.start_time = start_time

        assert self.is_ready

        if curr_outdir is None:
            this_video_dir: pathlib.Path = self.video_dir.joinpath(str(self.episode_id))
        else:
            curr_outdir = pathlib.Path(curr_outdir)
            video_dir_local: pathlib.Path = curr_outdir.joinpath("videos")
            video_dir_local.mkdir(parents=True, exist_ok=True)
            this_video_dir = video_dir_local.joinpath(str(self.episode_id))
        this_video_dir.mkdir(parents=True, exist_ok=True)
        n_cameras: int = self.realsense.n_cameras
        video_paths: List[str] = []
        for i in range(n_cameras):
            video_paths.append(str(this_video_dir.joinpath(f"{i}.mp4").absolute()))

        self.realsense.restart_put(start_time=start_time)
        self.realsense.start_recording(
            video_path=video_paths[:n_cameras], start_time=start_time
        )
        assert self.multi_cam_vis is not None
        self.multi_cam_vis.start_recording(f"{this_video_dir}/multi_cam.mp4")

        # self.obs_accumulator = TimestampObsAccumulator(
        #     start_time=start_time, dt=1 / self.frequency
        # )
        # self.rob_state_accumulator = TimestampObsAccumulator(
        #     start_time=start_time, dt=1 / self.frequency
        # )
        self.obs_hist = {}
        self.rob_state_hist = {}
        self.joint_action_accumulator = TimestampActionAccumulator(
            start_time=start_time, dt=1 / self.frequency
        )
        self.action_accumulator = TimestampActionAccumulator(
            start_time=start_time, dt=1 / self.frequency
        )
        print(f"Episode {self.episode_id} started!")

    def end_episode(
        self,
        curr_outdir: Optional[Union[str, pathlib.Path]] = None,
        incr_epi: bool = True,
    ) -> None:
        """Stop recording the episode and save data."""
        assert self.is_ready

        self.realsense.stop_recording()
        assert self.multi_cam_vis is not None
        self.multi_cam_vis.stop_recording()

        if self.obs_hist is not None:
            assert self.joint_action_accumulator is not None
            assert self.action_accumulator is not None

            obs_timestamps: np.ndarray = np.concatenate(self.obs_hist["timestamp"])
            obs_data: Dict[str, np.ndarray] = {}
            for key in self.obs_hist.keys():
                if key != "timestamp":
                    obs_data[key] = np.concatenate(self.obs_hist[key])

            robot_t = np.concatenate(self.rob_state_hist["robot_receive_timestamp"])
            robot_state: Dict[str, np.ndarray] = {}
            for key in self.rob_state_hist.keys():
                if key != "robot_receive_timestamp":
                    robot_state[key] = np.concatenate(self.rob_state_hist[key])

            num_cam: int = 0
            cam_width: int = -1
            cam_height: int = -1
            for key in obs_data.keys():
                if "camera" in key and "color" in key:
                    num_cam += 1
                    cam_height, cam_width = obs_data[key].shape[1:3]

            joint_action: np.ndarray = self.joint_action_accumulator.actions
            actions: np.ndarray = self.action_accumulator.actions
            action_timestamps: np.ndarray = self.joint_action_accumulator.timestamps

            # remove the first two empty ones
            joint_action = joint_action[2:]
            actions = actions[2:]
            action_timestamps = action_timestamps[2:]

            # initialize the episode data
            episode, config_dict, attr_dict = self.init_episode_data(
                num_cam=num_cam,
                cam_height=cam_height,
                cam_width=cam_width,
            )

            # sync timestamps to action timestamps
            sync_obs_idx, sync_action_idx = sync_timestamps(
                obs_timestamps, action_timestamps
            )
            sync_robot_idx, sync_action_idx_2 = sync_timestamps(
                robot_t, action_timestamps
            )
            overlap_sync = np.intersect1d(sync_action_idx, sync_action_idx_2)
            sync_obs_idx = sync_obs_idx[np.isin(sync_action_idx, overlap_sync)]
            sync_action_idx = sync_action_idx[np.isin(sync_action_idx, overlap_sync)]
            sync_robot_idx = sync_robot_idx[np.isin(sync_action_idx_2, overlap_sync)]
            if len(sync_obs_idx) == 0:
                print("No valid obs found for this episode.")
                self.joint_action_accumulator = None
                self.action_accumulator = None
                self.obs_hist = None
                self.rob_state_hist = None
                return
            episode["timestamp"] = obs_timestamps[sync_obs_idx][:-1]
            episode["joint_action"] = joint_action[sync_action_idx][:-1]
            episode["action"] = actions[sync_action_idx][:-1]
            for key, value in obs_data.items():
                if "camera" in key:
                    episode["obs"]["images"][key] = value[sync_obs_idx][:-1]
                else:
                    episode["obs"][key] = value[sync_obs_idx][:-1]
            for key, value in robot_state.items():
                episode["obs"][key] = value[sync_robot_idx][:-1]

            if curr_outdir is None:
                episode_path: pathlib.Path = self.output_dir.joinpath(
                    f"episode_{self.episode_id}.hdf5"
                )
            else:
                self.curr_outdir = pathlib.Path(curr_outdir)
                episode_path = self.curr_outdir.joinpath(
                    f"episode_{self.episode_id}.hdf5"
                )
            save_dict_to_hdf5(
                episode, config_dict, str(episode_path), attr_dict=attr_dict
            )

            print(f"Episode {self.episode_id} saved!")

            self.obs_hist = None
            self.rob_state_hist = None
            self.joint_action_accumulator = None
            self.action_accumulator = None

            if incr_epi:
                self.episode_id += 1

    def reset(self) -> None:
        """Dummy reset"""

    def drop_episode(self) -> None:
        """Drop the current episode and clear all buffers."""
        # Stop recording and clear accumulated buffers.
        self.realsense.stop_recording()
        assert self.multi_cam_vis is not None
        self.multi_cam_vis.stop_recording()
        self.obs_hist = None
        self.rob_state_hist = None
        self.joint_action_accumulator = None
        self.action_accumulator = None

        this_video_dir: pathlib.Path = self.video_dir.joinpath(str(self.episode_id))
        if this_video_dir.exists():
            shutil.rmtree(str(this_video_dir))
        print(f"Episode {self.episode_id} dropped!")


# def test_episode_start() -> None:
#     """Create a temporary environment and start an episode."""
#     os.system("mkdir -p tmp")
#     with RealAlohaEnv(output_dir="tmp") as env:
#         print("Created env!")
#         env.start_episode()
#         print("Started episode!")


# def test_env_obs_latency() -> None:
#     os.system("mkdir -p tmp")
#     with RealAlohaEnv(output_dir="tmp") as env:
#         print("Created env!")
#         for _ in range(100):
#             start_time: float = time.time()
#             _ = env.get_obs()
#             end_time: float = time.time()
#             print(f"obs latency: {end_time - start_time}")
#             time.sleep(0.1)


# def test_env_demo_replay() -> None:
#     os.system("mkdir -p tmp")
#     demo_path: str = (
#         "/home/yixuan/general_dp/data/real_aloha_demo/open_bag/episode_0.hdf5"
#     )
#     robot_sides: List[str] = ["right", "left"]
#     demo_dict, _ = load_dict_from_hdf5(demo_path)
#     with RealAlohaEnv(output_dir="tmp", robot_sides=robot_sides) as env:
#         print("Created env!")
#         timestamps: np.ndarray = (
#             time.time() + np.arange(len(demo_dict["cartesian_action"])) / 10 + 1.0
#         )
#         ik_init: List[Any] = [demo_dict["obs"]["full_joint_pos"][0]] * len(
#             demo_dict["cartesian_action"]
#         )
#         print(demo_dict["obs"]["full_joint_pos"][()])
#         start_step: int = 0
#         while True:
#             curr_time: float = time.monotonic()
#             loop_end_time: float = curr_time + 1.0
#             end_step: int = min(start_step + 10, len(demo_dict["cartesian_action"]))
#             action_batch: np.ndarray = demo_dict["cartesian_action"][
#                 start_step:end_step
#             ]
#             timestamp_batch: np.ndarray = timestamps[start_step:end_step]
#             ik_init_batch: Any = ik_init[start_step:end_step]
#             env.exec_actions(
#                 joint_actions=np.zeros((action_batch.shape[0], 7)),
#                 eef_actions=action_batch,
#                 timestamps=timestamp_batch,
#                 mode="eef",
#                 ik_init=ik_init_batch,
#             )
#             print(f"executed {end_step - start_step} actions")
#             start_step = end_step
#             precise_wait(loop_end_time)
#             if start_step >= len(demo_dict["cartesian_action"]):
#                 break


# def test_cache_replay() -> None:
#     os.system("mkdir -p tmp")
#     demo_path: str = (
#         "/home/yixuan/general_dp/data/real_aloha_demo/open_bag_demo_1/episode_0.hdf5"
#     )
#     demo_dict, _ = load_dict_from_hdf5(demo_path)

#     cache_path: str = (
#         "/home/yixuan/general_dp/data/real_aloha_demo/open_bag_demo_1\
#           /cache_no_seg_dino.zarr.zip"
#     )
#     robot_sides: List[str] = ["right", "left"]
#     with zarr.ZipStore(cache_path, mode="r") as zip_store:
#         replay_buffer: ReplayBuffer = ReplayBuffer.copy_from_store(
#             src_store=zip_store, store=zarr.MemoryStore()
#         )
#     actions: np.ndarray = replay_buffer["action"][()]
#     with RealAlohaEnv(output_dir="tmp", robot_sides=robot_sides) as env:
#         print("Created env!")
#         timestamps: np.ndarray = time.time() + np.arange(len(actions)) / 10 + 1.0
#         ik_init: List[Any] = [demo_dict["obs"]["full_joint_pos"][0]] * len(
#             actions
#         )
#         print(demo_dict["obs"]["full_joint_pos"][()])
#         # Convert action from rotation 6D to Euler angles.
#         actions_reshape: np.ndarray = actions.reshape(
#             actions.shape[0] * len(robot_sides), 10
#         )
#         action_pos: np.ndarray = actions_reshape[:, :3]
#         action_rot_6d: np.ndarray = actions_reshape[:, 3:9]
#         action_rot_mat: np.ndarray = pytorch3d.transforms.rotation_6d_to_matrix(
#             torch.from_numpy(action_rot_6d)
#         ).numpy()
#         action_rot_euler = st.Rotation.from_matrix(action_rot_mat).as_euler(
#             "xyz"
#         )
#         actions_reshape = np.concatenate(
#             [action_pos, action_rot_euler, actions_reshape[:, -1:]], axis=-1
#         )
#         actions = actions_reshape.reshape(actions.shape[0], len(robot_sides) * 7)

#         start_step: int = 0
#         while True:
#             curr_time = time.monotonic()
#             loop_end_time: float = curr_time + 1.0
#             end_step: int = min(start_step + 10, len(actions))
#             action_batch: np.ndarray = actions[start_step:end_step]
#             timestamp_batch: np.ndarray = timestamps[start_step:end_step]
#             ik_init_batch: Any = ik_init[start_step:end_step]
#             env.exec_actions(
#                 joint_actions=np.zeros((action_batch.shape[0], 7)),
#                 actions=action_batch,
#                 timestamps=timestamp_batch,
#             )
#             print(f"executed {end_step - start_step} actions")
#             start_step = end_step
#             precise_wait(loop_end_time)
#             if start_step >= len(actions):
#                 break


# if __name__ == "__main__":
#     # Uncomment one of the tests below to run.
#     # test_env_obs_latency()
#     # test_env_demo_replay()
#     test_cache_replay()
