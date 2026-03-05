import numbers
import pathlib
import time
from multiprocessing.managers import SharedMemoryManager
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
import pyrealsense2 as rs

from interactive_world_sim.real_world.single_realsense import SingleRealsense
from interactive_world_sim.real_world.video_recorder import VideoRecorder

T = TypeVar("T")


class MultiRealsense:
    """MultiRealsense class for handling multiple RealSense cameras."""

    def __init__(
        self,
        serial_numbers: Optional[List[str]] = None,
        shm_manager: Optional[SharedMemoryManager] = None,
        resolution: Tuple[int, int] = (1280, 720),
        capture_fps: int = 30,
        put_fps: Optional[int] = None,
        put_downsample: bool = True,
        record_fps: Optional[int] = None,
        enable_color: bool = True,
        enable_depth: bool = False,
        enable_infrared: bool = False,
        get_max_k: int = 30,
        advanced_mode_config: Optional[Union[dict, List[dict]]] = None,
        transform: Optional[
            Union[
                Callable[[Dict[Any, Any]], Dict[Any, Any]],
                List[Callable[[Dict[Any, Any]], Dict[Any, Any]]],
            ]
        ] = None,
        vis_transform: Optional[
            Union[
                Callable[[Dict[Any, Any]], Dict[Any, Any]],
                List[Callable[[Dict[Any, Any]], Dict[Any, Any]]],
            ]
        ] = None,
        recording_transform: Optional[
            Union[
                Callable[[Dict[Any, Any]], Dict[Any, Any]],
                List[Callable[[Dict[Any, Any]], Dict[Any, Any]]],
            ]
        ] = None,
        video_recorder: Optional[Union[VideoRecorder, List[VideoRecorder]]] = None,
        verbose: bool = False,
    ) -> None:
        if shm_manager is None:
            shm_manager = SharedMemoryManager()
            shm_manager.start()
        if serial_numbers is None:
            serial_numbers = SingleRealsense.get_connected_devices_serial()
        n_cameras: int = len(serial_numbers)

        advanced_mode_config = repeat_to_list(advanced_mode_config, n_cameras, dict)
        transform = repeat_to_list(transform, n_cameras, Callable)
        vis_transform = repeat_to_list(vis_transform, n_cameras, Callable)
        recording_transform = repeat_to_list(recording_transform, n_cameras, Callable)

        video_recorder = repeat_to_list(video_recorder, n_cameras, VideoRecorder)

        cameras: Dict[str, SingleRealsense] = {}
        for i, serial in enumerate(serial_numbers):
            cameras[serial] = SingleRealsense(
                shm_manager=shm_manager,
                serial_number=serial,
                resolution=resolution,
                capture_fps=capture_fps,
                put_fps=put_fps,
                put_downsample=put_downsample,
                record_fps=record_fps,
                enable_color=enable_color,
                enable_depth=enable_depth,
                enable_infrared=enable_infrared,
                get_max_k=get_max_k,
                advanced_mode_config=advanced_mode_config[i],
                transform=transform[i],
                vis_transform=vis_transform[i],
                recording_transform=recording_transform[i],
                video_recorder=video_recorder[i],
                verbose=verbose,
            )

        self.cameras: Dict[str, SingleRealsense] = cameras
        self.enable_depth: bool = enable_depth
        self.shm_manager: SharedMemoryManager = shm_manager

    def __enter__(self) -> "MultiRealsense":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()

    @property
    def n_cameras(self) -> int:
        """Return the number of cameras."""
        return len(self.cameras)

    @property
    def is_ready(self) -> bool:
        """Return True if all cameras are ready, else False."""
        is_ready_val: bool = True
        for camera in self.cameras.values():
            if not camera.is_ready:
                is_ready_val = False
        return is_ready_val

    def start(self, wait: bool = True, put_start_time: Optional[float] = None) -> None:
        """Start all cameras."""
        if put_start_time is None:
            put_start_time = time.time()
        for camera in self.cameras.values():
            camera.start(wait=False, put_start_time=put_start_time)

        if wait:
            self.start_wait()

    def stop(self, wait: bool = True) -> None:
        """Stop all cameras."""
        for camera in self.cameras.values():
            camera.stop(wait=False)

        if wait:
            self.stop_wait()

    def start_wait(self) -> None:
        """Start all cameras and wait for them to finish."""
        for camera in self.cameras.values():
            camera.start_wait()

    def stop_wait(self) -> None:
        """Stop all cameras and wait for them to finish."""
        for camera in self.cameras.values():
            camera.join()

    def get(
        self, k: Optional[int] = None, out: Optional[Any] = None
    ) -> Dict[int, Dict[str, np.ndarray]]:
        """Return data with order T,H,W,C in a dictionary

        {
            0: {
                'rgb': (T,H,W,C),
                'timestamp': (T,)
            },
            1: ...
        }
        """
        if out is None:
            out = dict()
        for i, camera in enumerate(self.cameras.values()):
            this_out = None
            if i in out:
                this_out = out[i]
            this_out = camera.get(k=k, out=this_out)
            out[i] = this_out
        return out

    def get_all(self) -> Dict:
        """Return all data with order T,H,W,C in a dictionary"""
        out = dict()
        for i, camera in enumerate(self.cameras.values()):
            this_out = camera.get_all()
            out[i] = this_out
        return out

    def get_vis(self, out: Optional[Dict] = None) -> Dict[str, np.ndarray]:
        """Return visual data with order T,H,W,C in a dictionary

        {
            'rgb': (T,H,W,C),
            'timestamp': (T,)
        }
        """
        results = list()
        for i, camera in enumerate(self.cameras.values()):
            this_out = None
            if out is not None:
                this_out = dict()
                for key, v in out.items():
                    # use the slicing trick to maintain the array
                    # when v is 1D
                    this_out[key] = v[i : i + 1].reshape(v.shape[1:])
            this_out = camera.get(out=this_out)
            if out is None:
                results.append(this_out)
        if out is None:
            out = dict()
            for key in results[0].keys():
                out[key] = np.stack([x[key] for x in results])
        return out

    def set_color_option(self, option: Any, value: float) -> None:
        """Set color option for all cameras."""
        n_camera: int = len(self.cameras)
        value_ls = repeat_to_list(value, n_camera, numbers.Number)
        for i, camera in enumerate(self.cameras.values()):
            camera.set_color_option(option, value_ls[i])

    def set_depth_option(self, option: Any, value: float) -> None:
        """Set depth option for all cameras."""
        n_camera: int = len(self.cameras)
        value_ls = repeat_to_list(value, n_camera, numbers.Number)
        for i, camera in enumerate(self.cameras.values()):
            camera.set_depth_option(option, value_ls[i])

    def set_depth_preset(self, preset: Union[str, List[str]]) -> None:
        """Set depth preset for all cameras."""
        n_camera: int = len(self.cameras)
        preset = repeat_to_list(preset, n_camera, str)
        for i, camera in enumerate(self.cameras.values()):
            camera.set_depth_preset(preset[i])

    def set_exposure(
        self,
        exposure: Optional[float] = None,
        gain: Optional[float] = None,
    ) -> None:
        """Set exposure and gain for all cameras.

        exposure: (1, 10000) 100us unit. (0.1 ms, 1/10000s)
        gain: (0, 128)
        """
        if exposure is None and gain is None:
            # auto exposure
            self.set_color_option(rs.option.enable_auto_exposure, 1.0)
        else:
            # manual exposure
            self.set_color_option(rs.option.enable_auto_exposure, 0.0)
            if exposure is not None:
                self.set_color_option(rs.option.exposure, exposure)
            if gain is not None:
                self.set_color_option(rs.option.gain, gain)

    def set_depth_exposure(
        self,
        exposure: Optional[float] = None,
        gain: Optional[float] = None,
    ) -> None:
        """Set depth exposure and gain for all cameras.

        exposure: (1, 10000) 100us unit. (0.1 ms, 1/10000s)
        gain: (0, 128)
        """
        if exposure is None and gain is None:
            # auto exposure
            self.set_depth_option(rs.option.enable_auto_exposure, 1.0)
        else:
            # manual exposure
            self.set_depth_option(rs.option.enable_auto_exposure, 0.0)
            if exposure is not None:
                self.set_depth_option(rs.option.exposure, exposure)
            if gain is not None:
                self.set_depth_option(rs.option.gain, gain)

    def set_white_balance(self, white_balance: Optional[int] = None) -> None:
        """Set white balance for all cameras."""
        if white_balance is None:
            self.set_color_option(rs.option.enable_auto_white_balance, 1.0)
        else:
            self.set_color_option(rs.option.enable_auto_white_balance, 0.0)
            assert white_balance is not None
            self.set_color_option(rs.option.white_balance, white_balance)

    def get_intrinsics(self) -> np.ndarray:
        """Return intrinsics for all cameras."""
        return np.array([c.get_intrinsics() for c in self.cameras.values()])

    def get_extrinsics(self) -> np.ndarray:
        """Return extrinsics for all cameras."""
        return np.array([c.get_extrinsics() for c in self.cameras.values()])

    def get_depth_scale(self) -> np.ndarray:
        """Return depth scale for all cameras."""
        return np.array([c.get_depth_scale() for c in self.cameras.values()])

    def start_recording(
        self, video_path: Union[str, List[str]], start_time: float
    ) -> None:
        """Start recording for all cameras."""
        if isinstance(video_path, str):
            # Treat video_path as a directory.
            video_dir: pathlib.Path = pathlib.Path(video_path)
            assert video_dir.parent.is_dir()
            video_dir.mkdir(parents=True, exist_ok=True)
            video_path_list: List[str] = []
            for i in range(self.n_cameras):
                video_path_list.append(str(video_dir.joinpath(f"{i}.mp4").absolute()))
            video_path = video_path_list
        assert len(video_path) == self.n_cameras

        for i, camera in enumerate(self.cameras.values()):
            camera.start_recording(video_path[i], start_time)

    def stop_recording(self) -> None:
        """Stop recording for all cameras."""
        for camera in self.cameras.values():
            camera.stop_recording()

    def restart_put(self, start_time: float) -> None:
        """Restart the put process for all cameras."""
        for camera in self.cameras.values():
            camera.restart_put(start_time)

    def calibrate_extrinsics(
        self,
        visualize: bool = True,
        board_size: Tuple[int, int] = (6, 9),
        squareLength: float = 0.03,
        markerLength: float = 0.022,
    ) -> None:
        """Calibrate extrinsics for all cameras."""
        for camera in self.cameras.values():
            camera.calibrate_extrinsics(
                visualize=visualize,
                board_size=board_size,
                squareLength=squareLength,
                markerLength=markerLength,
            )


def repeat_to_list(x: Any, n: int, cls: Any) -> List:
    if x is None:
        x = [None] * n
    if isinstance(x, cls):
        x = [x] * n
    assert len(x) == n
    return x
