import enum
import json
import multiprocessing as mp
import os
import time
import warnings
from multiprocessing.managers import SharedMemoryManager
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import cv2
import numpy as np
import pyrealsense2 as rs
from threadpoolctl import threadpool_limits

from interactive_world_sim.real_world.video_recorder import VideoRecorder
from interactive_world_sim.utils.shared_memory_queue import Empty, SharedMemoryQueue
from interactive_world_sim.utils.shared_memory_ring_buffer import SharedMemoryRingBuffer
from interactive_world_sim.utils.shared_ndarray import SharedNDArray
from interactive_world_sim.utils.timestamp_accumulator import (
    get_accumulate_timestamp_idxs,
)


class Command(enum.Enum):
    """Command enumeration for controlling the RealSense device."""

    SET_COLOR_OPTION = 0
    SET_DEPTH_OPTION = 1
    START_RECORDING = 2
    STOP_RECORDING = 3
    RESTART_PUT = 4


class SingleRealsense(mp.Process):
    """A multiprocessing Process that captures data from a RealSense device.

    Optionally stores frames and depth data to shared memory buffers, and handles
    commands for adjusting sensor options or starting/stopping video recording.
    """

    MAX_PATH_LENGTH: int = 4096  # Linux path has a limit of 4096 bytes

    def __init__(
        self,
        shm_manager: SharedMemoryManager,
        serial_number: str,
        resolution: Tuple[int, int] = (1280, 720),
        capture_fps: int = 30,
        put_fps: Optional[int] = None,
        put_downsample: bool = True,
        record_fps: Optional[int] = None,
        enable_color: bool = True,
        enable_depth: bool = False,
        enable_infrared: bool = False,
        get_max_k: int = 30,
        advanced_mode_config: Optional[Dict[str, Any]] = None,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        vis_transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        recording_transform: Optional[
            Callable[[Dict[str, Any]], Dict[str, Any]]
        ] = None,
        video_recorder: Optional[VideoRecorder] = None,
        verbose: bool = False,
        extrinsics_dir: str = os.path.join(os.path.dirname(__file__), "cam_extrinsics"),
    ) -> None:
        """Initialize a SingleRealsense instance.

        Args:
            shm_manager: SharedMemoryManager instance.
            serial_number: The device serial number.
            resolution: Desired resolution as (width, height).
            capture_fps: Capture frames per second.
            put_fps: FPS for shared memory put operations.
            put_downsample: Whether to downsample frames for shared memory.
            record_fps: FPS for video recording.
            enable_color: Whether to enable color stream.
            enable_depth: Whether to enable depth stream.
            enable_infrared: Whether to enable infrared stream.
            get_max_k: Maximum buffer size for shared memory ring buffer.
            advanced_mode_config: Advanced configuration for the device.
            transform: A transformation function for the data.
            vis_transform: A visualization transformation function.
            recording_transform: A transformation for video recording.
            video_recorder: An optional VideoRecorder instance.
            verbose: Enable verbose logging.
            extrinsics_dir: Directory to load extrinsic calibration.
        """
        super().__init__()
        if put_fps is None:
            put_fps = capture_fps
        if record_fps is None:
            record_fps = capture_fps

        # Create ring buffer(s)
        shape = resolution[::-1]
        examples: Dict[str, Union[np.ndarray, float]] = {}
        if enable_color:
            examples["color"] = np.empty(shape=(*shape, 3), dtype=np.uint8)
        if enable_depth:
            examples["depth"] = np.empty(shape=shape, dtype=np.uint16)
        examples["intrinsics"] = np.empty(shape=(3, 3), dtype=np.float32)
        examples["extrinsics"] = np.empty(shape=(4, 4), dtype=np.float32)
        os.system(f"mkdir -p {extrinsics_dir}")
        if extrinsics_dir is None:
            self.extrinsics: np.ndarray = np.eye(4)
            warnings.warn(
                "extrinsics_dir is None, using identity matrix.",
                stacklevel=2,
            )
        else:
            extrinsics_path: str = os.path.join(extrinsics_dir, f"{serial_number}.npy")
            if not os.path.exists(extrinsics_path):
                self.extrinsics = np.eye(4)
                warnings.warn(
                    f"extrinsics_path {extrinsics_path} does not exist, using "
                    "identity matrix.",
                    stacklevel=2,
                )
            else:
                self.extrinsics = np.load(
                    os.path.join(extrinsics_dir, f"{serial_number}.npy")
                )
        if enable_infrared:
            examples["infrared"] = np.empty(shape=shape, dtype=np.uint8)
        examples["camera_capture_timestamp"] = 0.0
        examples["camera_receive_timestamp"] = 0.0
        examples["timestamp"] = 0.0
        examples["step_idx"] = 0

        # Create ring buffer for main data.
        ring_buffer_examples = (
            examples if transform is None else transform(dict(examples))
        )
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=ring_buffer_examples,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=put_fps,
        )

        # Create command queue.
        cmd_examples: Dict[str, Union[int, float, np.ndarray]] = {
            "cmd": Command.SET_COLOR_OPTION.value,
            "option_enum": rs.option.exposure.value,
            "option_value": 0.0,
            "video_path": np.array("a" * self.MAX_PATH_LENGTH),
            "recording_start_time": 0.0,
            "put_start_time": 0.0,
        }
        command_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager, examples=cmd_examples, buffer_size=128
        )

        # Create shared array for intrinsics.
        intrinsics_array = SharedNDArray.create_from_shape(
            mem_mgr=shm_manager, shape=(7,), dtype=np.float64
        )
        intrinsics_array.get()[:] = 0
        dist_coeff_array = SharedNDArray.create_from_shape(
            mem_mgr=shm_manager, shape=(5,), dtype=np.float64
        )
        dist_coeff_array.get()[:] = 0

        # Create video recorder if none is provided.
        if video_recorder is None:
            video_recorder = VideoRecorder.create_h264(
                fps=record_fps,
                codec="h264",
                input_pix_fmt="bgr24",
                crf=18,
                thread_type="FRAME",
                thread_count=1,
            )

        # Store parameters.
        self.serial_number: str = serial_number
        self.resolution: Tuple[int, int] = resolution
        self.capture_fps: int = capture_fps
        self.put_fps: int = put_fps
        self.put_downsample: bool = put_downsample
        self.record_fps: int = record_fps
        self.enable_color: bool = enable_color
        self.enable_depth: bool = enable_depth
        self.enable_infrared: bool = enable_infrared
        self.advanced_mode_config: Optional[Dict[str, Any]] = advanced_mode_config
        self.transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = transform
        self.vis_transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = (
            vis_transform
        )
        self.recording_transform: Optional[
            Callable[[Dict[str, Any]], Dict[str, Any]]
        ] = recording_transform
        self.video_recorder: VideoRecorder = video_recorder
        self.verbose: bool = verbose
        self.put_start_time: Optional[float] = None
        self.extrinsics_dir: str = extrinsics_dir

        # Shared variables.
        self.stop_event = mp.Event()
        self.ready_event = mp.Event()
        self.ring_buffer: SharedMemoryRingBuffer = ring_buffer
        self.command_queue: SharedMemoryQueue = command_queue
        self.intrinsics_array: SharedNDArray = intrinsics_array
        self.dist_coeff_array: SharedNDArray = dist_coeff_array

    @staticmethod
    def get_connected_devices_serial() -> List[str]:
        """Return a sorted list of serial numbers for connected RealSense devices.

        Only devices from the D400 series are returned.
        """
        serials: List[str] = []
        for d in rs.context().devices:
            if d.get_info(rs.camera_info.name).lower() != "platform camera":
                serial = d.get_info(rs.camera_info.serial_number)
                product_line = d.get_info(rs.camera_info.product_line)
                if product_line == "D400":
                    serials.append(serial)
        serials = sorted(serials)
        return serials

    # ========= context manager ===========
    def __enter__(self) -> "SingleRealsense":
        """Enter the runtime context and start recording."""
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the runtime context and stop recording."""
        self.stop()

    # ========= user API ===========
    def start(self, wait: bool = True, put_start_time: Optional[float] = None) -> None:
        """Start the recording process.

        Args:
            wait: If True, wait for the process to be ready.
            put_start_time: Optional start time for synchronization.
        """
        self.put_start_time = put_start_time
        super().start()
        if wait:
            self.start_wait()

    def stop(self, wait: bool = True) -> None:
        """Signal the process to stop recording."""
        self.stop_event.set()
        if wait:
            self.end_wait()

    def start_wait(self) -> None:
        """Wait until the process is ready."""
        self.ready_event.wait()

    def end_wait(self) -> None:
        """Block until the process terminates."""
        self.join()

    @property
    def is_ready(self) -> bool:
        """Return True if the process is ready."""
        return self.ready_event.is_set()

    def get(
        self, k: Optional[int] = None, out: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Retrieve data from the ring buffer.

        If k is None, retrieve the most recent entry; otherwise, retrieve the last k.
        """
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k, out=out)

    def get_all(self) -> Dict[str, Any]:
        """Return all data from the ring buffer."""
        return self.ring_buffer.get_all()

    def set_color_option(self, option: rs.option, value: float) -> None:
        """Set a color sensor option via a command."""
        self.command_queue.put(
            {
                "cmd": Command.SET_COLOR_OPTION.value,
                "option_enum": option.value,
                "option_value": value,
            }
        )

    def set_depth_option(self, option: rs.option, value: float) -> None:
        """Set a depth sensor option via a command."""
        self.command_queue.put(
            {
                "cmd": Command.SET_DEPTH_OPTION.value,
                "option_enum": option.value,
                "option_value": value,
            }
        )

    def set_exposure(
        self, exposure: Optional[float] = None, gain: Optional[float] = None
    ) -> None:
        """Set manual exposure and gain, or enable auto exposure if both are None."""
        if exposure is None and gain is None:
            self.set_color_option(rs.option.enable_auto_exposure, 1.0)
        else:
            self.set_color_option(rs.option.enable_auto_exposure, 0.0)
            if exposure is not None:
                self.set_color_option(rs.option.exposure, exposure)
            if gain is not None:
                self.set_color_option(rs.option.gain, gain)

    def set_depth_exposure(
        self, exposure: Optional[float] = None, gain: Optional[float] = None
    ) -> None:
        """Set manual depth exposure and gain, or enable auto exposure"""
        if exposure is None and gain is None:
            self.set_depth_option(rs.option.enable_auto_exposure, 1.0)
        else:
            self.set_depth_option(rs.option.enable_auto_exposure, 0.0)
            if exposure is not None:
                self.set_depth_option(rs.option.exposure, exposure)
            if gain is not None:
                self.set_depth_option(rs.option.gain, gain)

    def set_depth_preset(self, preset: str) -> None:
        """Set a predefined depth mode preset."""
        visual_preset: Dict[str, int] = {
            "Custom": 0,
            "Default": 1,
            "Hand": 2,
            "High Accuracy": 3,
            "High Density": 4,
        }
        self.set_depth_option(rs.option.visual_preset, visual_preset[preset])

    def set_white_balance(self, white_balance: Optional[float] = None) -> None:
        """Set white balance manually, or enable auto white balance if None."""
        if white_balance is None:
            self.set_color_option(rs.option.enable_auto_white_balance, 1.0)
        else:
            self.set_color_option(rs.option.enable_auto_white_balance, 0.0)
            self.set_color_option(rs.option.white_balance, white_balance)

    def get_intrinsics(self) -> np.ndarray:
        """Return a 3x3 camera intrinsic matrix from shared memory."""
        assert self.ready_event.is_set()
        fx, fy, ppx, ppy = self.intrinsics_array.get()[:4]
        mat = np.eye(3)
        mat[0, 0] = fx
        mat[1, 1] = fy
        mat[0, 2] = ppx
        mat[1, 2] = ppy
        return mat

    def get_extrinsics(self) -> np.ndarray:
        """Return a 4x4 camera extrinsic matrix from shared memory."""
        return self.extrinsics

    def get_dist_coeff(self) -> np.ndarray:
        """Return a 1D array of 5 distortion coefficients from shared memory."""
        assert self.ready_event.is_set()
        return np.array(self.dist_coeff_array.get()[:])

    def get_depth_scale(self) -> float:
        """Return the depth scale from shared memory."""
        assert self.ready_event.is_set()
        scale: float = float(self.intrinsics_array.get()[-1])
        return scale

    def start_recording(self, video_path: str, start_time: float = -1) -> None:
        """Send a command to start video recording.

        Args:
            video_path: The output video path.
            start_time: The recording start time.
        """
        assert self.enable_color
        path_len: int = len(video_path.encode("utf-8"))
        if path_len > self.MAX_PATH_LENGTH:
            raise RuntimeError("video_path too long.")
        self.command_queue.put(
            {
                "cmd": Command.START_RECORDING.value,
                "video_path": video_path,
                "recording_start_time": start_time,
            }
        )

    def stop_recording(self) -> None:
        """Send a command to stop video recording."""
        self.command_queue.put({"cmd": Command.STOP_RECORDING.value})

    def restart_put(self, start_time: float) -> None:
        """Restart ring buffer indexing with a new start time."""
        self.command_queue.put(
            {"cmd": Command.RESTART_PUT.value, "put_start_time": start_time}
        )

    def calibrate_extrinsics(
        self,
        visualize: bool = True,
        board_size: Tuple[int, int] = (6, 9),
        squareLength: float = 0.03,
        markerLength: float = 0.022,
    ) -> None:
        """Calibrate extrinsics using a ChArUco board.

        Displays calibration image if visualize is True.
        """
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        board = cv2.aruco.CharucoBoard(
            size=board_size,
            squareLength=squareLength,
            markerLength=markerLength,
            dictionary=dictionary,
        )
        while not self.ready_event.is_set():
            time.sleep(0.1)
        intrinsic_matrix: np.ndarray = self.get_intrinsics()
        dist_coef: np.ndarray = self.get_dist_coeff()

        out: Dict[str, Any] = self.ring_buffer.get()
        colors: np.ndarray = out["color"]
        calibration_img: np.ndarray = colors.copy()
        cv2.imshow(f"calibration_img_{self.serial_number}", calibration_img)
        cv2.waitKey(1)

        corners, ids, _ = cv2.aruco.detectMarkers(
            image=calibration_img, dictionary=dictionary, parameters=None
        )
        if len(corners) == 0:
            warnings.warn("No markers detected.", stacklevel=2)
            return

        retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            markerCorners=corners,
            markerIds=ids,
            image=calibration_img,
            board=board,
            cameraMatrix=intrinsic_matrix,
            distCoeffs=dist_coef,
        )
        if charuco_corners is None:
            warnings.warn("No ChArUco corners detected.", stacklevel=2)
            return

        print("Number of corners:", len(charuco_corners))

        if visualize:
            cv2.aruco.drawDetectedCornersCharuco(
                image=calibration_img,
                charucoCorners=charuco_corners,
                charucoIds=charuco_ids,
            )
            cv2.imshow(f"calibration_{self.serial_number}", calibration_img)
            cv2.waitKey(1)

        rvec: Optional[np.ndarray] = None
        tvec: Optional[np.ndarray] = None
        retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            charuco_corners, charuco_ids, board, intrinsic_matrix, dist_coef, rvec, tvec
        )
        if not retval or rvec is None or tvec is None:
            warnings.warn("Pose estimation failed.", stacklevel=2)
            return

        reprojected_points, _ = cv2.projectPoints(
            board.getChessboardCorners()[charuco_ids, :],
            rvec,
            tvec,
            intrinsic_matrix,
            dist_coef,
        )
        reprojected_points = reprojected_points.reshape(-1, 2)
        charuco_corners = charuco_corners.reshape(-1, 2)
        error: float = np.sqrt(
            np.sum((reprojected_points - charuco_corners) ** 2, axis=1)
        ).mean()

        print("Reprojection Error:", error)

        R_board2cam: np.ndarray = cv2.Rodrigues(rvec)[0]
        t_board2cam: np.ndarray = tvec[:, 0]
        print("R_board2cam:", R_board2cam)
        print("t_board2cam:", t_board2cam)

        tf: np.ndarray = np.eye(4, dtype=np.float64)
        tf[:3, :3] = R_board2cam
        tf[:3, 3] = t_board2cam

        tf_world2board: np.ndarray = np.eye(4, dtype=np.float64)
        tf_world2board[:3, :3] = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]])
        tf = tf @ tf_world2board

        np.save(os.path.join(self.extrinsics_dir, f"{self.serial_number}.npy"), tf)

    def _init_depth_process(self) -> None:
        """Initialize RealSense depth post-processing filters."""
        self.depth_to_disparity: rs.disparity_transform = rs.disparity_transform(True)
        self.disparity_to_depth: rs.disparity_transform = rs.disparity_transform(False)
        self.spatial: rs.filter = rs.spatial_filter()
        self.spatial.set_option(rs.option.filter_magnitude, 2)
        self.spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
        self.spatial.set_option(rs.option.filter_smooth_delta, 20)
        self.hole_filling: rs.filter = rs.hole_filling_filter()
        self.hole_filling.set_option(rs.option.holes_fill, 1)
        self.decimation: rs.filter = rs.decimation_filter()
        self.decimation.set_option(rs.option.filter_magnitude, 2)
        self.temporal: rs.filter = rs.temporal_filter()
        self.temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
        self.temporal.set_option(rs.option.filter_smooth_delta, 20)
        self.align: rs.align = rs.align(rs.stream.color)

    def _process_depth(self, depth_frame: rs.depth_frame) -> rs.depth_frame:
        """Process a depth frame using various filters."""
        filtered_depth: rs.depth_frame = self.depth_to_disparity.process(depth_frame)
        filtered_depth = self.spatial.process(filtered_depth)
        filtered_depth = self.temporal.process(filtered_depth)
        filtered_depth = self.disparity_to_depth.process(filtered_depth)
        return filtered_depth

    # ========= interval API ===========
    def run(self) -> None:
        """Main loop running in a separate process. Captures and processes frames,

        writes data to the ring buffer, and handles incoming commands.
        """
        threadpool_limits(1)
        w, h = self.resolution
        fps: int = self.capture_fps
        align = rs.align(rs.stream.color)

        rs_config: rs.config = rs.config()
        if self.enable_color:
            rs_config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
        if self.enable_depth:
            rs_config.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
            self._init_depth_process()
        if self.enable_infrared:
            rs_config.enable_stream(rs.stream.infrared, w, h, rs.format.y8, fps)

        pipeline: rs.pipeline = rs.pipeline()
        try:
            rs_config.enable_device(self.serial_number)
            pipeline_profile: rs.pipeline_profile = pipeline.start(rs_config)

            # Enable global time.
            color_sensor: rs.sensor = pipeline_profile.get_device().first_color_sensor()
            color_sensor.set_option(rs.option.global_time_enabled, 1)

            # Setup advanced mode if specified.
            if self.advanced_mode_config is not None:
                json_text: str = json.dumps(self.advanced_mode_config)
                device = pipeline_profile.get_device()
                advanced_mode: rs.rs400_advanced_mode = rs.rs400_advanced_mode(device)
                advanced_mode.load_json(json_text)

            # Intrinsics.
            color_stream: rs.stream_profile = pipeline_profile.get_stream(
                rs.stream.color
            )
            intr = color_stream.as_video_stream_profile().get_intrinsics()
            order = ["fx", "fy", "ppx", "ppy", "height", "width"]
            for i, name in enumerate(order):
                self.intrinsics_array.get()[i] = getattr(intr, name)

            if self.enable_depth:
                depth_sensor: rs.sensor = (
                    pipeline_profile.get_device().first_depth_sensor()
                )
                depth_scale: float = depth_sensor.get_depth_scale()
                self.intrinsics_array.get()[-1] = depth_scale

            self.dist_coeff_array.get()[:] = np.array(intr.coeffs)

            if self.verbose:
                print(f"[SingleRealsense {self.serial_number}] Main loop started.")

            put_idx = None
            put_start_time: float = (
                self.put_start_time if self.put_start_time is not None else time.time()
            )
            iter_idx: int = 0
            t_start: float = time.time()

            # Warm up the camera.
            for _ in range(30):
                pipeline.wait_for_frames()

            while not self.stop_event.is_set():
                wait_start_time: float = time.time()
                try:
                    frameset: rs.composite_frame = pipeline.wait_for_frames(
                        timeout_ms=int(1000 * (1 / fps)) + 10
                    )
                    if not hasattr(self, "last_frame"):
                        self.last_frame = frameset
                    self.last_frame = frameset
                except RuntimeError:
                    warnings.warn(
                        "RuntimeError in wait_for_frames, skipping frame.", stacklevel=2
                    )
                    frameset = self.last_frame
                receive_time: float = time.time()

                # Align frames to color.
                frameset = align.process(frameset)
                wait_time: float = time.time() - wait_start_time

                # Grab data.
                grab_start_time: float = time.time()
                data: Dict[str, Any] = {}
                data["camera_receive_timestamp"] = receive_time
                data["camera_capture_timestamp"] = frameset.get_timestamp() / 1000

                if self.enable_color:
                    color_frame: rs.video_frame = frameset.get_color_frame()
                    data["color"] = np.asarray(color_frame.get_data())
                    data["camera_capture_timestamp"] = (
                        color_frame.get_timestamp() / 1000
                    )
                    color_shape = data["color"].shape

                if self.enable_depth:
                    depth_frame: rs.depth_frame = frameset.get_depth_frame()
                    proc_depth_frame: rs.depth_frame = self._process_depth(depth_frame)
                    data["depth"] = np.asarray(proc_depth_frame.get_data())
                if self.is_ready:
                    data["intrinsics"] = self.get_intrinsics()
                else:
                    warnings.warn(
                        "Intrinsics not ready, using identity matrix temporarily.",
                        stacklevel=2,
                    )
                    data["intrinsics"] = np.eye(3)
                data["extrinsics"] = self.extrinsics

                if self.enable_infrared:
                    infrared_frame: rs.video_frame = frameset.get_infrared_frame()
                    data["infrared"] = np.asarray(infrared_frame.get_data())

                grab_time: float = time.time() - grab_start_time
                if self.verbose:
                    print(
                        f"[SingleRealsense {self.serial_number}] Grab data {grab_time}"
                    )

                # Apply transform.
                transform_start_time: float = time.time()
                put_data: Dict[str, Any] = data
                if self.transform is not None:
                    put_data = self.transform(dict(data))

                # Downsample to put_fps if required.
                if self.put_downsample:
                    local_idxs, global_idxs, put_idx = get_accumulate_timestamp_idxs(
                        timestamps=[receive_time],
                        start_time=put_start_time,
                        dt=1 / self.put_fps,
                        next_global_idx=put_idx,
                        allow_negative=True,
                    )
                    for step_idx in global_idxs:
                        put_data["step_idx"] = step_idx
                        put_data["timestamp"] = receive_time
                        if self.is_ready:
                            self.ring_buffer.put(put_data, wait=False)
                else:
                    step_idx = int((receive_time - put_start_time) * self.put_fps)
                    put_data["step_idx"] = step_idx
                    put_data["timestamp"] = receive_time
                    if self.is_ready:
                        self.ring_buffer.put(put_data, wait=False)

                transform_time: float = time.time() - transform_start_time
                if self.verbose:
                    print(
                        f"[SingleRealsense {self.serial_number}] Transform time "
                        f"{transform_time}"
                    )

                # Signal ready.
                if iter_idx == 0:
                    self.ready_event.set()

                # Record frame if video recorder is ready.
                rec_start_time: float = time.time()
                rec_data: Dict[str, Any] = data
                if self.recording_transform == self.transform:
                    rec_data = put_data
                elif self.recording_transform is not None:
                    rec_data = self.recording_transform(dict(data))

                if self.video_recorder.is_ready():
                    self.video_recorder.write_frame(
                        rec_data["color"], frame_time=receive_time
                    )
                rec_time: float = time.time() - rec_start_time
                if self.verbose:
                    print(
                        f"[SingleRealsense {self.serial_number}] Record time {rec_time}"
                    )

                # Execute commands.
                cmd_start: float = time.time()
                try:
                    commands: Dict[str, np.ndarray] = self.command_queue.get_all()
                    n_cmd: int = len(commands["cmd"])
                except Empty:
                    n_cmd = 0

                for i in range(n_cmd):
                    command: Dict[str, Any] = {}
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd: int = command["cmd"]
                    if cmd == Command.SET_COLOR_OPTION.value:
                        sensor = pipeline_profile.get_device().first_color_sensor()
                        option = rs.option(command["option_enum"])
                        val = float(command["option_value"])
                        sensor.set_option(option, val)
                    elif cmd == Command.SET_DEPTH_OPTION.value:
                        sensor = pipeline_profile.get_device().first_depth_sensor()
                        option = rs.option(command["option_enum"])
                        val = float(command["option_value"])
                        sensor.set_option(option, val)
                    elif cmd == Command.START_RECORDING.value:
                        video_path = str(command["video_path"])
                        start_time_val = float(command["recording_start_time"])
                        if start_time_val < 0:
                            start_time_val = None  # type: ignore
                        self.video_recorder.start(
                            video_path, color_shape, start_time=start_time_val
                        )
                    elif cmd == Command.STOP_RECORDING.value:
                        self.video_recorder.stop()
                        put_idx = None
                    elif cmd == Command.RESTART_PUT.value:
                        put_idx = None
                        put_start_time = float(command["put_start_time"])

                cmd_time: float = time.time() - cmd_start
                if self.verbose:
                    print(f"[SingleRealsense {self.serial_number}] Cmd time {cmd_time}")

                iter_idx += 1

                # Performance / debug.
                t_end: float = time.time()
                duration: float = t_end - t_start
                frequency: float = round(1 / duration, 1) if duration > 0 else 0.0
                t_start = t_end
                if frequency < fps // 2:
                    warnings.warn(
                        f"[{self.serial_number}] FPS {frequency} is smaller than {fps}",
                        stacklevel=2,
                    )
                    print("Debugging info:")
                    print("wait_time:", wait_time)
                    print("grab_time:", grab_time)
                    print("transform_time:", transform_time)
                    print("rec_time:", rec_time)
                    print("cmd_time:", cmd_time)
                if self.verbose:
                    print(f"[SingleRealsense {self.serial_number}] FPS {frequency}")
        finally:
            self.video_recorder.stop()
            rs_config.disable_all_streams()
            self.ready_event.set()

        if self.verbose:
            print(f"[SingleRealsense {self.serial_number}] Exiting worker process.")


def get_real_exporure_gain_white_balance() -> None:
    """Instantiate a SingleRealsense, set exposure and white balance."""
    series_number: List[str] = SingleRealsense.get_connected_devices_serial()
    with SharedMemoryManager() as shm_manager:
        with SingleRealsense(
            shm_manager=shm_manager,
            serial_number=series_number[2],
            enable_color=True,
            enable_depth=True,
            enable_infrared=False,
            put_fps=30,
            record_fps=30,
            verbose=True,
        ) as realsense:
            realsense.set_exposure(200, 64)
            realsense.set_white_balance(2800)
            for _ in range(30):
                realsense.get()
                time.sleep(0.1)
            color_frame: np.ndarray = realsense.get()["color"]
            cv2.imshow("color", color_frame)
            cv2.waitKey(0)


if __name__ == "__main__":
    get_real_exporure_gain_white_balance()
