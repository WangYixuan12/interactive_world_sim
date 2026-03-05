import multiprocessing as mp
import time
from typing import Any, Dict, Optional

import cv2
import numpy as np
from threadpoolctl import threadpool_limits
from yixuan_utilities.draw_utils import center_crop, resize_to_height

from interactive_world_sim.real_world.multi_realsense import MultiRealsense
from interactive_world_sim.utils.shared_ndarray import SharedNDArray


class MultiCameraVisualizer(mp.Process):
    """Visualizes multiple camera streams in a grid layout."""

    def __init__(
        self,
        realsense: MultiRealsense,
        row: int,
        col: int,
        window_name: str = "Multi Cam Vis",
        vis_fps: int = 60,
        fill_value: int = 0,
        rgb_to_bgr: bool = True,
    ) -> None:
        super().__init__()
        self.row: int = row
        self.col: int = col
        self.window_name: str = window_name
        self.vis_fps: int = vis_fps
        self.fill_value: int = fill_value
        self.rgb_to_bgr: bool = rgb_to_bgr
        self.realsense: MultiRealsense = realsense
        # Shared variable for signaling stop across processes.
        self.stop_event = mp.Event()

        # --- shared-mask bookkeeping ---
        w, h = next(iter(self.realsense.cameras.values())).resolution
        self.mask = SharedNDArray.create_from_shape(
            mem_mgr=realsense.shm_manager,
            shape=(h, w),
            dtype=np.uint8,
        )
        self.mask.get()[:] = np.zeros(self.mask.shape, dtype=np.uint8)

        self.goal_img = SharedNDArray.create_from_shape(
            mem_mgr=realsense.shm_manager,
            shape=(h, w, 3),
            dtype=np.uint8,
        )
        self.goal_img.get()[:] = np.zeros(self.goal_img.shape, dtype=np.uint8)

        # Add subgoal image shared buffer
        self.subgoal_img = SharedNDArray.create_from_shape(
            mem_mgr=realsense.shm_manager,
            shape=(128, 128, 3),
            dtype=np.uint8,
        )
        self.subgoal_img.get()[:] = np.zeros(self.subgoal_img.shape, dtype=np.uint8)

        self.recording = SharedNDArray.create_from_shape(
            mem_mgr=realsense.shm_manager,
            shape=(1,),
            dtype=bool,
        )
        self.recording.get()[:] = False
        MAX_PATH_LENGTH: int = 4096
        self.recording_path = SharedNDArray.create_from_shape(
            mem_mgr=realsense.shm_manager,
            shape=(1,),
            dtype=np.dtype(f"<U{MAX_PATH_LENGTH}"),
        )
        self.recording_path.get()[:] = np.array(["a" * MAX_PATH_LENGTH])

    def __del__(self) -> None:
        self.stop()

    def start(self, wait: bool = False) -> None:
        """Start the visualizer process."""
        super().start()

    def stop(self, wait: bool = False) -> None:
        """Stop the visualizer process."""
        self.stop_event.set()
        if wait:
            self.stop_wait()

    def start_wait(self) -> None:
        """Start the visualizer process and wait for it to finish."""
        # You may add additional start-wait logic if needed.

    def stop_wait(self) -> None:
        """Stop the visualizer process and wait for it to finish."""
        self.join()

    def run(self) -> None:
        """Main loop for visualizing camera streams."""
        # Limit OpenCV threads and thread pools.
        cv2.setNumThreads(1)
        threadpool_limits(1)
        # Use a slice to reverse the channel order if needed.
        channel_slice: slice = slice(None)
        if self.rgb_to_bgr:
            channel_slice = slice(None, None, -1)

        vis_data: Dict[Any, Any] = {}
        vis_img: Optional[np.ndarray] = None

        try:
            while not self.stop_event.is_set():
                vis_data = self.realsense.get(out=vis_data)

                # Stack all color frames from each camera. The comprehension assumes
                # that each entry in vis_data is a dictionary containing a "color" key.
                color = np.stack(
                    [value["color"][..., ::-1] for value in vis_data.values()]
                )
                N, H, W, C = color.shape
                assert C == 3
                vis_imgs = []

                # Arrange each color image into a grid based on row and column.
                for r in range(self.row):
                    for c in range(self.col):
                        idx: int = c + r * self.col
                        if idx < N:
                            vis_imgs.append(color[idx, :, :, channel_slice])

                # overlay the mask on the last column of the grid.
                curr_mask = self.mask.get()[:]
                if curr_mask.max() > 0:
                    mask_vis = np.tile(curr_mask[..., None], (1, 1, 3)) * 255
                    idx = self.col - 1
                    color_i = color[idx, :, :, channel_slice]
                    color_i = center_crop(color_i, self.mask.shape[:2])
                    color_i = cv2.resize(
                        color_i, (self.mask.shape[1], self.mask.shape[0])
                    )
                    vis_imgs.append((color_i * 0.5 + mask_vis * 0.5).astype(np.uint8))

                # show subgoal image (to the left of mask)
                subgoal_img = self.subgoal_img.get()[:]
                # Resize subgoal to match height if needed
                if subgoal_img.max() > 0:
                    if subgoal_img.shape[0] != H:
                        subgoal_img = resize_to_height(subgoal_img, H)
                    vis_imgs.append(subgoal_img)

                # show goal image
                goal_img = self.goal_img.get()[:]
                if goal_img.max() > 0:
                    if goal_img.shape[0] != H:
                        goal_img = resize_to_height(goal_img, H)
                    goal_img = (goal_img * 0.5 + vis_imgs[-1] * 0.5).astype(np.uint8)
                    vis_imgs.append(goal_img)

                vis_img = np.concatenate(vis_imgs, axis=1)
                cv2.imshow(self.window_name, vis_img)
                cv2.pollKey()  # Polls for GUI events.

                if self.recording.get()[0]:
                    if not hasattr(self, "video_writer"):
                        vid_path = self.recording_path.get()[0]
                        self.video_writer = cv2.VideoWriter(
                            vid_path,
                            fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
                            fps=self.vis_fps,
                            frameSize=(vis_img.shape[1], vis_img.shape[0]),
                        )
                    self.video_writer.write(vis_img.copy())
                else:
                    if hasattr(self, "video_writer"):
                        self.video_writer.release()
                        del self.video_writer

                time.sleep(1 / self.vis_fps)
        except KeyboardInterrupt:
            if hasattr(self, "video_writer"):
                self.video_writer.release()
                del self.video_writer
        finally:
            if hasattr(self, "video_writer"):
                self.video_writer.release()
                del self.video_writer

    def set_mask(self, mask: np.ndarray) -> None:
        """Copy *mask* into the shared buffer so the child sees the change."""
        self.mask.get()[:] = mask[:]

    def set_goal_img(self, goal_img: np.ndarray) -> None:
        """Copy *goal_img* into the shared buffer so the child sees the change."""
        self.goal_img.get()[:] = goal_img[:]

    def set_subgoal_img(self, subgoal_img: np.ndarray) -> None:
        """Copy *subgoal_img* into the shared buffer so the child sees the change."""
        self.subgoal_img.get()[:] = subgoal_img[:]

    def start_recording(self, path: np.ndarray) -> None:
        """Set the recording flag to True."""
        self.recording.get()[:] = True
        self.recording_path.get()[:] = path[:]

    def stop_recording(self) -> None:
        """Set the recording flag to False."""
        self.recording.get()[:] = False
