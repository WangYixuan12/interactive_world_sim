from typing import Optional, Tuple

import cv2
import numpy as np

from interactive_world_sim.utils.timestamp_accumulator import (
    get_accumulate_timestamp_idxs,
)


class VideoRecorder:
    """Replaces the use of PyAV with OpenCV's cv2.VideoWriter.

    This class handles video recording using OpenCV. It supports basic
    operations such as starting, writing frames, and stopping recording.
    """

    def __init__(self, fps: float, codec: str, input_pix_fmt: str) -> None:
        """Initialize the VideoRecorder.

        :param fps: Frames per second for the output video.
        :param codec: Four-character code for the codec (e.g., 'mp4v', 'XVID', 'H264').
        :param input_pix_fmt: 'rgb24' or 'bgr24'. Used to handle color conversion.
        :param kwargs: Additional keyword arguments.
        """
        self.fps = fps
        self.codec = codec
        self.input_pix_fmt = input_pix_fmt
        # Runtime state:
        self._reset_state()

    def _reset_state(self) -> None:
        """Reset runtime internal state."""
        self.video_writer = None
        self.shape: Optional[Tuple[int, int, int]] = None
        self.dtype: Optional[np.dtype] = None
        self.start_time: Optional[float] = None
        self.next_global_idx = 0

    def __del__(self) -> None:
        self.stop()

    def is_ready(self) -> bool:
        """Return True if recording has been started, else False."""
        return self.video_writer is not None

    def start(
        self,
        file_path: str,
        shape: Tuple[int, int, int],
        start_time: Optional[float] = None,
    ) -> None:
        """Initialize the video writer for the given output file.

        :param file_path: Output file path.
        :param start_time: Start time for timestamp accumulation, if any.
        """
        if self.is_ready():
            # If still recording, stop first.
            self.stop()

        # OpenCV requires a known frame size which we set upon receiving the first frame
        # Meanwhile, store file_path, fourcc, and fps for later.
        self.output_file_path = file_path
        self.fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self.start_time = start_time
        self._init_writer(shape)

    def write_frame(self, img: np.ndarray, frame_time: Optional[float] = None) -> None:
        """Write a single frame, potentially repeating it based on timestamp.

        :param img: Frame image as a NumPy array.
        :param frame_time: The frame's timestamp.
        :raises RuntimeError: If start() has not been called.
        """
        if not self.is_ready():
            # Open the writer if first frame, otherwise raise error.
            if self.shape is None:
                self._init_writer(img.shape)
            else:
                raise RuntimeError("Must run start() before writing!")

        n_repeats = 1
        if self.start_time is not None:
            local_idxs, _, self.next_global_idx = get_accumulate_timestamp_idxs(
                timestamps=[frame_time] if frame_time is not None else [0.0],
                start_time=self.start_time,
                dt=1 / self.fps,
                next_global_idx=self.next_global_idx,
            )
            # Repeat the frame as many times as required.
            n_repeats = len(local_idxs)

        if self.shape is None:
            self._init_writer(img.shape)

        # Validate shape and dtype.
        if img.shape != self.shape:
            raise ValueError(
                f"Frame shape {img.shape} does not match initial shape "
                f"{self.shape}."
            )
        if img.dtype != self.dtype:
            raise ValueError(
                f"Frame dtype {img.dtype} does not match initial dtype "
                f"{self.dtype}."
            )

        # Convert color space if necessary.
        if self.input_pix_fmt == "rgb24":
            frame_to_write = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            # Assume BGR for 'bgr24'.
            frame_to_write = img

        for _ in range(n_repeats):
            assert self.video_writer is not None
            self.video_writer.write(frame_to_write)

    def stop(self) -> None:
        """Stop recording and release the video writer."""
        if self.is_ready() and self.video_writer is not None:
            self.video_writer.release()
        self._reset_state()

    def _init_writer(self, shape: Tuple[int, int, int]) -> None:
        """Initialize the video writer using the frame shape.

        :param shape: Tuple representing (height, width, channels).
        """
        h, w, _ = shape
        self.shape = shape
        self.dtype = np.dtype(np.uint8)  # Typically, frames are uint8 in OpenCV.
        self.video_writer = cv2.VideoWriter(
            self.output_file_path, self.fourcc, self.fps, (w, h)
        )
