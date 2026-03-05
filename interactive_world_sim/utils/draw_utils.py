import time
from typing import Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .pose_utils import PoseType, pose_convert


def save_video(
    video_path: str, imgs: np.ndarray, fps: int = 10, format: str = "bgr_uint8"
) -> None:
    """Save a video to a file.

    The format of the video is determined by the file extension.
    The format can be one of the following:
    - "bgr_uint8": BGR uint8 format.
    - "rgb_uint8": RGB uint8 format.
    - "bgr_float32": BGR float32 format.
    - "rgb_float32": RGB float32 format.
    Args:
        video_path (str): Path to save the video.
        imgs (np.ndarray): Images to save in shape (T, H, W, 3)
        fps (int): Frames per second.
        format (str): Format of the video.
    """
    assert imgs.ndim == 4 and imgs.shape[3] == 3
    H, W, _ = imgs.shape[1:]
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for img in imgs:
        if format == "rgb_uint8":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif format == "bgr_uint8":
            pass
        elif format == "bgr_float32":
            img = (img * 255).astype(np.uint8)
        elif format == "rgb_float32":
            img = (img * 255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            raise ValueError(f"Invalid format: {format}")
        writer.write(img)
    writer.release()


def concat_img_h(img_ls: list[np.ndarray]) -> np.ndarray:
    """Concatenate images horizontally.

    Args:
        img_ls (list[np.ndarray]): List of images to concatenate.

    Returns:
        np.ndarray: Concatenated image.
    """
    # Get the maximum height
    max_h = max(img.shape[0] for img in img_ls)

    # Resize images to the maximum height
    img_ls = [
        cv2.resize(img, (int(img.shape[1] * max_h / img.shape[0]), max_h))
        for img in img_ls
    ]

    # Concatenate images
    return cv2.hconcat(img_ls)


def concat_img_v(img_ls: list[np.ndarray]) -> np.ndarray:
    """Concatenate images vertically.

    Args:
        img_ls (list[np.ndarray]): List of images to concatenate.

    Returns:
        np.ndarray: Concatenated image.
    """
    # Get the maximum width
    max_w = max(img.shape[1] for img in img_ls)

    # Resize images to the maximum width
    img_ls = [
        cv2.resize(img, (max_w, int(img.shape[0] * max_w / img.shape[1])))
        for img in img_ls
    ]

    # Concatenate images
    return cv2.vconcat(img_ls)


def plot_2d_traj(
    img: np.ndarray,
    trajs: np.ndarray,
    radius: int = 3,
    total_len: Optional[int] = None,
    colors: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Plot 2D trajectories on the image as colored dots.

    This implementation minimizes Python loops by vectorizing operations.
    Args:
        img (np.ndarray): (H, W, 3) image in BGR format.
        trajs (np.ndarray): (n_trajs, n_steps, 2) trajectory array (col, row).
        radius (int): Radius of the plotted dot. If 0, points are single pixels.
        total_len (Optional[int]): Total length of the trajectory. If provided, the
            color of the trajectory is based on the relative position in the trajectory.
            If None, the total len = trajs.shape[1].
        colors (Optional[np.ndarray]): Custom colors for the trajectory points.

    Returns:
        np.ndarray: Modified image with trajectories drawn.
    """
    assert trajs.ndim == 3 and trajs.shape[2] == 2
    n_trajs, n_steps, _ = trajs.shape
    total_len = n_steps if total_len is None else total_len

    # Compute colormap for all points
    cmap = plt.get_cmap("plasma")
    if n_steps > 1:
        rel_positions = np.linspace(0, 1, total_len)[:n_steps]
    else:
        rel_positions = np.array([0.5])  # Single step, pick a middle color

    # Repeat relative positions for each trajectory
    if colors is None:
        rgba_colors = cmap(rel_positions)  # (N, 4)
        # Convert RGBA to BGR [0,255]
        bgr_colors = (rgba_colors[:, [2, 1, 0]] * 255).astype(np.uint8)  # (N, 3)
    else:
        bgr_colors = colors

    # For radius > 0, use cv2's circle function to draw filled circles
    for i in range(trajs.shape[0]):
        for t in range(trajs.shape[1]):
            x = int(trajs[i, t, 0])
            y = int(trajs[i, t, 1])
            color = tuple(bgr_colors[t].tolist())
            cv2.circle(img, (x, y), radius, color, thickness=-1)

    return img


def plot_bimanual_3d_traj(
    img: np.ndarray,
    cam_intrisics: np.ndarray,
    world_t_cam: np.ndarray,
    trajs: np.ndarray,
    pose_type: PoseType,
    radius: int = 3,
    total_len: Optional[int] = None,
) -> np.ndarray:
    """Plot 3D trajectories on the image as colored dots.

    Args:
        img (np.ndarray): (H, W, 3) image in BGR format.
        cam_intrisics (np.ndarray): Camera intrinsics (cx, cy, fx, fy).
        world_t_cam (np.ndarray): World to camera transformation matrix.
        trajs (np.ndarray): (n_trajs, n_steps, traj_dim) trajectory array.
        pose_type (PoseType): Pose type of the trajectories.
        radius (int): Radius of the plotted dot. If 0, points are single pixels.
        total_len (Optional[int]): Total length of the trajectory. If provided, the
            color of the trajectory is based on the relative position in the trajectory.
            If None, the total len = trajs.shape[1].

    Returns:
        np.ndarray: Modified image with trajectories drawn.
    """
    assert trajs.ndim == 3 and trajs.shape[2] == 20
    cx, cy, fx, fy = cam_intrisics
    for i in range(2):
        curr_trajs = trajs[:, :, i * 10 : (i + 1) * 10]
        n_trajs, n_steps, traj_dim = curr_trajs.shape
        curr_trajs = curr_trajs.reshape(-1, traj_dim)
        curr_trajs = pose_convert(
            curr_trajs, pose_type, PoseType.MAT
        )  # (n_trajs * n_steps, 4, 4)
        img_t_trajs = (
            np.linalg.inv(world_t_cam) @ curr_trajs
        )  # (n_trajs * n_steps, 4, 4)
        trajs_2d_x = cx + fx * img_t_trajs[:, 0, 3] / img_t_trajs[:, 2, 3]
        trajs_2d_y = cy + fy * img_t_trajs[:, 1, 3] / img_t_trajs[:, 2, 3]
        trajs_2d = np.stack([trajs_2d_x, trajs_2d_y], axis=-1)  # (n_trajs * n_steps, 2)
        trajs_2d = trajs_2d.reshape(n_trajs, n_steps, 2)  # (n_trajs, n_steps, 2)
        img = plot_2d_traj(img.copy(), trajs_2d, radius, total_len)
    return img


def plot_single_3d_pos_traj(
    img: np.ndarray,
    cam_intrisics: np.ndarray,
    world_t_cam: np.ndarray,
    trajs: np.ndarray,
    radius: int = 3,
    total_len: Optional[int] = None,
    colors: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Plot 3D trajectories on the image as colored dots.

    Args:
        img (np.ndarray): (H, W, 3) image in BGR format.
        cam_intrisics (np.ndarray): Camera intrinsics (cx, cy, fx, fy).
        world_t_cam (np.ndarray): World to camera transformation matrix.
        trajs (np.ndarray): (n_trajs, n_steps, traj_dim) trajectory array.
        pose_type (PoseType): Pose type of the trajectories.
        radius (int): Radius of the plotted dot. If 0, points are single pixels.
        total_len (Optional[int]): Total length of the trajectory. If provided, the
            color of the trajectory is based on the relative position in the trajectory.
            If None, the total len = trajs.shape[1].
        colors (Optional[np.ndarray]): (n_steps, 3) colors for the trajectory points.

    Returns:
        np.ndarray: Modified image with trajectories drawn.
    """
    assert trajs.ndim == 3
    cx, cy, fx, fy = cam_intrisics
    n_trajs, n_steps, traj_dim = trajs.shape
    trajs = trajs.reshape(-1, traj_dim)
    trajs_mat = np.tile(np.eye(4)[None], (n_trajs * n_steps, 1, 1))
    trajs_mat[:, :3, 3] = trajs[:, :3]
    img_t_trajs = np.linalg.inv(world_t_cam) @ trajs_mat  # (n_trajs * n_steps, 4, 4)
    trajs_2d_x = cx + fx * img_t_trajs[:, 0, 3] / img_t_trajs[:, 2, 3]
    trajs_2d_y = cy + fy * img_t_trajs[:, 1, 3] / img_t_trajs[:, 2, 3]
    trajs_2d = np.stack([trajs_2d_x, trajs_2d_y], axis=-1)  # (n_trajs * n_steps, 2)
    trajs_2d = trajs_2d.reshape(n_trajs, n_steps, 2)  # (n_trajs, n_steps, 2)
    return plot_2d_traj(img, trajs_2d, radius, total_len, colors=colors)


def draw_text(
    img: np.ndarray,
    text: str,
    uv_top_left: tuple[float, float],
    color: tuple[int, int, int] = (255, 255, 255),
    fontScale: float = 0.5,
    thickness: int = 1,
    fontFace: int = cv2.FONT_HERSHEY_SIMPLEX,
    outline_color: tuple[int, int, int] = (0, 0, 0),
    line_spacing: float = 1.5,
) -> np.ndarray:
    """Draws multiline with an outline."""
    assert isinstance(text, str)

    uv_top_left_np: np.ndarray = np.array(uv_top_left, dtype=float)
    assert uv_top_left_np.shape == (2,)

    for line in text.splitlines():
        (w, h), _ = cv2.getTextSize(
            text=line,
            fontFace=fontFace,
            fontScale=fontScale,
            thickness=thickness,
        )
        uv_bottom_left_i = uv_top_left_np + np.array([0, h])
        org = tuple(uv_bottom_left_i.astype(int))

        if outline_color is not None:
            cv2.putText(
                img,
                text=line,
                org=org,
                fontFace=fontFace,
                fontScale=fontScale,
                color=outline_color,
                thickness=thickness * 3,
                lineType=cv2.LINE_AA,
            )
        cv2.putText(
            img,
            text=line,
            org=org,
            fontFace=fontFace,
            fontScale=fontScale,
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

        uv_top_left_np += [0, h * line_spacing]
    return img


def draw_dual_stick_axes(
    l_xy: Tuple[float, float],
    r_xy: Tuple[float, float],
    radius_px: int = 100,
    margin_px: int = 20,
    bg_colour: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Draw two joystick-axis visualisers side-by-side

    Parameters
    ----------
    l_xy, r_xy
        Current (x, y) axis readings in the usual `[-1, 1]` range.
        Positive *x* = stick right, positive *y* = stick down  ⇢  eye-congruent.
    radius_px
        Radius, in pixels, of the drawn circle.
    margin_px
        Whitespace around and between the circles.
    bg_colour
        Background colour (BGR).

    Returns
    -------
    canvas : np.ndarray
        BGR image ready for `cv2.imshow(...)`.
    """
    # Canvas big enough for two circles and margins
    h = 2 * radius_px + 2 * margin_px
    w = 4 * radius_px + 3 * margin_px
    canvas = np.full((h, w, 3), bg_colour, dtype=np.uint8)

    # Helpers --------------------------------------------------------------- #
    def _draw_one(center: np.ndarray, xy_now: np.ndarray) -> None:
        cx, cy = center
        # Outer boundary
        cv2.circle(canvas, center, radius_px, (0, 0, 0), 2, lineType=cv2.LINE_AA)
        # Cross-hairs
        cv2.line(
            canvas,
            (cx - radius_px, cy),
            (cx + radius_px, cy),
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.line(
            canvas,
            (cx, cy - radius_px),
            (cx, cy + radius_px),
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

        # Convert axis reading → pixel
        def to_px(x: float, y: float) -> Tuple[int, int]:
            px = int(round(cx + x * radius_px))
            py = int(round(cy - y * radius_px))  # y  already points down for     y>0
            return px, py

        # Arrow for current reading
        px_now = to_px(*xy_now)
        cv2.arrowedLine(
            canvas,
            center,
            px_now,
            (0, 0, 255),
            2,
            tipLength=0.12,
            line_type=cv2.LINE_AA,
        )
        # Emphasise the latest dot
        cv2.circle(canvas, px_now, 6, (0, 0, 255), -1, cv2.LINE_AA)

    # Left stick
    left_ctr = (margin_px + radius_px, margin_px + radius_px)
    _draw_one(left_ctr, l_xy)

    # Right stick
    right_ctr = (margin_px * 2 + radius_px * 3, margin_px + radius_px)
    _draw_one(right_ctr, r_xy)

    return canvas


def test_plot_2d_traj() -> None:
    # Create a random image
    img = (np.ones((512, 512, 3)) * 255).astype(np.uint8)
    # trajs = np.array([[[10, 10], [20, 20], [30, 30]], [[50, 50], [60, 60], [70, 70]]])
    trajs = np.random.randint(0, 512, (1000, 20, 2))
    start_time = time.time()
    img = plot_2d_traj(img, trajs)
    print(f"Time taken: {time.time() - start_time:.3f}s")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()


if __name__ == "__main__":
    test_plot_2d_traj()
