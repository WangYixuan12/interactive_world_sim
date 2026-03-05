from typing import Optional

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from pytorch_lightning.loggers import WandbLogger
from torchmetrics.functional import (
    mean_squared_error,
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
    universal_image_quality_index,
)
from tqdm import tqdm

from interactive_world_sim.algorithms.common.metrics import (
    FrechetInceptionDistance,
    FrechetVideoDistance,
    LearnedPerceptualImagePatchSimilarity,
)

plt.set_loglevel("warning")


# FIXME: clean up & check this util
def log_video(
    observation_hat: torch.Tensor,
    observation_gt: Optional[torch.Tensor] = None,
    goal: Optional[torch.Tensor] = None,
    step: Optional[int] = None,
    namespace: str = "train",
    prefix: str = "video",
    context_frames: int = 0,
    color: tuple[int, int, int] = (255, 0, 0),
    logger: Optional[WandbLogger] = None,
    postfix: Optional[list[str]] = None,
    captions: Optional[str] = None,
    indent: int = 0,
) -> None:
    """take in video tensors in range [-1, 1] and log into wandb

    :param observation_hat: predicted observation tensor of shape (frame, batch,
        channel, height, width)
    :param observation_gt: ground-truth observation tensor of shape (frame, batch,
        channel, height, width)
    :param goal: goal tensor of shape (frame, batch, channel, height, width)
    :param step: an int indicating the step number
    :param namespace: a string specify a name space this video logging falls under,
        e.g. train, val
    :param prefix: a string specify a prefix for the video name
    :param context_frames: an int indicating how many frames in observation_hat are
        ground truth given as context
    :param color: a tuple of 3 numbers specifying the color of the border for ground
        truth frames
    :param logger: optional logger to use. use global wandb if not specified
    """
    assert (
        observation_hat.ndim == 5
    ), "observation_hat must have shape (frame, batch, channel, height, width)"
    if postfix is None:
        postfix = []
    if not logger:
        logger = wandb
    if observation_gt is None:
        observation_gt = torch.zeros_like(observation_hat)
    # Add red border of 1 pixel width to the context frames
    for i, c in enumerate(color):
        c_normalized = float(c) / 255.0
        observation_hat[:context_frames, :, i, [0, -1], :] = c_normalized
        observation_hat[:context_frames, :, i, :, [0, -1]] = c_normalized
        observation_gt[:, :, i, [0, -1], :] = c_normalized
        observation_gt[:, :, i, :, [0, -1]] = c_normalized
    if goal is not None:
        for i, c in enumerate(color):
            c_normalized = float(c) / 255.0
            goal[:, :, i, [0, -1], :] = c_normalized
            goal[:, :, i, :, [0, -1]] = c_normalized
        video = (
            torch.cat([observation_hat, observation_gt, goal], -1)
            .detach()
            .cpu()
            .numpy()
        )
    else:
        video = torch.cat([observation_hat, observation_gt], -1).detach().cpu().numpy()
    video = np.transpose(
        np.clip(video, a_min=0.0, a_max=1.0) * 255, (1, 0, 2, 3, 4)
    ).astype(np.uint8)
    # video[..., 1:] = video[..., :1]  # remove framestack, only visualize current frame
    n_samples = len(video)
    # use wandb directly here since pytorch lightning doesn't support logging videos yet
    for i in range(n_samples):
        name = f"{namespace}/{prefix}_{i + indent}" + (
            f"_{postfix[i]}" if i < len(postfix) else ""
        )
        logger.log(
            {
                name: wandb.Video(video[i], fps=1, caption=captions, format="mp4"),
                "trainer/global_step": step,
            }
        )


def get_validation_metrics_for_videos(
    observation_hat: torch.Tensor,
    observation_gt: torch.Tensor,
    lpips_model: Optional[LearnedPerceptualImagePatchSimilarity] = None,
    fid_model: Optional[FrechetInceptionDistance] = None,
    fvd_model: Optional[FrechetVideoDistance] = None,
) -> dict:
    """Get validation metrics for video prediction

    :param observation_hat: predicted observation tensor of shape (frame, batch,
        channel, height, width)
    :param observation_gt: ground-truth observation tensor of shape (frame, batch,
        channel, height, width)
    :param lpips_model: a LearnedPerceptualImagePatchSimilarity object from
        algorithm.common.metrics
    :param fid_model: a FrechetInceptionDistance object from algorithm.common.metrics
    :param fvd_model: a FrechetVideoDistance object from algorithm.common.metrics
    :return: a tuple of metrics
    """
    frame, batch, channel, height, width = observation_hat.shape
    output_dict = {}
    observation_gt = observation_gt.type_as(
        observation_hat
    )  # some metrics don't fully support fp16

    if frame < 9:
        fvd_model = None  # FVD requires at least 9 frames

    if fvd_model is not None:
        output_dict["fvd"] = fvd_model.compute(
            torch.clamp(observation_hat, -1.0, 1.0),
            torch.clamp(observation_gt, -1.0, 1.0),
        )

    # reshape to (frame * batch, channel, height, width) for image losses
    observation_hat = observation_hat.reshape(-1, channel, height, width)
    observation_gt = observation_gt.reshape(-1, channel, height, width)

    output_dict["mse"] = mean_squared_error(observation_hat, observation_gt)
    output_dict["psnr"] = peak_signal_noise_ratio(
        observation_hat, observation_gt, data_range=2.0
    )
    output_dict["ssim"] = structural_similarity_index_measure(
        observation_hat, observation_gt, data_range=2.0
    )
    output_dict["uiqi"] = universal_image_quality_index(observation_hat, observation_gt)
    # operations for LPIPS and FID
    observation_hat = torch.clamp(observation_hat, -1.0, 1.0)
    observation_gt = torch.clamp(observation_gt, -1.0, 1.0)

    if lpips_model is not None:
        lpips_model.update(observation_hat, observation_gt)
        lpips = lpips_model.compute().item()
        # Reset the states of non-functional metrics
        output_dict["lpips"] = lpips
        lpips_model.reset()

    if fid_model is not None:
        observation_hat_uint8 = ((observation_hat + 1.0) / 2 * 255).type(torch.uint8)
        observation_gt_uint8 = ((observation_gt + 1.0) / 2 * 255).type(torch.uint8)
        fid_model.update(observation_gt_uint8, real=True)
        fid_model.update(observation_hat_uint8, real=False)
        fid = fid_model.compute()
        output_dict["fid"] = fid
        # Reset the states of non-functional metrics
        fid_model.reset()

    return output_dict


def is_grid_env(env_id: str) -> bool:
    return "maze2d" in env_id or "diagonal2d" in env_id


def get_maze_grid(env_id: str) -> list:
    # import gym
    # maze_string = gym.make(env_id).str_maze_spec
    if "large" in env_id:
        maze_string = "############\\#OOOO#OOOOO#\\#O##O#O#O#O#\\#OOOOOO#OOO#\\#O####O###O#\\#OO#O#OOOOO#\\##O#O#O#O###\\#OO#OOO#OGO#\\############"  # noqa
    if "medium" in env_id:
        maze_string = "########\\#OO##OO#\\#OO#OOO#\\##OOO###\\#OO#OOO#\\#O#OO#O#\\#OOO#OG#\\########"  # noqa
    if "umaze" in env_id:
        maze_string = "#####\\#GOO#\\###O#\\#OOO#\\#####"
    lines = maze_string.split("\\")
    grid = [line[1:-1] for line in lines]
    return grid[1:-1]


def get_random_start_goal(
    env_id: str, batch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    maze_grid = get_maze_grid(env_id)
    s2i = {"O": 0, "#": 1, "G": 2}
    maze_grid = [[s2i[s] for s in r] for r in maze_grid]
    maze_grid = np.array(maze_grid)
    x, y = np.nonzero(maze_grid == 0)
    indices = np.random.randint(len(x), size=batch_size)
    start = np.stack([x[indices], y[indices]], -1) + 1
    x, y = np.nonzero(maze_grid == 2)
    goal = np.concatenate([x, y], -1)
    goal = np.tile(goal[None, :], (batch_size, 1)) + 1
    return start, goal


def plot_maze_layout(ax: plt.Axes, maze_grid: np.ndarray) -> None:
    ax.clear()

    if maze_grid is not None:
        for i, row in enumerate(maze_grid):
            for j, cell in enumerate(row):
                if cell == "#":
                    square = plt.Rectangle(
                        (i + 0.5, j + 0.5), 1, 1, edgecolor="black", facecolor="black"
                    )
                    ax.add_patch(square)

    ax.set_aspect("equal")
    ax.grid(True, color="white", linewidth=4)
    ax.set_axisbelow(True)
    ax.spines["top"].set_linewidth(4)
    ax.spines["right"].set_linewidth(4)
    ax.spines["bottom"].set_linewidth(4)
    ax.spines["left"].set_linewidth(4)
    ax.set_facecolor("lightgray")
    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False,
    )
    ax.set_xticks(np.arange(0.5, len(maze_grid) + 0.5))
    ax.set_yticks(np.arange(0.5, len(maze_grid[0]) + 0.5))
    ax.set_xlim(0.5, len(maze_grid) + 0.5)
    ax.set_ylim(0.5, len(maze_grid[0]) + 0.5)
    ax.grid(True, color="white", which="minor", linewidth=4)


def plot_start_goal(ax: plt.Axes, start_goal: tuple[np.ndarray, np.ndarray]) -> None:
    def draw_star(
        center: tuple[float, float],
        radius: float,
        num_points: int = 5,
        color: str = "black",
    ) -> None:
        angles = np.linspace(0.0, 2 * np.pi, num_points, endpoint=False) + 5 * np.pi / (
            2 * num_points
        )
        inner_radius = radius / 2.0

        points = []
        for angle in angles:
            points.extend(
                [
                    center[0] + radius * np.cos(angle),
                    center[1] + radius * np.sin(angle),
                    center[0] + inner_radius * np.cos(angle + np.pi / num_points),
                    center[1] + inner_radius * np.sin(angle + np.pi / num_points),
                ]
            )

        star = plt.Polygon(np.array(points).reshape(-1, 2), color=color)
        ax.add_patch(star)

    start_x, start_y = start_goal[0]
    start_outer_circle = plt.Circle(
        (start_x, start_y), 0.16, facecolor="white", edgecolor="black"
    )
    ax.add_patch(start_outer_circle)
    start_inner_circle = plt.Circle((start_x, start_y), 0.08, color="black")
    ax.add_patch(start_inner_circle)

    goal_x, goal_y = start_goal[1]
    goal_outer_circle = plt.Circle(
        (goal_x, goal_y), 0.16, facecolor="white", edgecolor="black"
    )
    ax.add_patch(goal_outer_circle)
    draw_star((goal_x, goal_y), radius=0.08)


def make_trajectory_images(
    env_id: str,
    trajectory: np.ndarray,
    batch_size: int,
    start: np.ndarray,
    goal: np.ndarray,
    plot_end_points: bool = True,
) -> list[np.ndarray]:
    images = []
    for batch_idx in range(batch_size):
        fig, ax = plt.subplots()
        if is_grid_env(env_id):
            maze_grid = get_maze_grid(env_id)
        else:
            maze_grid = None
        plot_maze_layout(ax, maze_grid)
        (
            ax.scatter(
                trajectory[:, batch_idx, 0],
                trajectory[:, batch_idx, 1],
                c=np.arange(len(trajectory)),
                cmap="Reds",
            ),
        )
        if plot_end_points:
            start_goal = (start[batch_idx], goal[batch_idx])
            plot_start_goal(ax, start_goal)
        # plt.title(f"sample_{batch_idx}")
        fig.tight_layout()
        fig.canvas.draw()
        img_shape = fig.canvas.get_width_height()[::-1] + (4,)
        img = (
            np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            .copy()
            .reshape(img_shape)
        )
        images.append(img)

        plt.close()
    return images


def make_convergence_animation(
    env_id: str,
    plan_history: list,
    trajectory: np.ndarray,
    start: np.ndarray,
    goal: np.ndarray,
    open_loop_horizon: int,
    namespace: str,
    interval: int = 100,
    plot_end_points: bool = True,
    batch_idx: int = 0,
) -> str:
    # - plan_history:
    #   contains for each time step all the MPC predicted plans for each pyramid noise
    #   level. Structured as a list of length (episode_len // open_loop_horizon), where
    #   each element corresponds to a control_time_step and stores a list of length
    #   pyramid_height, where each element is a plan at a different pyramid noise level
    #   and stored as a tensor of shape (episode_len // open_loop_horizon -
    #   control_time_step, batch_size, x_stacked_shape)

    # select index and prune history
    start, goal = start[batch_idx], goal[batch_idx]
    trajectory = trajectory[:, batch_idx]
    plan_history = [[pm[:, batch_idx] for pm in pt] for pt in plan_history]
    trajectory, plan_history = prune_history(
        plan_history, trajectory, goal, open_loop_horizon
    )

    # animate the convergence of the first plan
    fig, ax = plt.subplots()
    if "large" in env_id:
        fig.set_size_inches(3.5, 5)
    else:
        fig.set_size_inches(3, 3)
    ax.set_axis_off()
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)

    if is_grid_env(env_id):
        maze_grid = get_maze_grid(env_id)
    else:
        maze_grid = None

    def update(frame: int) -> None:
        plot_maze_layout(ax, maze_grid)

        plan_history_m = plan_history[0][frame]
        plan_history_m = plan_history_m.numpy()
        ax.scatter(
            plan_history_m[:, 0],
            plan_history_m[:, 1],
            c=np.arange(len(plan_history_m))[::-1],
            cmap="Reds",
        )

        if plot_end_points:
            plot_start_goal(ax, (start, goal))

    frames = tqdm(range(len(plan_history[0])), desc="Making convergence animation")
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=interval)
    prefix = wandb.run.id if wandb.run is not None else env_id
    filename = f"/tmp/{prefix}_{namespace}_convergence.mp4"
    ani.save(filename, writer="ffmpeg", fps=24)
    return filename


def prune_history(
    plan_history: list,
    trajectory: np.ndarray,
    goal: np.ndarray,
    open_loop_horizon: int,
) -> tuple[np.ndarray, list]:
    dist = np.linalg.norm(
        trajectory[:, :2] - np.array(goal)[None],
        axis=-1,
    )
    reached = dist < 0.2
    if reached.any():
        cap_idx = np.argmax(reached)
        trajectory = trajectory[: cap_idx + open_loop_horizon + 1]
        plan_history = plan_history[: cap_idx // open_loop_horizon + 2]

    pruned_plan_history: list = []
    for plans in plan_history:
        pruned_plan_history.append([])
        for m in range(len(plans)):
            plan = plans[m]
            pruned_plan_history[-1].append(plan)
        plan = pruned_plan_history[-1][-1]
        dist = np.linalg.norm(plan.numpy()[:, :2] - np.array(goal)[None], axis=-1)
        reached = dist < 0.2
        if reached.any():
            cap_idx = np.argmax(reached) + 1
            pruned_plan_history[-1] = [p[:cap_idx] for p in pruned_plan_history[-1]]
    return trajectory, pruned_plan_history


def make_mpc_animation(
    env_id: str,
    plan_history: list,
    trajectory: np.ndarray,
    start: np.ndarray,
    goal: np.ndarray,
    open_loop_horizon: int,
    namespace: str,
    interval: int = 100,
    plot_end_points: bool = True,
    batch_idx: int = 0,
) -> str:
    # - plan_history:
    #   contains for each time step all the MPC predicted plans for each pyramid noise
    #   level. Structured as a list of length (episode_len // open_loop_horizon), where
    #   each element corresponds to a control_time_step and stores a list of length
    #   pyramid_height, where each element is a plan at a different pyramid noise level
    #   and stored as a tensor of shape (episode_len // open_loop_horizon -
    #   control_time_step, batch_size, x_stacked_shape)

    # select index and prune history
    start, goal = start[batch_idx], goal[batch_idx]
    trajectory = trajectory[:, batch_idx]
    plan_history = [[pm[:, batch_idx] for pm in pt] for pt in plan_history]
    trajectory, plan_history = prune_history(
        plan_history, trajectory, goal, open_loop_horizon
    )

    # animate the convergence of the plans
    fig, ax = plt.subplots()
    if "large" in env_id:
        fig.set_size_inches(3.5, 5)
    else:
        fig.set_size_inches(3, 3)
    ax.set_axis_off()
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
    trajectory_colors = np.linspace(0, 1, len(trajectory))

    if is_grid_env(env_id):
        maze_grid = get_maze_grid(env_id)
    else:
        maze_grid = None

    def update(frame: int) -> None:
        control_time_step = 0
        while frame >= 0:
            frame -= len(plan_history[control_time_step])
            control_time_step += 1
        control_time_step -= 1
        m = frame + len(plan_history[control_time_step])
        num_steps_taken = 1 + open_loop_horizon * control_time_step
        plot_maze_layout(ax, maze_grid)

        plan_history_m = plan_history[control_time_step][m]
        plan_history_m = plan_history_m.numpy()
        ax.scatter(
            trajectory[:num_steps_taken, 0],
            trajectory[:num_steps_taken, 1],
            c=trajectory_colors[:num_steps_taken],
            cmap="Blues",
        )
        ax.scatter(
            plan_history_m[:, 0],
            plan_history_m[:, 1],
            c=np.arange(len(plan_history_m))[::-1],
            cmap="Reds",
        )

        if plot_end_points:
            plot_start_goal(ax, (start, goal))

    num_frames = sum([len(p) for p in plan_history])
    frames = tqdm(range(num_frames), desc="Making MPC animation")
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=interval)
    prefix = wandb.run.id if wandb.run is not None else env_id
    filename = f"/tmp/{prefix}_{namespace}_mpc.mp4"
    ani.save(filename, writer="ffmpeg", fps=24)

    return filename
