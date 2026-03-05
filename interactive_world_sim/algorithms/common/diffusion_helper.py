from typing import Any

import cv2
import numpy as np
import torch
from einops import rearrange
from tqdm import tqdm

from interactive_world_sim.utils.draw_utils import concat_img_h, concat_img_v
from interactive_world_sim.utils.normalizer import LinearNormalizer


@torch.no_grad()
def render_img(
    algo: Any,
    latent: torch.Tensor,
    resolution: int,
    normalizer: LinearNormalizer,
    num_views: int = 1,
) -> torch.Tensor:
    """Render an image conditioned on the latent state.

    Args:
        algo: The algorithm to use for rendering.
        latent: The latent state in shape of (B, D)
        resolution: The resolution of the rendered image.
        normalizer: The normalizer to use for rendering.
        num_views: The number of views to render.

    Returns:
        torch.Tensor: The rendered image in shape of (B, 3, resolution, resolution)
    """
    assert latent.ndim in [2, 4], "Latent state must have shape (B, D) or (B, C, H, W)"

    # create diffusion related variables
    schedules = np.arange(algo.sampling_timesteps, -1, -1)
    schedules = torch.from_numpy(schedules).to(algo.device)

    xs_pred = torch.randn(latent.shape[0], 3 * num_views, resolution, resolution)
    xs_pred = xs_pred.to(device=algo.device, dtype=algo.dtype)
    batch_size = 50
    for i in range(algo.sampling_timesteps):
        for j in range(0, xs_pred.shape[0], batch_size):
            actual_batch_size = xs_pred[j : j + batch_size].shape[0]
            curr_noise_level = schedules[i : i + 1].repeat(actual_batch_size)
            next_noise_level = schedules[i + 1 : i + 2].repeat(actual_batch_size)
            xs_pred[j : j + batch_size] = algo.diffusion_model.sample_step(
                xs_pred[j : j + batch_size],
                latent[j : j + batch_size],
                curr_noise_level=curr_noise_level,
                next_noise_level=next_noise_level,
            )

    # unnormalize xs and xs_pred and reorgnize them along B axis
    xs_pred_ls = []
    for c_i in range(num_views):
        curr_xs_pred = xs_pred[:, 3 * c_i : 3 * c_i + 3]
        curr_obs_key = algo.obs_keys[c_i]
        xs_pred_ls.append(normalizer[curr_obs_key].unnormalize(curr_xs_pred))
    xs_pred = torch.cat(xs_pred_ls, dim=1)  # (T, B * V, 3, H, W)
    assert xs_pred.shape == (latent.shape[0], 3 * num_views, resolution, resolution)
    xs_pred = xs_pred.clamp(0, 1)  # Clamp to [-1, 1]
    return xs_pred


@torch.no_grad()
def render_img_cm(
    algo: Any,
    latent: torch.Tensor,
    resolution: int,
    normalizer: LinearNormalizer,
    num_views: int = 1,
    batch_size: int = 50,
) -> torch.Tensor:
    """Render an image conditioned on the latent state.

    Args:
        algo: The algorithm to use for rendering.
        latent: The latent state in shape of (B, D)
        resolution: The resolution of the rendered image.
        normalizer: The normalizer to use for rendering.
        num_views: The number of views to render.
        batch_size: The batch size for rendering.

    Returns:
        torch.Tensor: The rendered image in shape of (B, 3, resolution, resolution)
    """
    assert latent.ndim in [2, 4], "Latent state must have shape (B, D) or (B, C, H, W)"

    xs_pred = torch.randn(latent.shape[0], 3 * num_views, resolution, resolution)
    xs_pred = xs_pred.to(device=algo.device, dtype=algo.dtype)
    curr_obs_key = algo.obs_keys[0]
    if hasattr(algo, "dec_infer_steps"):
        dec_infer_steps = algo.dec_infer_steps
    else:
        dec_infer_steps = 1
    for j in range(0, xs_pred.shape[0], batch_size):
        schedules = np.linspace(algo.timesteps - 1, 0, dec_infer_steps + 1)
        actual_batch_size = xs_pred[j : j + batch_size].shape[0]
        for step_i in range(dec_infer_steps):
            t = torch.tensor([schedules[step_i]], device=algo.device)
            s = torch.tensor([schedules[step_i + 1]], device=algo.device)
            t = t.repeat(actual_batch_size)
            s = s.repeat(actual_batch_size)
            t = t.long()
            s = s.long()

            xs_pred[j : j + batch_size] = algo._forward(  # noqa
                algo.decoder,
                xs_pred[j : j + batch_size],
                t,
                s,
                external_cond=latent[j : j + batch_size],
            )

            curr_obs_key = algo.obs_keys[0]

    # unnormalize xs and xs_pred and reorgnize them along B axis
    xs_pred_ls = []
    for c_i in range(num_views):
        curr_xs_pred = xs_pred[:, 3 * c_i : 3 * c_i + 3]
        curr_obs_key = algo.obs_keys[c_i]
        xs_pred_ls.append(normalizer[curr_obs_key].unnormalize(curr_xs_pred))
    xs_pred = torch.cat(xs_pred_ls, dim=1)  # (T, B * V, 3, H, W)
    assert xs_pred.shape == (latent.shape[0], 3 * num_views, resolution, resolution)
    xs_pred = xs_pred.clamp(0, 1)  # Clamp to [-1, 1]
    return xs_pred


@torch.no_grad()
def render_img_cm_mv(
    algo: Any,
    latent: torch.Tensor,
    resolution: int,
    normalizer: LinearNormalizer,
    num_views: int = 1,
    batch_size: int = 50,
) -> torch.Tensor:
    """Render an image conditioned on the latent state.

    Args:
        algo: The algorithm to use for rendering.
        latent: The latent state in shape of (B, D)
        resolution: The resolution of the rendered image.
        normalizer: The normalizer to use for rendering.
        num_views: The number of views to render.
        batch_size: The batch size for rendering.

    Returns:
        torch.Tensor: The rendered image in shape of (B, 3, resolution, resolution)
    """
    assert latent.ndim in [2, 4], "Latent state must have shape (B, D) or (B, C, H, W)"

    xs_pred = torch.randn(latent.shape[0], 3 * num_views, resolution, resolution)
    xs_pred = xs_pred.to(device=algo.device, dtype=algo.dtype)
    if hasattr(algo, "dec_infer_steps"):
        dec_infer_steps = algo.dec_infer_steps
    else:
        dec_infer_steps = 1
    for j in range(0, xs_pred.shape[0], batch_size):
        schedules = np.linspace(algo.timesteps - 1, 0, dec_infer_steps + 1)
        actual_batch_size = xs_pred[j : j + batch_size].shape[0]
        for step_i in range(dec_infer_steps):
            t = torch.tensor([schedules[step_i]], device=algo.device)
            s = torch.tensor([schedules[step_i + 1]], device=algo.device)
            t = t.repeat(actual_batch_size)
            s = s.repeat(actual_batch_size)
            t = t.long()
            s = s.long()

            for v_i in range(num_views):
                xs_pred[j : j + batch_size, v_i * 3 : (v_i + 1) * 3] = (
                    algo._forward(  # noqa
                        getattr(algo, f"decoder_{v_i}"),
                        xs_pred[j : j + batch_size, v_i * 3 : (v_i + 1) * 3],
                        t,
                        s,
                        external_cond=latent[
                            j : j + batch_size, v_i * 4 : (v_i + 1) * 4
                        ],
                    )
                )

    # unnormalize xs and xs_pred and reorgnize them along B axis
    xs_pred_ls = []
    for c_i in range(num_views):
        curr_xs_pred = xs_pred[:, 3 * c_i : 3 * c_i + 3]
        curr_obs_key = algo.obs_keys[c_i]
        xs_pred_ls.append(normalizer[curr_obs_key].unnormalize(curr_xs_pred))
    xs_pred = torch.cat(xs_pred_ls, dim=1)  # (T, B * V, 3, H, W)
    assert xs_pred.shape == (latent.shape[0], 3 * num_views, resolution, resolution)
    xs_pred = xs_pred.clamp(0, 1)  # Clamp to [-1, 1]
    return xs_pred


@torch.no_grad()
def render_img_flow(
    algo: Any,
    latent: torch.Tensor,
    resolution: int,
    normalizer: LinearNormalizer,
    num_views: int = 1,
    steps: int = 1,
) -> torch.Tensor:
    """Render an image conditioned on the latent state.

    Args:
        algo: The algorithm to use for rendering.
        latent: The latent state in shape of (B, D)
        resolution: The resolution of the rendered image.
        normalizer: The normalizer to use for rendering.
        num_views: The number of views to render.
        steps: The number of diffusion steps.

    Returns:
        torch.Tensor: The rendered image in shape of (B, 3, resolution, resolution)
    """
    assert latent.ndim in [2, 4], "Latent state must have shape (B, D) or (B, C, H, W)"

    xs_pred = torch.randn(latent.shape[0], 3 * num_views, resolution, resolution)
    xs_pred = xs_pred.to(device=algo.device, dtype=algo.dtype)
    batch_size = 50
    schedules = torch.linspace(0, 1.0, steps + 1)
    for j in range(0, xs_pred.shape[0], batch_size):
        actual_batch_size = xs_pred[j : j + batch_size].shape[0]
        for step_i in range(steps):
            t = torch.tensor([schedules[step_i]], device=algo.device)
            s = torch.tensor([schedules[step_i + 1]], device=algo.device)
            t = t.repeat(actual_batch_size)
            s = s.repeat(actual_batch_size)

            model = lambda x, t, external_cond=latent[j : j + batch_size]: algo.decoder(
                x, t, external_cond
            )
            xs_pred[j : j + batch_size] = algo.noise_scheduler.step(
                model,
                xs_pred[j : j + batch_size],
                t,
                s,
            )

    # unnormalize xs and xs_pred and reorgnize them along B axis
    xs_pred_ls = []
    for c_i in range(num_views):
        curr_xs_pred = xs_pred[:, 3 * c_i : 3 * c_i + 3]
        curr_obs_key = algo.obs_keys[c_i]
        xs_pred_ls.append(normalizer[curr_obs_key].unnormalize(curr_xs_pred))
    xs_pred = torch.cat(xs_pred_ls, dim=1)  # (T, B * V, 3, H, W)
    assert xs_pred.shape == (latent.shape[0], 3 * num_views, resolution, resolution)
    xs_pred = xs_pred.clamp(0, 1)  # Clamp to [-1, 1]
    return xs_pred


@torch.no_grad()
def predict_future_frames(
    algo: Any,
    batch: dict,
    n_frames: int,
    vis: bool = False,
) -> torch.Tensor:
    """Predict future frames for Diffusion Forcing."""
    obs_ls = [algo.normalizer[k].normalize(batch["obs"][k]) for k in algo.obs_keys]
    xs = torch.cat(obs_ls, dim=2)
    batch_size = xs.shape[0]
    xs = rearrange(xs, "b (t fs) c h w -> t b (fs c) h w", fs=algo.frame_stack)
    conditions = [algo.normalizer[k].normalize(batch["goal"][k]) for k in algo.obs_keys]
    conditions = torch.cat(conditions, dim=1)[None]
    curr_frame = 0

    # context
    n_context_frames = algo.context_frames // algo.frame_stack
    xs_pred = xs[:n_context_frames].clone()
    curr_frame += n_context_frames

    pbar = tqdm(total=n_frames, initial=curr_frame, desc="Sampling")
    while curr_frame < n_frames:
        if algo.chunk_size > 0:
            horizon = min(n_frames - curr_frame, algo.chunk_size)
        else:
            horizon = n_frames - curr_frame
        assert horizon <= algo.n_tokens, "horizon exceeds the number of tokens."
        scheduling_matrix = algo._generate_scheduling_matrix(horizon)  # noqa

        chunk = torch.randn(
            (horizon, batch_size, *algo.x_stacked_shape), device=algo.device
        )
        chunk = torch.clamp(chunk, -algo.clip_noise, algo.clip_noise)
        xs_pred = torch.cat([xs_pred, chunk], 0)

        # sliding window: only input the last n_tokens frames
        start_frame = max(0, curr_frame + horizon - algo.n_tokens)

        pbar.set_postfix(
            {
                "start": start_frame,
                "end": curr_frame + horizon,
            }
        )

        for m in range(scheduling_matrix.shape[0] - 1):
            from_noise_levels = np.concatenate(
                (np.zeros((curr_frame,), dtype=np.int64), scheduling_matrix[m])
            )[:, None].repeat(batch_size, axis=1)
            to_noise_levels = np.concatenate(
                (
                    np.zeros((curr_frame,), dtype=np.int64),
                    scheduling_matrix[m + 1],
                )
            )[:, None].repeat(batch_size, axis=1)

            from_noise_levels = torch.from_numpy(from_noise_levels).to(algo.device)
            to_noise_levels = torch.from_numpy(to_noise_levels).to(algo.device)

            # update xs_pred by DDIM or DDPM sampling
            # input frames within the sliding window
            xs_pred_pad = torch.tile(xs_pred[start_frame:], (1, 2, 1, 1, 1))
            if conditions is not None:
                cond_pad = torch.tile(conditions, (1, 2, 1, 1, 1))
            else:
                cond_pad = None
            from_noise_levels_pad = torch.tile(from_noise_levels[start_frame:], (1, 2))
            to_noise_levels_pad = torch.tile(to_noise_levels[start_frame:], (1, 2))
            external_cond_mask_pad = torch.cat(
                [torch.ones(batch_size), torch.zeros(batch_size)], dim=0
            )
            external_cond_mask_pad = external_cond_mask_pad.to(
                device=algo.device, dtype=torch.bool
            )
            xs_pred_compose = algo.diffusion_model.sample_step(
                xs_pred_pad,
                cond_pad,
                from_noise_levels_pad,
                to_noise_levels_pad,
                external_cond_mask=external_cond_mask_pad,
            )
            xs_pred[start_frame:] = (
                algo.guidance_scale * xs_pred_compose[:, batch_size:]
                + (1 - algo.guidance_scale) * xs_pred_compose[:, :batch_size]
            )

            if vis:
                xs_pred_np = xs_pred[start_frame:, 0].cpu().numpy()  # (T, 3, H, W)
                xs_pred_np = xs_pred_np / 2.0 + 0.5
                xs_pred_np = (xs_pred_np * 255).astype(np.uint8)
                xs_pred_np = np.transpose(xs_pred_np, (0, 2, 3, 1))
                goal_img = batch["goal"][algo.obs_keys[0]][0].cpu().numpy()
                goal_img = np.transpose(goal_img, (1, 2, 0))
                goal_img = (goal_img * 255.0).astype(np.uint8)
                boarder_color = np.array([255, 0, 0])
                boarder_width = 2
                goal_img[:boarder_width, :] = boarder_color
                goal_img[-boarder_width:, :] = boarder_color
                goal_img[:, :boarder_width] = boarder_color
                goal_img[:, -boarder_width:] = boarder_color
                xs_pred_np = np.concatenate([xs_pred_np, goal_img[None]], axis=0)
                concat_imgs = []
                for i in range(0, 12, 4):
                    if i >= len(xs_pred_np):
                        break
                    concat_chunks = xs_pred_np[i : i + 4]
                    if len(concat_chunks) < 4:
                        concat_chunks = np.concatenate(
                            [
                                concat_chunks,
                                np.zeros((4 - len(concat_chunks), 128, 128, 3)),
                            ],
                            axis=0,
                        )
                    concat_imgs.append(concat_img_h(concat_chunks).astype(np.uint8))
                xs_pred_np = concat_img_v(concat_imgs)
                xs_pred_np = cv2.cvtColor(
                    xs_pred_np, cv2.COLOR_RGB2BGR
                )  # (128, 128 * N, 3)

                original_H, original_W = xs_pred_np.shape[:2]
                xs_pred_np = cv2.resize(
                    xs_pred_np,
                    (original_W * 4, original_H * 4),
                    interpolation=cv2.INTER_AREA,
                )
                cv2.imshow("xs_pred", xs_pred_np)
                cv2.waitKey(1)
        xs_pred = xs_pred.clamp(-1, 1)

        curr_frame += horizon
        pbar.update(horizon)

    # unnormalize xs and xs_pred and reorgnize them along B axis
    xs_pred_ls = []
    for c_i in range(len(algo.obs_keys)):
        curr_xs_pred = xs_pred[:, :, 3 * c_i : 3 * c_i + 3]
        curr_obs_key = algo.obs_keys[c_i]
        xs_pred_ls.append(algo.normalizer[curr_obs_key].unnormalize(curr_xs_pred))
    xs_pred = torch.cat(xs_pred_ls, dim=1)  # (T, B * V, 3, H, W)

    xs_pred = rearrange(
        xs_pred, "t b (fs c) h w -> b (t fs) c h w", fs=algo.frame_stack
    )
    return xs_pred
