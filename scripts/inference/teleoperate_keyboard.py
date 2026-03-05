import math
import time
from pathlib import Path
from typing import Optional, Union

import cv2
import hydra
import lightning.pytorch as pl
import numpy as np
import torch
from einops import rearrange
from omegaconf import DictConfig, OmegaConf
from yixuan_utilities.draw_utils import center_crop
from yixuan_utilities.hdf5_utils import load_dict_from_hdf5, save_dict_to_hdf5
from yixuan_utilities.joystick_utils import Joystick
from yixuan_utilities.keyboard_utils import KeyReader
from yixuan_utilities.kinematics_helper import KinHelper

from interactive_world_sim.algorithms.common.diffusion_helper import render_img_cm
from interactive_world_sim.algorithms.latent_dynamics.latent_world_model import (
    LatentWorldModel,
)
from interactive_world_sim.datasets.latent_dynamics import (
    RealAlohaDataset,
    SimAlohaDataset,
)
from interactive_world_sim.utils.action_utils import joint_pos_to_action_primitive
from interactive_world_sim.utils.aloha_conts import (
    MASTER_GRIPPER_JOINT_UNNORMALIZE_FN,
    PUPPET_GRIPPER_JOINT_NORMALIZE_FN,
)
from interactive_world_sim.utils.draw_utils import (
    concat_img_h,
    plot_single_3d_pos_traj,
)
from interactive_world_sim.utils.normalizer import LinearNormalizer


def load_model(ckpt_path: str) -> pl.LightningModule:
    """Build the lightning module

    :return:  a pytorch-lightning module to be launched
    """
    cfg_path = Path(ckpt_path).parent.parent / ".hydra" / "config.yaml"
    cfg = OmegaConf.load(cfg_path)
    dtype = torch.float32 if "dtype" not in cfg.algorithm else cfg.algorithm.dtype
    # cfg.algorithm.dec_infer_steps = 1
    cfg.n_frames = 10
    cfg.algorithm.n_frames = 10
    if "diffusion" in cfg.algorithm and "sampling_timesteps" in cfg.algorithm.diffusion:
        cfg.algorithm.diffusion.sampling_timesteps = 10

    if (
        "diffusion" in cfg.algorithm.dynamics
        and "sampling_timesteps" in cfg.algorithm.dynamics.diffusion
    ):
        cfg.algorithm.dynamics.diffusion.sampling_timesteps = 10
    cfg.algorithm.load_ae = None
    algo = LatentWorldModel.load_from_checkpoint(
        ckpt_path,
        cfg=cfg.algorithm,
        map_location="cuda:0",
        dtype=dtype,
        strict=False,
        weights_only=False,
    )
    algo.dynamics = algo.dynamics.to(dtype)
    algo.eval()
    algo.dynamics.eval()
    return algo


def process_img(img: np.ndarray) -> np.ndarray:
    # crop
    h = w = 128
    img = center_crop(img, (h, w))
    img = cv2.resize(img, (h, w), cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return img


def build_dataset(cfg: DictConfig, split: str) -> Optional[torch.utils.data.Dataset]:
    # build the dataset
    compatible_datasets = {
        "sim_aloha_dataset": SimAlohaDataset,
        "real_aloha_dataset": RealAlohaDataset,
    }
    dataset = compatible_datasets[cfg.dataset._name](cfg.dataset)  # noqa
    if split == "training":
        return dataset
    elif split == "validation":
        return dataset.get_validation_dataset()
    elif split == "test":
        return dataset
    else:
        raise NotImplementedError(f"split '{split}' is not implemented")


def read_joystick(ctrl: Joystick) -> np.ndarray:
    axes, hats, _ = ctrl.read_joystick()
    x_l = axes[0]
    y_l = -axes[1]
    x_r = axes[3]
    y_r = -axes[4]
    lb = hats[4]
    rb = hats[5]
    B = hats[1]
    if np.abs([x_l, y_l, x_r, y_r]).max() < 0.1 and np.abs([lb, rb, B]).max() == 0:
        return np.zeros(4), np.zeros(3)
    else:
        if np.linalg.norm([x_l, y_l]) > 1.0:
            x_l /= np.linalg.norm([x_l, y_l])
            y_l /= np.linalg.norm([x_l, y_l])
        if np.linalg.norm([x_r, y_r]) > 1.0:
            x_r /= np.linalg.norm([x_r, y_r])
            y_r /= np.linalg.norm([x_r, y_r])
        return np.array([x_l, y_l, x_r, y_r]), np.array([lb, rb, B])


def read_keyboard(ctrl: KeyReader, scene: str) -> np.ndarray:
    keys = ctrl.read()
    signal = np.zeros(3)
    if scene in [
        "real",
        "real_cam_0",
        "sim",
        "bimanual_sweep_cam_0",
        "bimanual_sweep_cam_1",
        "single_grasp_cam_0",
        "single_grasp_cam_1",
    ]:
        delta_action = np.zeros(4)
        if "w" in keys:
            delta_action[1] = 1
        if "s" in keys:
            delta_action[1] = -1
        if "a" in keys:
            delta_action[0] = -1
        if "d" in keys:
            delta_action[0] = 1
        if "i" in keys:
            delta_action[3] = 1
        if "k" in keys:
            delta_action[3] = -1
        if "j" in keys:
            delta_action[2] = -1
        if "l" in keys:
            delta_action[2] = 1
    elif scene in ["bimanual_rope_cam_0", "bimanual_rope_cam_1"]:
        delta_action = np.zeros(6)
        if "w" in keys:
            delta_action[1] = 1
        if "s" in keys:
            delta_action[1] = -1
        if "a" in keys:
            delta_action[0] = -1
        if "d" in keys:
            delta_action[0] = 1
        if "i" in keys:
            delta_action[4] = 1
        if "k" in keys:
            delta_action[4] = -1
        if "j" in keys:
            delta_action[3] = -1
        if "l" in keys:
            delta_action[3] = 1
        if "q" in keys:
            delta_action[2] = 1
        if "e" in keys:
            delta_action[2] = -1
        if "u" in keys:
            delta_action[5] = 1
        if "o" in keys:
            delta_action[5] = -1

    if "c" in keys:
        signal[0] = 1
    elif "s" in keys:
        signal[1] = 1
    elif "q" in keys:
        signal[2] = 1
    return delta_action / (np.linalg.norm(delta_action) + 1e-8), signal


def read_dataset(ctrl: np.ndarray, time_idx: int) -> np.ndarray:
    return ctrl[time_idx]


def dict_list_to_np(episode: dict) -> dict:
    for key in list(episode.keys()):
        if isinstance(episode[key], list):
            episode[key] = np.stack(episode[key], axis=0)
        elif isinstance(episode[key], dict):
            episode[key] = dict_list_to_np(episode[key])
    return episode


def kybd_action_to_rob_action(delta_action: np.ndarray, scene: str) -> np.ndarray:
    if scene == "real":
        delta_action_rob = np.array(
            [
                -delta_action[3],
                delta_action[2],
                -delta_action[1],
                delta_action[0],
            ]
        )
    elif scene == "real_cam_0":
        delta_action_rob = np.array(
            [
                delta_action[1],
                -delta_action[0],
                delta_action[3],
                -delta_action[2],
            ]
        )
    elif scene == "bimanual_rope_cam_0":
        delta_action_rob = np.array(
            [
                delta_action[1],
                -delta_action[0],
                delta_action[2],
                0.0,
                delta_action[4],
                -delta_action[3],
                delta_action[5],
                0.0,
            ]
        )
    elif scene == "bimanual_rope_cam_1":
        delta_action_rob = np.array(
            [
                -delta_action[4],
                delta_action[3],
                delta_action[5],
                0.0,
                -delta_action[1],
                delta_action[0],
                delta_action[2],
                0.0,
            ]
        )
    elif scene == "sim":
        delta_action_rob = np.array(
            [
                delta_action[0],
                delta_action[1],
                delta_action[2],
                delta_action[3],
            ]
        )
    elif scene == "bimanual_sweep_cam_1":
        delta_action_rob = np.array(
            [
                -delta_action[3],
                delta_action[2],
                -delta_action[1],
                delta_action[0],
            ]
        )
    elif scene == "bimanual_sweep_cam_0":
        delta_action_rob = np.array(
            [
                delta_action[1],
                -delta_action[0],
                delta_action[3],
                -delta_action[2],
            ]
        )
    elif scene == "single_grasp_cam_1":
        delta_action_rob = np.array(
            [
                -delta_action[3],
                delta_action[2],
                -delta_action[1],
                delta_action[0],
            ]
        )
    elif scene == "single_grasp_cam_0":
        delta_action_rob = np.array(
            [
                delta_action[1],
                -delta_action[0],
                delta_action[3],
                -delta_action[2],
            ]
        )
    else:
        raise NotImplementedError(f"scene '{scene}' not recognized")
    return delta_action_rob


def rob_action_to_kybd_action(delta_action: np.ndarray, scene: str) -> np.ndarray:
    if scene == "real":
        delta_action_kybd = np.array(
            [
                delta_action[3],
                -delta_action[2],
                delta_action[1],
                -delta_action[0],
            ]
        )
    elif scene == "real_cam_0":
        delta_action_kybd = np.array(
            [
                -delta_action[1],
                delta_action[0],
                -delta_action[3],
                delta_action[2],
            ]
        )
    elif scene == "sim":
        delta_action_kybd = np.array(
            [
                delta_action[0],
                delta_action[1],
                delta_action[2],
                delta_action[3],
            ]
        )
    return delta_action_kybd


def record_one_episode(
    controller: Union[Joystick, KeyReader, np.ndarray],
    models: list[LatentWorldModel],
    normalizer: LinearNormalizer,
    dt: float,
    resolution: int,
    curr_latent_tensor_list: list[torch.Tensor],
    curr_action: torch.Tensor,
    obs_key: str,
    episode_id: int,
    output_dir: str,
    act_horizon: int,
    scene: str,
) -> bool:
    """Record one episode using world model."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(f"{output_dir}/out_vid").mkdir(parents=True, exist_ok=True)
    episode_data: dict = {
        "action": [],
        "obs": {
            "images": {
                obs_key: [],
            }
        },
    }
    start_recording = False
    save_recording = False
    dtype = models[0].dtype
    device = models[0].device
    curr_latent_tensor_list = [ts.to(dtype) for ts in curr_latent_tensor_list]
    curr_action = curr_action.to(dtype)

    fovy = 45.0
    H = 512
    W = 512
    f = 0.5 * H / math.tan(fovy * math.pi / 360)
    cx = W / 2
    cy = H / 2
    fx = f
    fy = f
    intrinsics = np.array([cx, cy, fx, fy])
    extrinsics = np.array(
        [[1, 0, 0, 0], [0, -1, 0, -0.019], [0, 0, -1, 0.685], [0, 0, 0, 1]]
    )
    xs_pred_ls = [
        render_img_cm(
            model,
            curr_latent_tensor[:, -1],
            resolution,
            normalizer=normalizer,
            num_views=len(model.obs_keys),
        )
        for model, curr_latent_tensor in zip(
            models, curr_latent_tensor_list, strict=False
        )
    ]
    xs_pred_vis_ls = []
    for v_i in range(len(models)):
        xs_pred_np = (
            xs_pred_ls[v_i].permute(0, 2, 3, 1).detach().cpu().float().numpy()[0]
        )
        xs_pred_np = (xs_pred_np * 255).astype(np.uint8)
        xs_pred_np = np.clip(xs_pred_np, 0, 255)
        xs_pred_vis = cv2.resize(
            xs_pred_np,
            (640, 640),
            interpolation=cv2.INTER_AREA,
        )
        xs_pred_vis = cv2.cvtColor(xs_pred_vis, cv2.COLOR_RGB2BGR)
        xs_pred_vis_ls.append(xs_pred_vis)
    xs_pred_vis = concat_img_h(xs_pred_vis_ls)
    if scene == "sim":
        curr_action_unnorm = normalizer["action"].unnormalize(curr_action)
        curr_action_unnorm = curr_action_unnorm.detach().cpu().numpy()
        action_3d = curr_action_unnorm.reshape(2, 1, 2)
        action_3d = np.concatenate([action_3d, 0.02 * np.ones((2, 1, 1))], axis=-1)
        xs_pred_vis = plot_single_3d_pos_traj(
            xs_pred_vis, intrinsics, extrinsics, action_3d, radius=5
        )

    delta_action = np.zeros(4)
    concat_img = xs_pred_vis
    text_label = f"Episode: {episode_id}"
    cv2.putText(
        concat_img,
        text_label,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )
    cv2.imshow("pred", concat_img)
    cv2.waitKey(100)

    out_vid = cv2.VideoWriter(
        f"{output_dir}/out_vid/{episode_id}.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,
        (concat_img.shape[1], concat_img.shape[0]),
    )
    action_ls = []
    skip_frame = 1
    time_idx = skip_frame
    hist_context = 10
    action_hist = []
    step_i = 0
    if type(controller) == np.ndarray:
        start_recording = True
    while True:
        # read input from device
        if isinstance(controller, Joystick):
            delta_action, signal = read_joystick(controller)
        elif isinstance(controller, np.ndarray):
            if time_idx >= controller.shape[0]:
                break
            next_action = read_dataset(controller, time_idx)
            next_action_norm = normalizer["action"].normalize(
                torch.from_numpy(next_action)
            )
            next_action_norm = next_action_norm.to(device).float()
            delta_action = (next_action_norm - curr_action).detach().cpu().numpy()
            time_idx += skip_frame
            signal = np.zeros(3)
            delta_action *= 50.0
            delta_action = rob_action_to_kybd_action(delta_action, scene=scene)
        else:
            delta_action, signal = read_keyboard(controller, scene=scene)

        if signal[0] == 1:
            start_recording = True
        if signal[1] == 1 and start_recording:
            start_recording = False
            save_recording = True
            break
        if signal[2] == 1 and start_recording:
            start_recording = False
            save_recording = False
            break

        if isinstance(controller, np.ndarray):
            curr_action = next_action_norm
        else:
            action_max = (
                models[0]
                .normalizer["action"]
                .state_dict()["params_dict.input_stats.max"]
            )
            action_min = (
                models[0]
                .normalizer["action"]
                .state_dict()["params_dict.input_stats.min"]
            )
            action_range = action_max - action_min
            if scene in ["bimanual_rope_cam_0", "bimanual_rope_cam_1"]:
                action_range = torch.cat([action_range[:3], action_range[4:7]]).to(
                    device
                )
            action_range_scale = action_range / action_range.max()
            action_range_scale = action_range_scale.detach().cpu().numpy()
            if scene in [
                "real",
                "real_cam_0",
            ]:
                delta_action = delta_action / (50.0 * action_range_scale)
            elif scene in ["bimanual_rope_cam_0", "bimanual_rope_cam_1"]:
                delta_action = delta_action / (30.0 * action_range_scale)
            elif scene in ["bimanual_sweep_cam_0", "bimanual_sweep_cam_1"]:
                delta_action = delta_action / 20.0
            elif scene == "sim":
                delta_action = delta_action / (100.0 * action_range_scale)
            elif scene in ["single_grasp_cam_0", "single_grasp_cam_1"]:
                delta_action[:3] = (
                    delta_action[:3]
                    * action_range_scale[:3].max()
                    / (50.0 * action_range_scale[:3])
                )
                delta_action[3] = delta_action[3] / 10.0
            else:
                raise NotImplementedError(f"scene '{scene}' not recognized")
            if np.linalg.norm(delta_action) < 1e-8:
                continue
            delta_action_rob = kybd_action_to_rob_action(delta_action, scene=scene)
            delta_action_tensor = torch.from_numpy(delta_action_rob).to(device)
            curr_action = curr_action + delta_action_tensor
        # curr_action_real = normalizer["action"].unnormalize(curr_action)
        # curr_action_real = curr_action_real + delta_action_tensor
        # curr_action = normalizer["action"].normalize(curr_action_real)

        start_time = time.time()

        curr_action = torch.clamp(curr_action, -1.0, 1.0)
        action_ls.append(curr_action)
        if len(action_ls) == act_horizon:
            action_chunk = torch.stack(action_ls)  # (act_horizon, 4)
            action_chunk = action_chunk.reshape(1, -1)
            if start_recording:
                action_chunk_unnorm = normalizer["action"].unnormalize(action_chunk)
                action_chunk_unnorm = action_chunk_unnorm.detach().cpu().numpy()
                episode_data["action"].append(action_chunk_unnorm)
                episode_data["obs"]["images"][obs_key].append(xs_pred_np)
            action_hist.append(action_chunk)
            action = torch.cat(action_hist, dim=0)[-(hist_context + 1) :]
            action = rearrange(action, "t a -> 1 t a")
            action = action.to(device=device, dtype=dtype)
            for i in range(len(models)):
                with torch.no_grad():
                    latent_pred = models[i].dynamics_forward(
                        curr_latent_tensor_list[i], action
                    )
                curr_latent_tensor_list[i] = torch.cat(
                    [curr_latent_tensor_list[i], latent_pred], axis=1
                )
                curr_latent_tensor_list[i] = curr_latent_tensor_list[i][
                    :, -hist_context:
                ]
            action_ls = []

            # render the predicted image
            xs_pred_ls = [
                render_img_cm(
                    model,
                    curr_latent_tensor[:, -1],
                    resolution,
                    normalizer=normalizer,
                    num_views=len(model.obs_keys),
                )
                for model, curr_latent_tensor in zip(
                    models, curr_latent_tensor_list, strict=False
                )
            ]
            xs_pred_vis_ls = []
            for v_i in range(len(models)):
                xs_pred_tensor = xs_pred_ls[v_i].permute(0, 2, 3, 1)
                xs_pred_np = xs_pred_tensor.detach().cpu().float().numpy()[0]
                xs_pred_np = (xs_pred_np * 255).astype(np.uint8)
                xs_pred_np = np.clip(xs_pred_np, 0, 255)
                xs_pred_vis = cv2.resize(
                    xs_pred_np,
                    (640, 640),
                    interpolation=cv2.INTER_AREA,
                )
                xs_pred_vis = cv2.cvtColor(xs_pred_vis, cv2.COLOR_RGB2BGR)
                xs_pred_vis_ls.append(xs_pred_vis)
            xs_pred_vis = concat_img_h(xs_pred_vis_ls)
        else:
            continue
        concat_img = xs_pred_vis
        out_vid.write(concat_img)
        if start_recording:
            text_label = f"Episode: {episode_id} [Recording]"
        else:
            text_label = f"Episode: {episode_id}"
        cv2.putText(
            concat_img,
            text_label,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.imshow("pred", concat_img)
        cv2.waitKey(1)

        step_i += 1
        print("freq:", 1 / (time.time() - start_time))
        print("steps:", step_i)
        # print("dyn_time:", dyn_time)
        # print("render_time:", render_time)
        print()
        # if step_i >= 300:
        #     return False
    if type(controller) == np.ndarray:
        save_recording = True
    if save_recording:
        episode_data["action"] = np.stack(episode_data["action"], axis=0)
        episode_data["obs"]["images"][obs_key] = np.stack(
            episode_data["obs"]["images"][obs_key], axis=0
        )
        config_dict: dict = {
            "obs": {"images": {}},
        }
        episode_data = dict_list_to_np(episode_data)
        color_save_kwargs = {
            "chunks": (1, resolution, resolution, 3),
            "dtype": "uint8",
        }
        config_dict["obs"]["images"][obs_key] = color_save_kwargs
        save_dict_to_hdf5(
            episode_data,
            config_dict,
            f"{output_dir}/episode_{episode_id}.hdf5",
        )
        out_vid.release()
    return save_recording


@hydra.main(
    version_base=None,
    config_path="../../configurations",
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    """Collect demonstration for the Push-T task.

    Usage: python demo_pusht.py -o data/pusht_demo.zarr

    This script is compatible with both Linux and MacOS.
    Hover mouse close to the blue circle to start.
    Push the T block into the green area.
    The episode will automatically terminate if the task is succeeded.
    Press "Q" to exit.
    Press "R" to retry.
    Hold "Space" to pause.
    """
    obs_keys = cfg.dataset.obs_keys
    output_dir = cfg.output_dir

    # build algo and load ckpt
    models: list[LatentWorldModel] = [
        load_model(ckpt_path) for ckpt_path in cfg.ckpt_paths
    ]
    normalizer = models[0].normalizer

    # measure model size
    for model in models:
        total_params = sum(p.numel() for p in model.parameters())
        param_size_mb = total_params * 4 / (1024**2)  # Assuming float32

        print(f"\n{'='*60}")
        print("Model Size Statistics:")
        print(f"{'='*60}")
        print(f"Total parameters: {total_params / 1e6} M parameters")
        print(f"Estimated size (float32): {param_size_mb:.2f} MB")
        print(f"{'='*60}\n")

    # set up env
    dt = 1 / 30.0
    device = models[0].device
    if cfg.use_joystick:
        controller = Joystick()
    elif cfg.use_dataset:
        pass
    else:
        controller = KeyReader()
    # episqode_id = len(list(Path(output_dir).glob("episode_*.hdf5")))
    # episode_id = 28
    episode_id = 0
    # episode_id = 54
    # episode_id = 93
    while True:
        # t = np.random.randint(0, dataset.replay_buffer["action"].shape[0] - 1)
        load_epi_path = f"{cfg.dataset.dataset_dir}/episode_{episode_id}.hdf5"
        if not Path(load_epi_path).exists():
            break
        load_epi_data, _ = load_dict_from_hdf5(load_epi_path)
        if cfg.use_dataset:
            controller = load_epi_data["action"][()]
        t = 0

        # get the action at time t
        if cfg.scene in ["sim"]:
            curr_action = load_epi_data["action"][t]
        else:
            joint_pos = load_epi_data["obs"]["joint_pos"][t]
            num_rob = joint_pos.shape[0] // 7
            for r_i in range(num_rob):
                joint_pos[r_i * 7 + 6] = MASTER_GRIPPER_JOINT_UNNORMALIZE_FN(
                    PUPPET_GRIPPER_JOINT_NORMALIZE_FN(joint_pos[r_i * 7 + 6])
                )

            kin_helper = KinHelper("trossen_vx300s")
            if cfg.scene in ["bimanual_rope_cam_0", "bimanual_rope_cam_1"]:
                ctrl_mode = "bimanual_rope"
            elif cfg.scene in ["bimanual_sweep_cam_0", "bimanual_sweep_cam_1"]:
                ctrl_mode = "bimanual_sweep_v2"
            elif cfg.scene in ["single_grasp_cam_0", "single_grasp_cam_1"]:
                ctrl_mode = "single_grasp"
            elif cfg.scene in ["real", "real_cam_0", "sim"]:
                ctrl_mode = "bimanual_push"
            else:
                raise NotImplementedError(f"scene '{cfg.scene}' not recognized")
            robot_bases = (
                load_epi_data["robot_bases"][t]
                if "robot_bases" in load_epi_data
                else load_epi_data["obs"]["world_t_robot_base"][t]
            )
            curr_action = joint_pos_to_action_primitive(
                joint_pos=joint_pos,
                ctrl_mode=ctrl_mode,
                base_pose_in_world=robot_bases,
                kin_helper=kin_helper,
            )
        # curr_action = load_epi_data["obs"]["ee_pos"][t, :, :2, 3].reshape(-1)
        curr_action = torch.from_numpy(curr_action).to(device).float()
        curr_action = normalizer["action"].normalize(curr_action)

        # get image at time t
        img_tensor_list = []
        curr_latent_tensor_list = []
        for o_i, obs_key in enumerate(obs_keys):
            raw_img = load_epi_data["obs"]["images"][obs_key][t]
            raw_img = center_crop(
                raw_img, (cfg.dataset.resolution, cfg.dataset.resolution)
            )
            raw_img = cv2.resize(
                raw_img,
                (cfg.dataset.resolution, cfg.dataset.resolution),
                interpolation=cv2.INTER_AREA,
            )
            img = raw_img / 255.0
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
            img_tensor = normalizer[obs_key].normalize(img_tensor)
            img_tensor = img_tensor.to(device)
            img_tensor_list.append(img_tensor)
            with torch.no_grad():
                curr_latent_tensor_list.append(
                    models[o_i].encoder_forward(img_tensor)[:, None]
                )
        img_tensor = torch.cat(img_tensor_list, dim=1)
        save_epi = record_one_episode(
            controller,
            models,
            normalizer,
            dt,
            cfg.dataset.resolution,
            curr_latent_tensor_list,
            curr_action,
            obs_key,
            episode_id,
            output_dir,
            cfg.act_horizon,
            cfg.scene,
        )
        if save_epi:
            episode_id += 1


if __name__ == "__main__":
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.register_new_resolver("torch", lambda x: getattr(torch, x))
    main()
