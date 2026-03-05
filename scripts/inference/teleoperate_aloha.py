import glob
import math
import os
import time
from multiprocessing.managers import SharedMemoryManager
from pathlib import Path
from typing import Any, List, Optional

import cv2
import hydra
import lightning.pytorch as pl
import numpy as np
import torch
from einops import rearrange
from interbotix_xs_modules.arm import InterbotixManipulatorXS
from omegaconf import DictConfig, OmegaConf
from yixuan_utilities.draw_utils import center_crop
from yixuan_utilities.hdf5_utils import load_dict_from_hdf5, save_dict_to_hdf5
from yixuan_utilities.kinematics_helper import KinHelper

from interactive_world_sim.algorithms.common.diffusion_helper import (
    render_img_cm,
)
from interactive_world_sim.algorithms.latent_dynamics.latent_world_model import (
    LatentWorldModel,
)
from interactive_world_sim.datasets.latent_dynamics import (
    RealAlohaDataset,
    SimAlohaDataset,
)
from interactive_world_sim.utils.action_utils import joint_pos_to_action_primitive
from interactive_world_sim.utils.aloha_conts import (
    DT,
    MASTER_GRIPPER_JOINT_CLOSE,
    MASTER_GRIPPER_JOINT_MID,
    MASTER_GRIPPER_JOINT_UNNORMALIZE_FN,
    PUPPET_GRIPPER_JOINT_NORMALIZE_FN,
    START_ARM_POSE,
)
from interactive_world_sim.utils.aloha_utils import (
    get_arm_gripper_positions,
    move_arms,
    move_grippers,
    torque_off,
    torque_on,
)
from interactive_world_sim.utils.draw_utils import (
    concat_img_h,
    plot_single_3d_pos_traj,
)
from interactive_world_sim.utils.keystroke_counter import Key, KeyCode, KeystrokeCounter
from interactive_world_sim.utils.normalizer import LinearNormalizer

SCENE_CTRL_MODE_MAPPING = {
    "bimanual_rope_cam_0": "bimanual_rope",
    "bimanual_rope_cam_1": "bimanual_rope",
    "bimanual_sweep_cam_0": "bimanual_sweep",
    "bimanual_sweep_cam_1": "bimanual_sweep",
    "single_grasp_cam_0": "single_grasp",
    "single_grasp_cam_1": "single_grasp",
    "real": "bimanual_push",
    "real_cam_0": "bimanual_push",
    "sim": "bimanual_push",
}

SCENE_ROBOT_SIDES_MAPPING = {
    "bimanual_rope_cam_0": ["right", "left"],
    "bimanual_rope_cam_1": ["right", "left"],
    "bimanual_sweep_cam_0": ["right", "left"],
    "bimanual_sweep_cam_1": ["right", "left"],
    "single_grasp_cam_0": ["right"],
    "single_grasp_cam_1": ["left"],
    "real": ["right", "left"],
    "real_cam_0": ["right", "left"],
    "sim": ["right", "left"],
}


def prep_robots(
    master_bot: InterbotixManipulatorXS, start_arm_qpos: Optional[np.ndarray] = None
) -> None:
    # reboot gripper motors, and set operating modes for all motors
    master_bot.dxl.robot_set_operating_modes("group", "arm", "position")
    master_bot.dxl.robot_set_operating_modes("single", "gripper", "position")
    torque_on(master_bot)

    # move arms to starting position
    if start_arm_qpos is None:
        start_arm_qpos = START_ARM_POSE[:6]
    move_arms([master_bot], [start_arm_qpos], move_time=1)
    # move grippers to starting position
    move_grippers([master_bot], [MASTER_GRIPPER_JOINT_MID], move_time=0.5)


def press_to_start(master_bot: InterbotixManipulatorXS) -> None:
    # press gripper to start data collection
    # disable torque for only gripper joint of master robot to allow user movement
    master_bot.dxl.robot_torque_enable("single", "gripper", False)
    print("Close the gripper to start")
    close_thresh = (MASTER_GRIPPER_JOINT_MID + MASTER_GRIPPER_JOINT_CLOSE) / 2.0
    pressed = False
    while not pressed:
        gripper_pos = get_arm_gripper_positions(master_bot)
        if gripper_pos < close_thresh:
            pressed = True
        time.sleep(DT / 10)
    time.sleep(1.0)
    torque_off(master_bot)
    print("Started!")


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


def read_keyboard(ctrl: KeystrokeCounter) -> np.ndarray:
    signal = np.zeros(3)

    press_events: List[Any] = ctrl.get_press_events()
    for key_stroke in press_events:
        if key_stroke == KeyCode(char="c"):
            # Start recording.
            signal[0] = 1
        elif key_stroke == KeyCode(char="s"):
            # Stop recording.
            signal[1] = 1
        elif key_stroke == Key.backspace:
            # Exit program.
            signal[2] = 1
    return signal


def read_aloha_controller(
    ctrl: list[InterbotixManipulatorXS],
    scene: str,
    kin_helper: KinHelper,
    base_pose_in_world: np.ndarray,
) -> np.ndarray:
    joint_pos_ls = []
    for master_bot in ctrl:
        joint_pos = master_bot.dxl.robot_get_joint_states().position[:7]
        joint_pos_ls.append(joint_pos)
    joint_pos = np.concatenate(joint_pos_ls)

    action = joint_pos_to_action_primitive(
        joint_pos=joint_pos,
        ctrl_mode=SCENE_CTRL_MODE_MAPPING[scene],
        base_pose_in_world=base_pose_in_world,
        kin_helper=kin_helper,
    )
    return action


def dict_list_to_np(episode: dict) -> dict:
    for key in list(episode.keys()):
        if isinstance(episode[key], list):
            episode[key] = np.stack(episode[key], axis=0)
        elif isinstance(episode[key], dict):
            episode[key] = dict_list_to_np(episode[key])
    return episode


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
    controller: list[InterbotixManipulatorXS],
    kybd_reader: KeystrokeCounter,
    models: list[LatentWorldModel],
    normalizer: LinearNormalizer,
    dt: float,
    resolution: int,
    curr_latent_tensor_list: list[torch.Tensor],
    curr_action: torch.Tensor,
    init_joint: np.ndarray,
    obs_key: str,
    episode_id: int,
    output_dir: str,
    act_horizon: int,
    scene: str,
    total_steps: int,
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
            (1024, 1024),
            interpolation=cv2.INTER_AREA,
        )
        xs_pred_vis = cv2.cvtColor(xs_pred_vis, cv2.COLOR_RGB2BGR)
        xs_pred_vis = xs_pred_vis[:, ::-1, :]  # horizontal flip
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
        10,
        (concat_img.shape[1], concat_img.shape[0]),
    )
    action_ls = []
    hist_context = 10
    action_hist = []
    step_i = 0
    kin_helper = KinHelper("trossen_vx300s")
    base_pose_in_world = np.tile(np.eye(4)[None], (2, 1, 1))
    curr_file_path = os.path.join(os.path.dirname(__file__))
    base_pose_in_world[0] = np.load(
        f"{curr_file_path}/../../interactive_world_sim/real_world/aloha_extrinsics/right_base_pose_in_world.npy"
    )
    base_pose_in_world[1] = np.load(
        f"{curr_file_path}/../../interactive_world_sim/real_world/aloha_extrinsics/left_base_pose_in_world.npy"
    )
    while True:
        start_time = time.time()

        signal = read_keyboard(kybd_reader)
        if signal[0] == 1:
            start_recording = True
            # initialize robots to the start joint positions
            curr_action = joint_pos_to_action_primitive(
                joint_pos=init_joint,
                ctrl_mode=SCENE_CTRL_MODE_MAPPING[scene],
                base_pose_in_world=base_pose_in_world,
                kin_helper=kin_helper,
            )
            for rob_i, master_bot in enumerate(controller):
                start_arm_qpos = init_joint[rob_i * 7 : rob_i * 7 + 6]
                prep_robots(master_bot, start_arm_qpos=start_arm_qpos)
                press_to_start(master_bot)
        if signal[1] == 1 and start_recording:
            start_recording = False
            save_recording = True
            break
        if signal[2] == 1 and start_recording:
            start_recording = False
            save_recording = False
            break
        if total_steps > 0 and step_i >= total_steps and start_recording:
            start_recording = False
            save_recording = True
            print(f"Reached total steps {total_steps}, stopping recording.")
            break
        if not start_recording:
            time.sleep(dt)
            continue

        curr_action = read_aloha_controller(
            controller,
            scene=scene,
            kin_helper=kin_helper,
            base_pose_in_world=base_pose_in_world,
        )
        curr_action_tensor = torch.tensor(curr_action, dtype=dtype).squeeze(0)
        curr_action = normalizer["action"].normalize(curr_action_tensor)
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
                    (1024, 1024),
                    interpolation=cv2.INTER_AREA,
                )
                xs_pred_vis = cv2.cvtColor(xs_pred_vis, cv2.COLOR_RGB2BGR)
                xs_pred_vis = xs_pred_vis[:, ::-1, :]  # horizontal flip
                xs_pred_vis_ls.append(xs_pred_vis)
            xs_pred_vis = concat_img_h(xs_pred_vis_ls)
            if scene == "sim":
                curr_action_unnorm = normalizer["action"].unnormalize(curr_action)
                curr_action_unnorm = curr_action_unnorm.detach().cpu().numpy()
                action_3d = curr_action_unnorm.reshape(2, 1, 2)
                action_3d = np.concatenate(
                    [action_3d, 0.02 * np.ones((2, 1, 1))], axis=-1
                )
                xs_pred_vis = plot_single_3d_pos_traj(
                    xs_pred_vis, intrinsics, extrinsics, action_3d, radius=5
                )
        else:
            continue
        concat_img = xs_pred_vis
        out_vid.write(concat_img)
        if start_recording and total_steps > 0:
            text_label = (
                f"Episode: {episode_id} [Recording] Steps: {step_i}/{total_steps}"
            )
        elif start_recording and total_steps <= 0:
            text_label = f"Episode: {episode_id} [Recording] Steps: {step_i}"
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

        if start_recording:
            step_i += 1
        time.sleep(max(0.0, dt - (time.time() - start_time)))
        print("freq:", 1 / (time.time() - start_time))
        print("steps:", step_i)
        # print("dyn_time:", dyn_time)
        # print("render_time:", render_time)
        print()
        # if step_i >= 300:
        #     return False
    if save_recording:
        episode_data["action"] = np.concatenate(episode_data["action"], axis=0)[:-1]
        episode_data["obs"]["images"][obs_key] = np.stack(
            episode_data["obs"]["images"][obs_key], axis=0
        )[:-1]
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
    total_steps = cfg.total_steps if "total_steps" in cfg else -1

    # build algo and load ckpt
    models: list[LatentWorldModel] = [
        load_model(ckpt_path) for ckpt_path in cfg.ckpt_paths
    ]
    normalizer = models[0].normalizer

    # set up env
    dt = 1 / 10.0
    shm_manager = SharedMemoryManager()
    shm_manager.start()
    robot_sides = SCENE_ROBOT_SIDES_MAPPING[cfg.scene]
    ctrl_mode = SCENE_CTRL_MODE_MAPPING[cfg.scene]
    controller = []

    kybd_reader = KeystrokeCounter()
    kybd_reader.start()

    for rob_i, side in enumerate(robot_sides):
        if rob_i == 0:
            init_node = True
        else:
            init_node = False
        controller.append(
            InterbotixManipulatorXS(
                robot_model="wx250s",
                group_name="arm",
                gripper_name="gripper",
                robot_name=f"master_{side}",
                init_node=init_node,
            )
        )
        prep_robots(controller[-1])
        press_to_start(controller[-1])
    existing_episode_ids = []
    for file_path in glob.glob(f"{output_dir}/episode_*.hdf5"):
        file_name = os.path.basename(file_path)
        episode_id = int(file_name.split("_")[1].split(".")[0])
        existing_episode_ids.append(episode_id)
    target_episode_ids = []
    for file_path in glob.glob(f"{cfg.dataset.dataset_dir}/episode_*.hdf5"):
        file_name = os.path.basename(file_path)
        episode_id = int(file_name.split("_")[1].split(".")[0])
        target_episode_ids.append(episode_id)
    missing_episode_ids = list(set(target_episode_ids) - set(existing_episode_ids))
    missing_episode_ids.sort()
    print("missing_episode_ids:", missing_episode_ids)
    missing_i = 0
    while missing_i < len(missing_episode_ids):
        episode_id = missing_episode_ids[missing_i]
        # t = np.random.randint(0, dataset.replay_buffer["action"].shape[0] - 1)
        load_epi_path = f"{cfg.dataset.dataset_dir}/episode_{episode_id}.hdf5"
        if not Path(load_epi_path).exists():
            break
        load_epi_data, _ = load_dict_from_hdf5(load_epi_path)
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
        device = models[0].device
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

        init_joint = load_epi_data["joint_action"][t]  # master joint pos

        save_epi = record_one_episode(
            controller,
            kybd_reader,
            models,
            normalizer,
            dt,
            cfg.dataset.resolution,
            curr_latent_tensor_list,
            curr_action,
            init_joint,
            obs_key,
            episode_id,
            output_dir,
            cfg.act_horizon,
            cfg.scene,
            total_steps=total_steps,
        )
        if save_epi:
            missing_i += 1


if __name__ == "__main__":
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.register_new_resolver("torch", lambda x: getattr(torch, x))
    main()
