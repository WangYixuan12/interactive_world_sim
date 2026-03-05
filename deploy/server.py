# deploy/server.py
import asyncio
import re
import struct
import time
from pathlib import Path
from typing import Any, Dict

import cv2
import lightning.pytorch as pl
import numpy as np
import torch
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from omegaconf import OmegaConf
from yixuan_utilities.draw_utils import center_crop
from yixuan_utilities.hdf5_utils import load_dict_from_hdf5
from yixuan_utilities.kinematics_helper import KinHelper

from interactive_world_sim.algorithms.common.diffusion_helper import render_img_cm
from interactive_world_sim.algorithms.latent_dynamics.latent_world_model import (
    LatentWorldModel,
)
from interactive_world_sim.utils.action_utils import joint_pos_to_action_primitive
from interactive_world_sim.utils.aloha_conts import (
    MASTER_GRIPPER_JOINT_UNNORMALIZE_FN,
    PUPPET_GRIPPER_JOINT_NORMALIZE_FN,
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

curr_file_dir = Path(__file__).parent.resolve()
repo_root = curr_file_dir.parent

# Serve per-task init images as static files
for _task_dir_name in [
    "real_pusht_topdown",
    "real_bimanual_rope_cam0",
    "real_single_grasp_cam0",
    "real_bimanual_sweep_cam0",
]:
    _task_dir = curr_file_dir / _task_dir_name
    _task_dir.mkdir(exist_ok=True)
    app.mount(
        f"/{_task_dir_name}", StaticFiles(directory=str(_task_dir)), name=_task_dir_name
    )


@app.get("/")
def root() -> HTMLResponse:
    return HTMLResponse((curr_file_dir / "index.html").read_text())


OmegaConf.register_new_resolver("eval", lambda expr: eval(expr, {"np": np}))
OmegaConf.register_new_resolver("torch", lambda x: getattr(torch, x))


def _fix_target_paths(cfg_str: str) -> str:
    """Update _target_ paths to use the interactive_world_sim package."""
    cfg_str = re.sub(
        r"(_target_:\s*)algorithms\.",
        r"\1interactive_world_sim.algorithms.",
        cfg_str,
    )
    cfg_str = re.sub(
        r"(_target_:\s*)utils\.",
        r"\1interactive_world_sim.utils.",
        cfg_str,
    )
    return cfg_str


def load_model(algo_name: str, ckpt_path: str) -> pl.LightningModule:
    # Config lives at <task_dir>/.hydra/config.yaml where task_dir is two levels up
    # from the checkpoint: <task_dir>/checkpoints/best.ckpt
    cfg_path = Path(ckpt_path).parent.parent / ".hydra" / "config.yaml"
    cfg_str = _fix_target_paths(cfg_path.read_text())
    cfg = OmegaConf.create(cfg_str)
    dtype = torch.float32 if "dtype" not in cfg.algorithm else cfg.algorithm.dtype
    cfg.algorithm.load_ae = None
    cfg.n_frames = 10
    cfg.algorithm.n_frames = 10
    compatible_algorithms = {"latent_world_model": LatentWorldModel}
    algo = compatible_algorithms[algo_name].load_from_checkpoint(
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


act_horizon = 1
algo_name = "latent_world_model"
t = 0
hist_context = 10


def load_task_config(
    scene: str,
) -> tuple[int, LatentWorldModel, Any, torch.Tensor, np.ndarray]:
    data_dir = repo_root / "data" / "mini"

    if scene == "pusht":
        resolution = 128
        ckpt_path = str(
            repo_root / "outputs" / "pusht_cam1" / "checkpoints" / "best.ckpt"
        )
        dataset_path = str(data_dir / "pusht" / "val" / "episode_0.hdf5")
        obs_keys = ["camera_1_color"]
    elif scene == "bimanual_rope_cam_0":
        resolution = 128
        ckpt_path = str(
            repo_root / "outputs" / "bimanual_rope_cam0" / "checkpoints" / "best.ckpt"
        )
        dataset_path = str(data_dir / "bimanual_rope" / "val" / "episode_0.hdf5")
        obs_keys = ["camera_0_color"]
    elif scene == "single_grasp_cam_0":
        resolution = 128
        ckpt_path = str(
            repo_root / "outputs" / "single_grasp_cam0" / "checkpoints" / "best.ckpt"
        )
        dataset_path = str(data_dir / "single_grasp" / "val" / "episode_0.hdf5")
        obs_keys = ["camera_0_color"]
    elif scene == "bimanual_sweep_cam_0":
        resolution = 128
        ckpt_path = str(
            repo_root / "outputs" / "bimanual_sweep_cam0" / "checkpoints" / "best.ckpt"
        )
        dataset_path = str(data_dir / "bimanual_sweep" / "val" / "episode_0.hdf5")
        obs_keys = ["camera_0_color"]
    else:
        raise ValueError(f"Unknown scene: '{scene}'")

    model: LatentWorldModel = load_model(algo_name=algo_name, ckpt_path=ckpt_path)
    normalizer = model.normalizer
    load_epi_data, _ = load_dict_from_hdf5(dataset_path)

    # Build initial latent from the first frame of the episode
    img_tensor_list = []
    for obs_key in obs_keys:
        raw_img = load_epi_data["obs"]["images"][obs_key][t]
        raw_img = center_crop(raw_img, (resolution, resolution))
        raw_img = cv2.resize(
            raw_img, (resolution, resolution), interpolation=cv2.INTER_AREA
        )
        img = raw_img / 255.0
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        img_tensor = normalizer[obs_key].normalize(img_tensor)
        img_tensor = img_tensor.to(model.device)
        img_tensor_list.append(img_tensor)
    img_tensor = torch.cat(img_tensor_list, dim=1)
    with torch.no_grad():
        curr_latent_tensor = model.encoder_forward(img_tensor)[None]

    # Build initial action from joint positions
    joint_pos = load_epi_data["obs"]["joint_pos"][t]
    num_rob = joint_pos.shape[0] // 7
    for r_i in range(num_rob):
        joint_pos[r_i * 7 + 6] = MASTER_GRIPPER_JOINT_UNNORMALIZE_FN(
            PUPPET_GRIPPER_JOINT_NORMALIZE_FN(joint_pos[r_i * 7 + 6])
        )

    kin_helper = KinHelper("trossen_vx300s")
    if scene in ["bimanual_rope_cam_0", "bimanual_rope_cam_1"]:
        ctrl_mode = "bimanual_rope"
    elif scene in ["bimanual_sweep_cam_0", "bimanual_sweep_cam_1"]:
        ctrl_mode = "bimanual_sweep"
    elif scene in ["single_grasp_cam_0", "single_grasp_cam_1"]:
        ctrl_mode = "single_grasp"
    elif scene in ["pusht", "pusht_cam_0"]:
        ctrl_mode = "bimanual_push"
    else:
        raise NotImplementedError(f"scene '{scene}' not recognized")

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
    return resolution, model, normalizer, curr_latent_tensor, curr_action


def encode_frame_jpeg(frame: np.ndarray) -> bytes:
    # frame: uint8 HWC RGB — OpenCV expects BGR
    bgr = frame[:, :, ::-1]
    ok, buf = cv2.imencode(".png", bgr, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    if not ok:
        return b""
    return buf.tobytes()


def parse_action(msg: Dict[str, Any], scene: str) -> np.ndarray:
    if scene in ["pusht", "bimanual_sweep_cam_0", "single_grasp_cam_0"]:
        delta_action = np.zeros(4)
        if msg["w"] == 1:
            delta_action[1] = 1
        if msg["s"] == 1:
            delta_action[1] = -1
        if msg["a"] == 1:
            delta_action[0] = -1
        if msg["d"] == 1:
            delta_action[0] = 1
        if msg["i"] == 1:
            delta_action[3] = 1
        if msg["k"] == 1:
            delta_action[3] = -1
        if msg["j"] == 1:
            delta_action[2] = -1
        if msg["l"] == 1:
            delta_action[2] = 1
    elif scene == "bimanual_rope_cam_0":
        delta_action = np.zeros(6)
        if msg["w"] == 1:
            delta_action[1] = 1
        if msg["s"] == 1:
            delta_action[1] = -1
        if msg["a"] == 1:
            delta_action[0] = -1
        if msg["d"] == 1:
            delta_action[0] = 1
        if msg["q"] == 1:
            delta_action[2] = -1
        if msg["e"] == 1:
            delta_action[2] = 1
        if msg["i"] == 1:
            delta_action[4] = 1
        if msg["k"] == 1:
            delta_action[4] = -1
        if msg["j"] == 1:
            delta_action[3] = -1
        if msg["l"] == 1:
            delta_action[3] = 1
        if msg["u"] == 1:
            delta_action[5] = 1
        if msg["o"] == 1:
            delta_action[5] = -1
    return delta_action


def kybd_action_to_rob_action(delta_action: np.ndarray, scene: str) -> np.ndarray:
    if scene == "pusht":
        delta_action_rob = np.array(
            [
                -delta_action[3],
                delta_action[2],
                -delta_action[1],
                delta_action[0],
            ]
        )
    elif scene in ["bimanual_sweep_cam_0", "single_grasp_cam_0"]:
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
    else:
        raise NotImplementedError(f"scene '{scene}' not recognized")
    return delta_action_rob


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    scene = "pusht"
    resolution, model, normalizer, curr_latent_tensor, curr_action = load_task_config(
        scene=scene
    )
    curr_action = torch.from_numpy(curr_action).to(model.device).float()
    curr_action = normalizer["action"].normalize(curr_action)
    action_hist = [curr_action.clone()]
    FRAME_DT = 0.1  # 10 Hz render/control

    latest_delta = np.zeros_like(curr_action.cpu().numpy()[0])
    latest_t_client = 0.0
    reload_scene = False
    delta_lock = asyncio.Lock()

    async def recv_loop() -> None:
        nonlocal latest_delta
        nonlocal latest_t_client
        nonlocal scene
        nonlocal reload_scene
        try:
            while True:
                data = await ws.receive_json()
                msg_type = data.get("type", "action")
                async with delta_lock:
                    if msg_type in ("init", "set_task"):
                        scene = data.get("task")
                        reload_scene = True
                    elif msg_type == "action":
                        action = data.get("action", {})
                        latest_delta = parse_action(action, scene)
                        latest_t_client = data.get("t_client", 0.0)
        except Exception:
            return

    recv_task = asyncio.create_task(recv_loop())

    try:
        with torch.inference_mode():
            next_t = time.perf_counter()
            while True:
                if reload_scene:
                    (
                        resolution,
                        model,
                        normalizer,
                        curr_latent_tensor,
                        curr_action,
                    ) = load_task_config(scene=scene)
                    curr_action = torch.from_numpy(curr_action).to(model.device).float()
                    curr_action = normalizer["action"].normalize(curr_action)
                    action_hist = [curr_action.clone()]
                    reload_scene = False
                now = time.perf_counter()
                if now < next_t:
                    await asyncio.sleep(next_t - now)
                next_t += FRAME_DT

                async with delta_lock:
                    delta_action = latest_delta.copy()
                    t_client = latest_t_client

                if np.linalg.norm(delta_action) < 1e-8:
                    continue

                action_max = model.normalizer["action"].state_dict()[
                    "params_dict.input_stats.max"
                ]
                action_min = model.normalizer["action"].state_dict()[
                    "params_dict.input_stats.min"
                ]
                action_range = action_max - action_min
                if scene in ["bimanual_rope_cam_0", "bimanual_rope_cam_1"]:
                    action_range = torch.cat([action_range[:3], action_range[4:7]]).to(
                        model.device
                    )
                action_range_scale = action_range / action_range.max()
                action_range_scale = action_range_scale.detach().cpu().numpy()
                if scene in [
                    "pusht",
                    "pusht_cam_0",
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

                delta_action_rob = kybd_action_to_rob_action(delta_action, scene=scene)
                delta_action_tensor = torch.from_numpy(delta_action_rob).to(
                    model.device
                )

                curr_action = curr_action + delta_action_tensor
                curr_action = torch.clamp(curr_action, -1.0, 1.0)

                action_hist.append(curr_action.clone())
                action = torch.cat(action_hist, dim=0)[-(hist_context + 1) :].float()

                latent_pred = model.dynamics_forward(curr_latent_tensor, action[None])
                curr_latent_tensor = torch.cat(
                    [curr_latent_tensor, latent_pred], axis=1
                )
                curr_latent_tensor = curr_latent_tensor[:, -hist_context:]

                xs_pred = render_img_cm(
                    model,
                    curr_latent_tensor[:, -1],
                    resolution,
                    normalizer=normalizer,
                    num_views=len(model.obs_keys),
                )
                xs_pred_np = xs_pred.permute(0, 2, 3, 1).detach().cpu().numpy()[0]
                xs_pred_np = (xs_pred_np * 255).astype(np.uint8)
                xs_pred_np = np.clip(xs_pred_np, 0, 255)

                jpg_bytes = encode_frame_jpeg(xs_pred_np)
                prefix = struct.pack(">d", t_client)
                await ws.send_bytes(prefix + jpg_bytes)

    except Exception:
        pass
    finally:
        recv_task.cancel()
