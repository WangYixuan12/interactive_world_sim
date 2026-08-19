from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def demo_name(names: list[str], split: str, offset: int) -> str:
    ordered = sorted(names, key=lambda name: int(name.rsplit("_", 1)[1]))
    demos = ordered[:-5] if split == "train" else ordered[-5:]
    if not 0 <= offset < len(demos):
        raise ValueError(f"demo offset must be in [0, {len(demos) - 1}]")
    return demos[offset]


def held_out_demo_name(names: list[str], offset: int) -> str:
    return demo_name(names, "heldout", offset)


def _orient_like(frame: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, str, float]:
    variants = {
        "identity": frame,
        "flip_vertical": np.flip(frame, axis=0),
        "flip_horizontal": np.flip(frame, axis=1),
        "rotate_180": np.flip(frame, axis=(0, 1)),
    }
    name, best = min(
        variants.items(),
        key=lambda item: np.mean(
            (item[1].astype(np.float32) - reference.astype(np.float32)) ** 2
        ),
    )
    mse = float(np.mean((best.astype(np.float32) - reference.astype(np.float32)) ** 2))
    return np.ascontiguousarray(best), name, mse


def _apply_orientation(frames: list[np.ndarray], name: str) -> np.ndarray:
    array = np.stack(frames)
    if name == "flip_vertical":
        array = np.flip(array, axis=1)
    elif name == "flip_horizontal":
        array = np.flip(array, axis=2)
    elif name == "rotate_180":
        array = np.flip(array, axis=(1, 2))
    return np.ascontiguousarray(array)


def collect_environment(args: argparse.Namespace) -> None:
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    paths = sorted(Path(args.dataset_dir).glob("*.hdf5"))
    if len(paths) != 90:
        raise ValueError(f"expected 90 task files, found {len(paths)}")
    if not 1 <= len(args.task_indices) <= 2:
        raise ValueError("one or two task indices are required")

    output: dict[str, np.ndarray] = {}
    metadata: list[dict[str, object]] = []
    for example, task_index in enumerate(args.task_indices):
        path = paths[task_index]
        with h5py.File(path) as file:
            data = file["data"]
            selected_demo = demo_name(list(data), args.demo_split, args.demo_offset)
            demo = data[selected_demo]
            horizon = min(args.horizon, len(demo["actions"]))
            actions = np.asarray(demo["actions"][:horizon], dtype=np.float32)
            references = {
                "agent": np.asarray(demo["obs/agentview_rgb"][0]),
                "wrist": np.asarray(demo["obs/eye_in_hand_rgb"][0]),
            }
            initial_state = np.asarray(demo.attrs["init_state"])
            bddl_name = Path(str(data.attrs["bddl_file_name"])).name
            bddl_path = Path(get_libero_path("bddl_files")) / "libero_90" / bddl_name

        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl_path), camera_heights=128, camera_widths=128
        )
        agent_frames: list[np.ndarray] = []
        wrist_frames: list[np.ndarray] = []
        try:
            env.reset()
            env.set_init_state(initial_state)
            for action in actions:
                obs, _, _, _ = env.step(action)
                agent_frames.append(np.asarray(obs["agentview_image"], dtype=np.uint8))
                wrist_frames.append(
                    np.asarray(obs["robot0_eye_in_hand_image"], dtype=np.uint8)
                )
        finally:
            env.close()

        _, agent_orientation, agent_mse = _orient_like(agent_frames[0], references["agent"])
        _, wrist_orientation, wrist_mse = _orient_like(wrist_frames[0], references["wrist"])
        output[f"actions_{example}"] = actions
        output[f"env_agent_{example}"] = _apply_orientation(
            agent_frames, agent_orientation
        )
        output[f"env_wrist_{example}"] = _apply_orientation(
            wrist_frames, wrist_orientation
        )
        metadata.append(
            {
                "task_index": task_index,
                "task_file": path.name,
                "demo": selected_demo,
                "split": args.demo_split,
                "horizon": horizon,
                "agent_orientation": agent_orientation,
                "wrist_orientation": wrist_orientation,
                "initial_agent_mse_vs_recording": agent_mse,
                "initial_wrist_mse_vs_recording": wrist_mse,
            }
        )

    destination = Path(args.output_data)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, metadata=np.array(json.dumps(metadata)), **output)
    print(json.dumps({"environment_data": str(destination), "examples": metadata}, indent=2))


def _load_model(checkpoint: Path):
    import torch
    from omegaconf import OmegaConf

    from interactive_world_sim.algorithms.latent_dynamics.latent_world_model import (
        LatentWorldModel,
    )

    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", lambda expr: eval(expr, {"np": np}))
    if not OmegaConf.has_resolver("torch"):
        OmegaConf.register_new_resolver("torch", lambda name: getattr(torch, name))
    cfg = OmegaConf.load(checkpoint.parent.parent / ".hydra/config.yaml")
    cfg.algorithm.n_frames = 10
    cfg.algorithm.load_ae = None
    model = LatentWorldModel.load_from_checkpoint(
        checkpoint,
        cfg=cfg.algorithm,
        map_location="cuda:0",
        dtype=torch.float32,
        strict=False,
        weights_only=False,
    )
    model.eval()
    model.dynamics.eval()
    return model


def _predict(model, agent: np.ndarray, wrist: np.ndarray, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import torch
    from einops import rearrange

    from interactive_world_sim.algorithms.common.diffusion_helper import render_img_cm

    views = []
    for key, frame in zip(model.obs_keys, (agent[0], wrist[0]), strict=True):
        tensor = torch.from_numpy(frame).permute(2, 0, 1).float().div(255).cuda()
        views.append(model.normalizer[key].normalize(tensor))
    initial = torch.cat(views).unsqueeze(0)
    z_initial = model.encoder_forward(initial)
    action_tensor = torch.from_numpy(actions).float().unsqueeze(0).cuda()
    predicted = model.dynamics_forward(z_initial.unsqueeze(1), action_tensor)
    rendered = render_img_cm(
        model,
        rearrange(predicted, "b t c h w -> (b t) c h w"),
        agent.shape[1],
        model.normalizer,
        num_views=2,
    )
    rendered = (
        rendered.mul(255)
        .byte()
        .permute(0, 2, 3, 1)
        .cpu()
        .numpy()
    )
    return (
        np.concatenate([agent[:1], rendered[:, :, :, :3]]),
        np.concatenate([wrist[:1], rendered[:, :, :, 3:]]),
    )


def _panel_frame(examples: list[dict[str, object]], step: int) -> np.ndarray:
    import cv2

    headings = ("Environment agent", "World model agent", "Environment wrist", "World model wrist")
    rows = []
    for example in examples:
        tiles = [
            example["env_agent"][step],
            example["wm_agent"][step],
            example["env_wrist"][step],
            example["wm_wrist"][step],
        ]
        row = np.concatenate(tiles, axis=1)
        row = cv2.copyMakeBorder(row, 26, 0, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
        for column, heading in enumerate(headings):
            cv2.putText(row, heading, (column * 128 + 4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(row, str(example["label"]), (4, 124), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 255, 0), 1, cv2.LINE_AA)
        rows.append(row)
    panel = np.concatenate(rows, axis=0)
    cv2.putText(panel, f"step {step}", (430, 124), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def predict_and_render(args: argparse.Namespace) -> None:
    import cv2
    import imageio.v2 as imageio
    import torch

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = _load_model(Path(args.checkpoint))
    source = np.load(args.input_data, allow_pickle=False)
    metadata = json.loads(str(source["metadata"]))
    examples: list[dict[str, object]] = []
    for index, item in enumerate(metadata):
        env_agent = source[f"env_agent_{index}"]
        env_wrist = source[f"env_wrist_{index}"]
        wm_agent, wm_wrist = _predict(
            model, env_agent, env_wrist, source[f"actions_{index}"]
        )
        examples.append(
            {
                "label": f"{item['split']} / task {item['task_index']} / {item['demo']}",
                "env_agent": env_agent,
                "env_wrist": env_wrist,
                "wm_agent": wm_agent,
                "wm_wrist": wm_wrist,
            }
        )

    frames = [_panel_frame(examples, step) for step in range(min(len(x["env_agent"]) for x in examples))]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video = output_dir / "comparison.mp4"
    imageio.mimsave(video, frames, fps=args.fps, macro_block_size=2)
    preview = output_dir / "comparison_first_last.png"
    cv2.imwrite(str(preview), cv2.cvtColor(np.concatenate([frames[0], frames[-1]], axis=1), cv2.COLOR_RGB2BGR))
    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "action_source": f"{metadata[0]['split']} expert actions",
        "world_model_observation_source": "environment frame 0 only",
        "examples": metadata,
        "video": str(video),
        "preview": str(preview),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="phase", required=True)
    environment = subparsers.add_parser("environment")
    environment.add_argument("--dataset-dir", required=True)
    environment.add_argument("--output-data", required=True)
    environment.add_argument("--task-indices", nargs="+", type=int, default=[0, 45])
    environment.add_argument("--demo-split", choices=("train", "heldout"), default="heldout")
    environment.add_argument("--demo-offset", type=int, default=0)
    environment.add_argument("--horizon", type=int, default=200)
    model = subparsers.add_parser("model")
    model.add_argument("--input-data", required=True)
    model.add_argument("--checkpoint", required=True)
    model.add_argument("--output-dir", required=True)
    model.add_argument("--fps", type=int, default=20)
    model.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.phase == "environment":
        collect_environment(arguments)
    else:
        predict_and_render(arguments)
