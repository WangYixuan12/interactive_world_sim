from typing import Any

import cv2
import numpy as np
import torch
import transforms3d
from dm_control import mujoco
from gym_aloha.constants import (
    convert_puppet_from_joint_to_position,
)
from gym_aloha.env import AlohaEnv
from yixuan_utilities.draw_utils import center_crop
from yixuan_utilities.hdf5_utils import load_dict_from_hdf5
from yixuan_utilities.kinematics_helper import KinHelper

from interactive_world_sim.utils.pose_utils import (
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)
from interactive_world_sim.utils.real_time_plotter import RealTimePlotter

from .base_env import BaseEnv


def pos_quat_to_mat(pose_in_pos_quat: np.ndarray) -> np.ndarray:
    pos = pose_in_pos_quat[:3]
    quat = pose_in_pos_quat[3:]
    mat = np.eye(4)
    mat[:3, :3] = transforms3d.quaternions.quat2mat(quat)
    mat[:3, 3] = pos
    return mat


def mat_to_rot_6d(mat: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to 6D rotation representation."""
    assert mat.shape == (4, 4), f"Invalid matrix shape: {mat.shape}"
    rot_mat = mat[:3, :3]
    rot_6d = matrix_to_rotation_6d(torch.from_numpy(rot_mat).unsqueeze(0))
    pos = mat[:3, 3]
    return np.concatenate([pos, rot_6d.squeeze().numpy()])


def rot_6d_to_mat(rot_6d: np.ndarray) -> np.ndarray:
    """Convert 6D rotation representation to rotation matrix."""
    assert rot_6d.shape == (9,), f"Invalid rot_6d shape: {rot_6d.shape}"
    pos = rot_6d[:3]
    rot_6d = rot_6d[3:]
    rot_mat = rotation_6d_to_matrix(torch.from_numpy(rot_6d).unsqueeze(0))
    rot_mat = rot_mat.squeeze().numpy()
    mat = np.eye(4)
    mat[:3, :3] = rot_mat
    mat[:3, 3] = pos
    return mat


class SimAlohaSingleArmEnv(BaseEnv):
    """Base class for all environments."""

    def __init__(
        self, task: str = "transfer_cube", render_size: tuple[int, int] = (128, 128)
    ):
        self.env = AlohaEnv(task=task)
        self.kin_helper = KinHelper(robot_name="trossen_vx300s")
        self.render_size = render_size

        self.curr_vel = np.zeros(3)
        self.k_p, self.k_v = 50, 10  # PD control
        self.dt = 1 / 30.0
        self.acc_lim = 0.5

    def step(self, action: np.ndarray) -> tuple:
        """Run one timestep of the environment's dynamics."""
        init_state = self.env._env.physics.data.qpos[:].copy()  # noqa

        # compute EEF pose in each base frame
        left_base_t_right_action_mat = np.eye(4)
        left_base_t_right_action_mat[:3, 3] = action[:3]
        right_gripper = action[3:4]
        obs = self.env._env.task.get_observation(self.env._env.physics)  # noqa
        world_t_left_base = pos_quat_to_mat(obs["left_base"])
        world_t_right_base = pos_quat_to_mat(obs["right_base"])
        right_base_t_left_base = np.linalg.inv(world_t_right_base) @ world_t_left_base
        right_base_t_right_action_mat = (
            right_base_t_left_base @ left_base_t_right_action_mat
        )
        right_base_t_right_action_mat[:3, :3] = np.array(
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
        )
        # ### DEBUG ONLY
        # if hasattr(self, "plotters"):
        #     if self.iter // 10 == 0 or self.iter % 10 == 1:
        #         right_base_t_right_action_mat[:3, 3] += 0.5
        # ### END OF DEBUG ONLY
        curr_pos = self.kin_helper.compute_fk_from_link_idx(
            init_state[8:16], [self.kin_helper.sapien_eef_idx]
        )[0][:3, 3]
        target_pos = right_base_t_right_action_mat[:3, 3].copy()
        acceleration = self.k_p * (target_pos - curr_pos) + self.k_v * (
            np.zeros(3) - self.curr_vel
        )
        acceleration = np.clip(acceleration, -self.acc_lim, self.acc_lim)
        next_vel = acceleration * self.dt + self.curr_vel
        pid_action = curr_pos + (next_vel + self.curr_vel) * self.dt / 2.0
        self.curr_vel = next_vel
        right_base_t_right_action_mat[:3, 3] = pid_action

        # solve IK
        left_joints = obs["qpos"][:7]
        left_joints = np.concatenate([left_joints, left_joints[6:7]])
        left_gripper = obs["qpos"][6:7]
        right_init_qpos = obs["qpos"][7:14]
        right_init_qpos = np.concatenate([right_init_qpos, right_init_qpos[6:7]])
        right_joints = self.kin_helper.compute_ik_from_mat(
            right_init_qpos, right_base_t_right_action_mat
        )
        env_action = np.concatenate(
            [left_joints[:6], left_gripper, right_joints[:6], right_gripper]
        )

        step_obs = self.env.step(env_action)

        ### DEBUG ONLY
        curr_state = self.env._env.physics.data.qpos[:].copy()  # noqa
        curr_pos_after_step = self.kin_helper.compute_fk_from_link_idx(
            init_state[8:16], [self.kin_helper.sapien_eef_idx]
        )[0][:3, 3]
        if not hasattr(self, "plotters"):
            self.plotters = [
                RealTimePlotter(
                    title="PD control",
                    window_size=300,
                    num_lines=3,
                    y_max=1,
                    y_min=-1,
                    legends=["target", "pid", "after_step"],
                )
                for i in range(3)
            ]
            self.iter = 0
        for i in range(3):
            self.plotters[i].append(
                np.array([self.iter]),
                np.stack([target_pos, pid_action, curr_pos_after_step], axis=1)[
                    i : i + 1
                ],
            )
        self.iter += 1
        ### END OF DEBUG ONLY
        return step_obs

    def render(self, mode: str = "human") -> dict:
        """Render the environment."""
        obs = self.env._env.task.get_observation(self.env._env.physics)  # noqa
        if mode == "original":
            return obs["images"]
        elif mode == "human":
            keys = list(obs["images"].keys())
            img_obs = {}
            for key in keys:
                img = obs["images"][key]
                img = center_crop(img, self.render_size)
                img = cv2.resize(img, self.render_size, interpolation=cv2.INTER_AREA)
                img_obs[key] = img
            return img_obs
        else:
            raise ValueError(f"Unknown render mode: {mode}")

    def compute_init_state(self, hdf5_file_path: str, t: int = 0) -> np.ndarray:
        """Compute the initial state of the environment from replay buffer."""
        hdf5_data, _ = load_dict_from_hdf5(hdf5_file_path)
        env_state = hdf5_data["env_state"][t]
        joint_qpos = hdf5_data["obs"]["joint_pos"][t]
        qpos = np.concatenate([joint_qpos, env_state])

        left_arm = qpos[:6]
        right_arm = qpos[7:13]
        left_gripper_joint = qpos[6]
        left_gripper_pos = convert_puppet_from_joint_to_position(left_gripper_joint)
        left_gripper_qpos = np.array([left_gripper_pos, left_gripper_pos])
        right_gripper_joint = qpos[13]
        right_gripper_pos = convert_puppet_from_joint_to_position(right_gripper_joint)
        right_gripper_qpos = np.array([right_gripper_pos, right_gripper_pos])
        robot_qpos = np.concatenate(
            [
                left_arm,
                left_gripper_qpos,
                right_arm,
                right_gripper_qpos,
            ]
        )
        new_qpos = np.concatenate([robot_qpos, qpos[14:]])

        return new_qpos

    def reset(self, state: Any = None) -> None:
        """Reset the environment and return the initial observation."""
        if state is not None:
            self.env._env.physics.data.qpos[:] = state  # noqa
            self.env._env.physics.forward()  # noqa
        else:
            self.env.reset()

    def get_state(self) -> np.ndarray:
        """Return the current state of the environment."""
        return self.env._env.physics.data.qpos.copy()  # noqa

    def get_observations(self) -> dict:
        """Get the current observation of the environment."""
        return self.env._env.task.get_observation(self.env._env.physics)  # noqa

    def get_render_size(self) -> tuple[int, int]:
        """Return the render size of the environment."""
        return self.render_size

    def get_curr_pos(self) -> np.ndarray:
        """Return the current position of the environment."""
        obs = self.env._env.task.get_observation(self.env._env.physics)  # noqa

        world_t_left_base = pos_quat_to_mat(obs["left_base"])
        world_t_right_base = pos_quat_to_mat(obs["right_base"])
        left_base_t_right_base = np.linalg.inv(world_t_left_base) @ world_t_right_base

        # compute FK for obs
        right_qpos = obs["qpos"][7:]
        right_qpos = np.concatenate([right_qpos, right_qpos[6:7]])
        right_base_t_right_eef = self.kin_helper.compute_fk_from_link_idx(
            right_qpos, [self.kin_helper.sapien_eef_idx]
        )[0]
        left_base_t_right_eef = left_base_t_right_base @ right_base_t_right_eef
        right_pos = left_base_t_right_eef[:3, 3]
        right_gripper = right_qpos[6:7]
        curr_pos = np.concatenate([right_pos, right_gripper])
        return curr_pos

    def get_cam_intrinsic(self, name: str, shape: tuple[int, int]) -> np.ndarray:
        """Return the intrinsic matrix of the camera."""
        cam = mujoco.Camera(self.env._env.physics, camera_id=name)  # noqa
        cam.update()
        # c_xy = cam.matrices().image
        # cx = c_xy[0, 2]
        # cy = c_xy[1, 2]
        f_xy = cam.matrices().focal
        fx = -f_xy[0, 0]
        fy = f_xy[1, 1]
        cx = shape[1] / 2
        cy = shape[0] / 2
        width = cam.width
        height = cam.height
        fx = fx * shape[1] / width
        fy = fy * shape[0] / height
        return np.array([cx, cy, fx, fy])

    def get_cam_extrinsic(self, name: str) -> np.ndarray:
        """Return the extrinsic matrix of the camera."""
        cam = mujoco.Camera(self.env._env.physics, camera_id=name)  # noqa
        cam.update()
        rotation = cam.matrices().rotation
        translation = cam.matrices().translation
        translation[:3, 3] = -translation[:3, 3]
        translation[:3, :3] = rotation[:3, :3].T
        translation[:3, 1] = -translation[:3, 1]
        translation[:3, 2] = -translation[:3, 2]
        return translation
