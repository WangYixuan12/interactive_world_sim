from typing import Any

import cv2
import numpy as np
import torch
import transforms3d
from dm_control import mujoco
from gym_aloha.constants import (
    PUPPET_GRIPPER_JOINT_CLOSE,
    convert_puppet_from_joint_to_position,
)
from gym_aloha.env import AlohaEnv
from gym_aloha.tasks.sim import BOX_POSE
from gym_aloha.utils import sample_box_pose
from yixuan_utilities.draw_utils import center_crop
from yixuan_utilities.hdf5_utils import load_dict_from_hdf5
from yixuan_utilities.kinematics_helper import KinHelper

from interactive_world_sim.utils.pose_utils import (
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)

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


class SimAlohaPushTEnv(BaseEnv):
    """Base class for all environments."""

    def __init__(
        self,
        task: str = "transfer_cube",  # Changed from "pusht" to "transfer_cube"
        render_size: tuple[int, int] = (128, 128),
        delta_action: bool = False,
    ):
        self.env = AlohaEnv(task=task)
        self.kin_helper = KinHelper(robot_name="trossen_vx300s")
        self.render_size = render_size

        self.curr_vel = np.zeros(4)
        self.k_p, self.k_v = 100, 20  # Increased PD gains for faster response
        self.freq = 120  # Increased from 60 to 120 for much faster control
        self.dt = 1 / self.freq
        self.acc_lim = 2.0  # Increased acceleration limit
        self.vel_lim = 0.5  # Increased velocity limit
        self.delta_action = delta_action
        self.last_xy = np.zeros(4)
        self.iter_num = 0

    def step(self, action: np.ndarray) -> dict:
        """Run one timestep of the environment's dynamics."""
        assert action.shape == (4,), f"Invalid action shape: {action.shape}"
        init_state = self.env._env.physics.data.qpos[:].copy()  # noqa

        # compute EEF pose in each base frame
        obs = self.env._env.task.get_observation(self.env._env.physics)  # noqa

        world_t_robot_bases = np.zeros((2, 4, 4))

        # Check if left_base and right_base are available
        if "left_base" in obs and "right_base" in obs:
            world_t_left_base = pos_quat_to_mat(obs["left_base"])
            world_t_right_base = pos_quat_to_mat(obs["right_base"])
            world_t_robot_bases[0] = np.eye(4)
            world_t_robot_bases[1] = (
                np.linalg.inv(world_t_left_base) @ world_t_right_base
            )
        else:
            # Use identity matrices if base poses are not available
            world_t_robot_bases[0] = np.eye(4)
            world_t_robot_bases[1] = np.eye(4)
        solved_joints = np.zeros(12)
        for i in range(2):
            # compute current EEF pose
            init_qpos = init_state[i * 8 : i * 8 + 8]
            world_t_rob = world_t_robot_bases[i]
            # if self.iter_num % self.freq == 0:
            #     # reset reference traj every one second
            #     rob_t_curr_eef = self.kin_helper.compute_fk_from_link_idx(
            #         init_qpos, [self.kin_helper.sapien_eef_idx]
            #     )[0]
            #     world_t_curr_eef = world_t_rob @ rob_t_curr_eef
            #     curr_xy = world_t_curr_eef[:2, 3]
            #     self.last_xy[i * 2 : i * 2 + 2] = curr_xy

            # curr_xy = self.last_xy[i * 2 : i * 2 + 2]
            rob_t_curr_eef = self.kin_helper.compute_fk_from_link_idx(
                init_qpos, [self.kin_helper.sapien_eef_idx]
            )[0]
            world_t_curr_eef = world_t_rob @ rob_t_curr_eef
            curr_xy = world_t_curr_eef[:2, 3]

            # compute target EEF pose
            world_t_action_mat = np.eye(4)
            if self.delta_action:
                world_t_action_mat[:2, 3] = action[i * 2 : i * 2 + 2] + curr_xy
                self.last_xy[i * 2 : i * 2 + 2] = world_t_action_mat[:2, 3]
            else:
                world_t_action_mat[:2, 3] = action[i * 2 : i * 2 + 2]

            # solve PID control
            target_xy = world_t_action_mat[:2, 3]
            curr_vel = self.curr_vel[i * 2 : i * 2 + 2]
            acceleration = self.k_p * (target_xy - curr_xy) + self.k_v * (
                np.zeros(2) - curr_vel
            )
            # if self.delta_action:
            #     self.accu_error[i * 2 : i * 2 + 2] += (target_xy - curr_xy)
            #     acceleration += self.k_i * self.accu_error[i * 2 : i * 2 + 2]
            acceleration = np.clip(acceleration, -self.acc_lim, self.acc_lim)
            next_vel = acceleration * self.dt + curr_vel
            next_vel = np.clip(next_vel, -self.vel_lim, self.vel_lim)
            pid_action = curr_xy + (next_vel + curr_vel) * self.dt / 2.0
            self.curr_vel[i * 2 : i * 2 + 2] = next_vel
            pid_eef_pose = world_t_action_mat.copy()
            pid_eef_pose[:2, 3] = pid_action

            theta = np.pi * 5.0 / 12.0
            rob_t_pid_mat = np.linalg.inv(world_t_rob) @ pid_eef_pose
            # rob_t_pid_mat = np.linalg.inv(world_t_rob) @ world_t_action_mat
            rob_t_pid_mat[:3, :3] = np.array(
                [
                    [np.sin(theta), 0.0, np.cos(theta)],
                    [0.0, 1.0, 0.0],
                    [-np.cos(theta), 0.0, np.sin(theta)],
                ]
            )
            rob_t_pid_mat[2, 3] = 0.02

            # solve IK
            solved_joints[6 * i : 6 * i + 6] = self.kin_helper.compute_ik_from_mat(
                init_qpos, rob_t_pid_mat
            )[:6]
        self.iter_num += 1
        grippers = np.ones(1) * PUPPET_GRIPPER_JOINT_CLOSE
        env_action = np.concatenate(
            [solved_joints[:6], grippers, solved_joints[6:], grippers]
        )
        self.last_qpos = np.concatenate(
            [solved_joints[:6], np.zeros(2), solved_joints[6:], np.zeros(2)]
        )

        step_obs = self.env.step(env_action)

        # ### DEBUG ONLY
        # curr_state = self.env._env.physics.data.qpos[:].copy()
        # curr_pos_after_step = self.kin_helper.compute_fk_from_link_idx(
        #     init_state[8:16], [self.kin_helper.sapien_eef_idx]
        # )[0][:3, 3]
        # if not hasattr(self, "plotters"):
        #     self.plotters = [
        #         RealTimePlotter(
        #             title="PD control",
        #             window_size=300,
        #             num_lines=3,
        #             y_max=1,
        #             y_min=-1,
        #             legends=["target", "pid", "after_step"],
        #         )
        #         for i in range(3)
        #     ]
        #     self.iter = 0
        # for i in range(3):
        #     self.plotters[i].append(
        #         np.array([self.iter]),
        #         np.stack([target_xy, pid_action, curr_pos_after_step], axis=1)[
        #             i : i + 1
        #         ],
        #     )
        # self.iter += 1
        # ### END OF DEBUG ONLY
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
        # Set BOX_POSE before reset to avoid assertion error
        BOX_POSE[0] = sample_box_pose(seed=42)  # Use a fixed seed for reproducibility

        if state is not None:
            self.env._env.physics.data.qpos[:] = state  # noqa
            self.env._env.physics.forward()  # noqa
        else:
            self.env.reset()
        self.iter_num = 0
        self.last_xy = np.zeros(4)
        # self.curr_vel = np.zeros(4)

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

        world_t_bases = np.zeros((2, 4, 4))

        # Check if left_base and right_base are available
        if "left_base" in obs and "right_base" in obs:
            world_t_bases[0] = pos_quat_to_mat(obs["left_base"])
            world_t_bases[1] = pos_quat_to_mat(obs["right_base"])
        else:
            # Use identity matrices if base poses are not available
            world_t_bases[0] = np.eye(4)
            world_t_bases[1] = np.eye(4)

        curr_pos = np.zeros(4)
        for i in range(2):
            # compute current EEF pose
            curr_qpos = obs["qpos"][i * 7 : i * 7 + 6]
            curr_qpos = np.concatenate([curr_qpos, np.zeros(2)])
            rob_t_eef = self.kin_helper.compute_fk_from_link_idx(
                curr_qpos, [self.kin_helper.sapien_eef_idx]
            )[0]
            world_t_eef = world_t_bases[i] @ rob_t_eef
            left_t_eef = np.linalg.inv(world_t_bases[0]) @ world_t_eef

            curr_pos[i * 2 : i * 2 + 2] = left_t_eef[:2, 3]

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
        scale = max(shape[0] / height, shape[1] / width)
        fx = fx * scale
        fy = fy * scale
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
