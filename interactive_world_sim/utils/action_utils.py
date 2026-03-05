import numpy as np
from yixuan_utilities.kinematics_helper import KinHelper

from .aloha_conts import (
    MASTER_GRIPPER_JOINT_NORMALIZE_FN,
    PUPPET_GRIPPER_JOINT_CLOSE,
    PUPPET_GRIPPER_JOINT_OPEN,
    PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN,
)


def action_primitive_to_joint_pos(
    action_primitive: np.ndarray,
    ctrl_mode: str,
    base_pose_in_world: np.ndarray,
    kin_helper: KinHelper,
    last_joint_pos: np.ndarray,
) -> np.ndarray:
    """Convert action to joint command."""
    assert last_joint_pos.shape[0] % 7 == 0
    if ctrl_mode == "joint":
        joint_pos = action_primitive
    elif ctrl_mode == "bimanual_push":
        joint_pos = np.zeros(14)
        for i in range(2):
            # compute robot_t_eef
            action_xy = action_primitive[i * 2 : i * 2 + 2]
            world_t_eef = np.eye(4)
            world_t_eef[:2, 3] = action_xy
            world_t_eef[2, 3] = 0.10
            world_t_robot: np.ndarray = base_pose_in_world[i]
            robot_t_eef = np.linalg.inv(world_t_robot) @ world_t_eef
            theta = np.pi * 5.0 / 12.0
            robot_t_eef[:3, :3] = np.array(
                [
                    [np.sin(theta), 0.0, np.cos(theta)],
                    [0.0, 1.0, 0.0],
                    [-np.cos(theta), 0.0, np.sin(theta)],
                ]
            )
            robot_t_eef[0, 3] = np.clip(robot_t_eef[0, 3], 0.25, 1.0)
            robot_t_eef[1, 3] = np.clip(robot_t_eef[1, 3], -0.25, 0.25)

            last_qpos = last_joint_pos[i * 7 : i * 7 + 6]
            init_qpos = np.concatenate([last_qpos, np.zeros(2)])
            ik_joint = kin_helper.compute_ik_from_mat(init_qpos, robot_t_eef)
            joint_pos[7 * i : 7 * i + 6] = ik_joint[:6]
            joint_pos[7 * i + 6] = PUPPET_GRIPPER_JOINT_CLOSE

    elif ctrl_mode == "single_push":
        joint_pos = np.zeros(7)
        # compute robot_t_eef
        action_xy = action_primitive[:2]
        world_t_eef = np.eye(4)
        world_t_eef[:2, 3] = action_xy
        world_t_eef[2, 3] = 0.10  #  height
        world_t_robot = base_pose_in_world[0]
        robot_t_eef = np.linalg.inv(world_t_robot) @ world_t_eef
        theta = np.pi * 5.0 / 12.0
        robot_t_eef[:3, :3] = np.array(
            [
                [np.sin(theta), 0.0, np.cos(theta)],
                [0.0, 1.0, 0.0],
                [-np.cos(theta), 0.0, np.sin(theta)],
            ]
        )
        robot_t_eef[0, 3] = np.clip(robot_t_eef[0, 3], 0.25, 1.0)
        robot_t_eef[1, 3] = np.clip(robot_t_eef[1, 3], -0.25, 0.25)

        last_qpos = last_joint_pos[:6]
        init_qpos = np.concatenate([last_qpos, np.zeros(2)])
        ik_joint = kin_helper.compute_ik_from_mat(init_qpos, robot_t_eef)
        joint_pos[:6] = ik_joint[:6]
        joint_pos[6] = PUPPET_GRIPPER_JOINT_CLOSE
    elif ctrl_mode == "single_sweep":
        joint_pos = np.zeros(7)
        # compute robot_t_eef
        action_xy = action_primitive[:2]
        world_t_eef = np.eye(4)
        world_t_eef[:2, 3] = action_xy

        # Fixed height for sweep mode
        world_t_eef[2, 3] = 0.25  # Static height
        world_t_robot = base_pose_in_world[0]
        robot_t_eef = np.linalg.inv(world_t_robot) @ world_t_eef

        # Get wrist rotation from action[2] if available
        if len(action_primitive) > 2:
            # Use third component for wrist orientation
            wrist_angle = action_primitive[2] * np.pi  # Map from [0,1] to [0,π]

            # Base orientation with wrist rotation
            robot_t_eef[:3, :3] = np.array(
                [
                    [np.cos(wrist_angle), -np.sin(wrist_angle), 0.0],
                    [np.sin(wrist_angle), np.cos(wrist_angle), 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ) @ np.array(
                [
                    [0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0],
                    [-1.0, 0.0, 0.0],
                ]
            )
        else:
            # Use default orientation - end effector pointing down
            robot_t_eef[:3, :3] = np.array(
                [
                    [0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0],
                    [-1.0, 0.0, 0.0],
                ]
            )

        last_qpos = last_joint_pos[:6]
        init_qpos = np.concatenate([last_qpos, np.zeros(2)])
        ik_joint = kin_helper.compute_ik_from_mat(init_qpos, robot_t_eef)
        joint_pos[:6] = ik_joint[:6]

        # Check if action includes a gripper position value (len > 2)
        gripper_value = action_primitive[3]
        wider_range_value = (
            PUPPET_GRIPPER_JOINT_CLOSE
            + (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE)
            * 1.5
            * gripper_value
        )
        joint_pos[6] = wider_range_value

    elif ctrl_mode == "single_wipe":
        joint_pos = np.zeros(7)
        # compute robot_t_eef
        action_xy = action_primitive[:2]
        world_t_eef = np.eye(4)
        world_t_eef[:2, 3] = action_xy

        # Check if action has a component for height (3rd component)
        if len(action_primitive) > 2:
            # Scale the third component to a reasonable height range
            # action[2] is expected to be in [0,1], map to [0.05, 0.25]
            height = 0.05 + action_primitive[2] * 0.20
            world_t_eef[2, 3] = height  # Use master's height
        else:
            world_t_eef[2, 3] = 0.10  # Default height

        world_t_robot = base_pose_in_world[0]
        robot_t_eef = np.linalg.inv(world_t_robot) @ world_t_eef
        theta = np.pi * 5.0 / 12.0
        robot_t_eef[:3, :3] = np.array(
            [
                [np.sin(theta), 0.0, np.cos(theta)],
                [0.0, 1.0, 0.0],
                [-np.cos(theta), 0.0, np.sin(theta)],
            ]
        )

        robot_t_eef[0, 3] = np.clip(robot_t_eef[0, 3], 0.25, 1.0)
        robot_t_eef[1, 3] = np.clip(robot_t_eef[1, 3], -0.25, 0.25)

        last_qpos = last_joint_pos[:6]
        init_qpos = np.concatenate([last_qpos, np.zeros(2)])
        ik_joint = kin_helper.compute_ik_from_mat(init_qpos, robot_t_eef)
        joint_pos[:6] = ik_joint[:6]

        # For single_wipe, use the 4th component for gripper (if available)
        gripper_value = action_primitive[3]
        wider_range_value = (
            PUPPET_GRIPPER_JOINT_CLOSE
            + (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE)
            * 1.5
            * gripper_value
        )
        joint_pos[6] = wider_range_value

    elif ctrl_mode == "single_rope":
        # action: (x, y, z, theta, gripper)
        joint_pos = np.zeros(7)
        # compute robot_t_eef
        action_xy = action_primitive[:2]
        world_t_eef = np.eye(4)
        world_t_eef[:3, 3] = action_primitive[:3]  # x, y, z
        theta = action_primitive[3]  # wrist rotation
        world_t_robot = base_pose_in_world[0]
        robot_t_eef = np.linalg.inv(world_t_robot) @ world_t_eef
        robot_t_eef[:3, :3] = np.array(
            [
                [0.0, -np.sin(theta), np.cos(theta)],
                [0.0, np.cos(theta), np.sin(theta)],
                [-1.0, 0.0, 0.0],
            ]
        )
        last_qpos = last_joint_pos[:6]
        init_qpos = np.concatenate([last_qpos, np.zeros(2)])
        ik_joint = kin_helper.compute_ik_from_mat(init_qpos, robot_t_eef)
        joint_pos[:6] = ik_joint[:6]
        joint_pos[6] = action_primitive[4]  # gripper
    elif ctrl_mode == "bimanual_sweep":
        joint_pos = np.zeros(14)
        for i in range(2):
            # compute robot_t_eef
            action_xy = action_primitive[i * 2 : i * 2 + 2]
            world_t_eef = np.eye(4)
            world_t_eef[:2, 3] = action_xy
            if i == 0:
                world_t_eef[2, 3] = 0.30
            elif i == 1:
                world_t_eef[2, 3] = 0.12
            world_t_robot = base_pose_in_world[i]
            robot_t_eef = np.linalg.inv(world_t_robot) @ world_t_eef
            if i == 0:
                theta = np.pi / 8.0
                robot_t_eef[:3, :3] = np.array(
                    [
                        [np.sin(theta), np.cos(theta), 0.0],
                        [0.0, 0.0, -1.0],
                        [-np.cos(theta), np.sin(theta), 0.0],
                    ]
                )
            elif i == 1:
                theta = np.pi / 2.0
                robot_t_eef[:3, :3] = np.array(
                    [
                        [np.sin(theta), 0.0, np.cos(theta)],
                        [0.0, 1.0, 0.0],
                        [-np.cos(theta), 0.0, np.sin(theta)],
                    ]
                )
            robot_t_eef[0, 3] = np.clip(robot_t_eef[0, 3], 0.25, 1.0)
            robot_t_eef[1, 3] = np.clip(robot_t_eef[1, 3], -0.25, 0.25)

            last_qpos = last_joint_pos[i * 7 : i * 7 + 6]
            init_qpos = np.concatenate([last_qpos, np.zeros(2)])
            ik_joint = kin_helper.compute_ik_from_mat(init_qpos, robot_t_eef)
            joint_pos[7 * i : 7 * i + 6] = ik_joint[:6]
            joint_pos[7 * i + 6] = PUPPET_GRIPPER_JOINT_CLOSE
    elif ctrl_mode == "bimanual_sweep_v2":
        joint_pos = np.zeros(14)
        for i in range(2):
            # compute robot_t_eef
            action_xy = action_primitive[i * 2 : i * 2 + 2]
            world_t_eef = np.eye(4)
            world_t_eef[:2, 3] = action_xy
            if i == 0:
                world_t_eef[2, 3] = 0.30
            elif i == 1:
                world_t_eef[2, 3] = 0.10
            world_t_robot = base_pose_in_world[i]
            robot_t_eef = np.linalg.inv(world_t_robot) @ world_t_eef
            if i == 0:
                theta = np.pi / 8.0
                robot_t_eef[:3, :3] = np.array(
                    [
                        [np.sin(theta), np.cos(theta), 0.0],
                        [0.0, 0.0, -1.0],
                        [-np.cos(theta), np.sin(theta), 0.0],
                    ]
                )
                robot_t_eef[0, 3] = np.clip(robot_t_eef[0, 3], 0.25, 0.6)
                robot_t_eef[1, 3] = np.clip(robot_t_eef[1, 3], -0.2, 0.2)
            elif i == 1:
                theta = np.pi / 2.0 + np.pi / 36.0
                robot_t_eef[:3, :3] = np.array(
                    [
                        [np.sin(theta), 0.0, np.cos(theta)],
                        [0.0, 1.0, 0.0],
                        [-np.cos(theta), 0.0, np.sin(theta)],
                    ]
                )
                robot_t_eef[0, 3] = np.clip(robot_t_eef[0, 3], 0.4, 0.5)
                robot_t_eef[1, 3] = np.clip(robot_t_eef[1, 3], -0.2, 0.2)

            last_qpos = last_joint_pos[i * 7 : i * 7 + 6]
            init_qpos = np.concatenate([last_qpos, np.zeros(2)])
            ik_joint = kin_helper.compute_ik_from_mat(init_qpos, robot_t_eef)
            joint_pos[7 * i : 7 * i + 6] = ik_joint[:6]
            joint_pos[7 * i + 6] = PUPPET_GRIPPER_JOINT_CLOSE
    elif ctrl_mode == "single_grasp":
        # action: (x, y, z, gripper)
        joint_pos = np.zeros(7)
        # compute robot_t_eef
        action_xy = action_primitive[:2]
        world_t_eef = np.eye(4)
        world_t_eef[:3, 3] = action_primitive[:3]  # x, y, z
        world_t_robot = base_pose_in_world[0]
        robot_t_eef = np.linalg.inv(world_t_robot) @ world_t_eef
        theta = np.pi * 11.0 / 24.0
        robot_t_eef[:3, :3] = np.array(
            [
                [np.sin(theta), 0.0, np.cos(theta)],
                [0.0, 1.0, 0.0],
                [-np.cos(theta), 0.0, np.sin(theta)],
            ]
        )
        last_qpos = last_joint_pos[:6]
        init_qpos = np.concatenate([last_qpos, np.zeros(2)])
        ik_joint = kin_helper.compute_ik_from_mat(init_qpos, robot_t_eef)
        joint_pos[:6] = ik_joint[:6]
        joint_pos[6] = action_primitive[3]  # gripper
    elif ctrl_mode == "bimanual_pack":
        # action: (left_x, left_y, left_z, right_x, right_y, right_z)
        joint_pos = np.zeros(14)
        for i in range(2):
            action_xyz = action_primitive[i * 3 : (i + 1) * 3]
            world_t_eef = np.eye(4)
            world_t_eef[:3, 3] = action_xyz
            world_t_robot = base_pose_in_world[i]
            robot_t_eef = np.linalg.inv(world_t_robot) @ world_t_eef
            theta = np.pi * 11.0 / 24.0
            robot_t_eef[:3, :3] = np.array(
                [
                    [np.sin(theta), 0.0, np.cos(theta)],
                    [0.0, 1.0, 0.0],
                    [-np.cos(theta), 0.0, np.sin(theta)],
                ]
            )
            last_qpos = last_joint_pos[i * 7 : i * 7 + 6]
            init_qpos = np.concatenate([last_qpos, np.zeros(2)])
            ik_joint = kin_helper.compute_ik_from_mat(init_qpos, robot_t_eef)
            joint_pos[7 * i : 7 * i + 6] = ik_joint[:6]
            if i == 0:
                joint_pos[7 * i + 6] = PUPPET_GRIPPER_JOINT_OPEN
            elif i == 1:
                joint_pos[7 * i + 6] = PUPPET_GRIPPER_JOINT_CLOSE
    elif ctrl_mode == "bimanual_rope":
        # action: (right_x, right_y, right_z, right_gripper,
        # left_x, left_y, left_z, left_gripper)
        joint_pos = np.zeros(14)
        for i in range(2):
            action_xyz = action_primitive[i * 4 : i * 4 + 3]
            world_t_eef = np.eye(4)
            world_t_eef[:3, 3] = action_xyz
            world_t_robot = base_pose_in_world[i]
            robot_t_eef = np.linalg.inv(world_t_robot) @ world_t_eef
            theta = np.pi * 5.0 / 12.0
            robot_t_eef[:3, :3] = np.array(
                [
                    [np.sin(theta), 0.0, np.cos(theta)],
                    [0.0, 1.0, 0.0],
                    [-np.cos(theta), 0.0, np.sin(theta)],
                ]
            )
            last_qpos = last_joint_pos[i * 7 : i * 7 + 6]
            init_qpos = np.concatenate([last_qpos, np.zeros(2)])
            ik_joint = kin_helper.compute_ik_from_mat(init_qpos, robot_t_eef)
            joint_pos[7 * i : 7 * i + 6] = ik_joint[:6]
            joint_pos[7 * i + 6] = action_primitive[i * 4 + 3]  # gripper
    elif ctrl_mode == "single_chain_in_box":
        # action: (right_x, right_y, right_z, right_gripper,
        # left_x, left_y, left_z, left_gripper)
        joint_pos = np.zeros(7)
        for i in range(1):
            action_xyz = action_primitive[i * 4 : i * 4 + 3]
            world_t_eef = np.eye(4)
            world_t_eef[:3, 3] = action_xyz
            world_t_robot = base_pose_in_world[i]
            robot_t_eef = np.linalg.inv(world_t_robot) @ world_t_eef
            theta = np.pi * 5.0 / 12.0
            robot_t_eef[:3, :3] = np.array(
                [
                    [np.sin(theta), 0.0, np.cos(theta)],
                    [0.0, 1.0, 0.0],
                    [-np.cos(theta), 0.0, np.sin(theta)],
                ]
            )
            last_qpos = last_joint_pos[i * 7 : i * 7 + 6]
            init_qpos = np.concatenate([last_qpos, np.zeros(2)])
            ik_joint = kin_helper.compute_ik_from_mat(init_qpos, robot_t_eef)
            joint_pos[7 * i : 7 * i + 6] = ik_joint[:6]
            joint_pos[7 * i + 6] = action_primitive[i * 4 + 3]  # gripper
    elif ctrl_mode == "bimanual_box":
        # action: (right_pos_3d, right_rot_6d, right_gripper,
        # left_pos_3d, left_rot_6d, left_gripper)
        joint_pos = action_primitive
    else:
        raise NotImplementedError(f"Unknown control mode: {ctrl_mode}")
    return joint_pos


def joint_pos_to_action_primitive(
    joint_pos: np.ndarray,
    ctrl_mode: str,
    base_pose_in_world: np.ndarray,
    kin_helper: KinHelper,
) -> np.ndarray:
    """Convert joint positions to action primitives based on control mode."""
    assert joint_pos.shape[0] % 7 == 0
    num_robots = joint_pos.shape[0] // 7
    if ctrl_mode == "joint":
        actions = joint_pos[None]
    elif ctrl_mode == "bimanual_push":
        action = np.zeros(4)
        for rob_i in range(num_robots):
            fk_joint_pos: np.ndarray = joint_pos[rob_i * 7 : (rob_i + 1) * 7]
            fk_joint_pos = np.concatenate([fk_joint_pos, fk_joint_pos[-1:]])
            rob_t_eef = kin_helper.compute_fk_from_link_idx(
                fk_joint_pos, [kin_helper.sapien_eef_idx]
            )[0]
            rob_t_eef[0, 3] = np.clip(rob_t_eef[0, 3], 0.25, 1.0)
            rob_t_eef[1, 3] = np.clip(rob_t_eef[1, 3], -0.25, 0.25)
            world_t_robot = base_pose_in_world[rob_i]
            world_t_eef = world_t_robot @ rob_t_eef
            action[rob_i * 2 : (rob_i + 1) * 2] = world_t_eef[:2, 3]
        actions = action[None]
    elif ctrl_mode == "single_push":
        action = np.zeros(2)
        rob_i = 0
        fk_joint_pos = joint_pos[rob_i * 7 : (rob_i + 1) * 7]
        fk_joint_pos = np.concatenate([fk_joint_pos, fk_joint_pos[-1:]])
        rob_t_eef = kin_helper.compute_fk_from_link_idx(
            fk_joint_pos, [kin_helper.sapien_eef_idx]
        )[0]
        world_t_robot = base_pose_in_world[rob_i]
        world_t_eef = world_t_robot @ rob_t_eef
        action = world_t_eef[:2, 3]
        actions = action[None]
    elif ctrl_mode == "single_sweep":
        action = np.zeros(4)  # 4D array for XY, height, and gripper
        rob_i = 0
        fk_joint_pos = joint_pos[rob_i * 7 : (rob_i + 1) * 7]
        fk_joint_pos = np.concatenate([fk_joint_pos, fk_joint_pos[-1:]])
        rob_t_eef = kin_helper.compute_fk_from_link_idx(
            fk_joint_pos, [kin_helper.sapien_eef_idx]
        )[0]
        world_t_robot = base_pose_in_world[rob_i]
        world_t_eef = world_t_robot @ rob_t_eef
        action[:2] = world_t_eef[:2, 3]  # XY coordinates

        # Extract height from the world_t_eef and normalize to [0,1]
        height = world_t_eef[2, 3]
        # Map from physical height range [0.05, 0.25] to [0,1]
        normalized_height = (height - 0.05) / 0.20
        normalized_height = np.clip(normalized_height, 0, 1)
        action[2] = normalized_height  # Height

        action[3] = joint_pos[rob_i * 7 + 6]  # Gripper position
        actions = action[None]
    elif ctrl_mode == "single_wipe":
        action = np.zeros(4)  # 4D array for XY, height, and gripper
        rob_i = 0
        fk_joint_pos = joint_pos[rob_i * 7 : (rob_i + 1) * 7]
        fk_joint_pos = np.concatenate([fk_joint_pos, fk_joint_pos[-1:]])
        rob_t_eef = kin_helper.compute_fk_from_link_idx(
            fk_joint_pos, [kin_helper.sapien_eef_idx]
        )[0]
        world_t_robot = base_pose_in_world[rob_i]
        world_t_eef = world_t_robot @ rob_t_eef
        action[:2] = world_t_eef[:2, 3]  # XY coordinates

        # Extract height from the world_t_eef and normalize to [0,1]
        height = world_t_eef[2, 3]
        # Map from physical height range [0.05, 0.25] to [0,1]
        normalized_height = (height - 0.05) / 0.20
        normalized_height = np.clip(normalized_height, 0, 1)
        action[2] = normalized_height  # Height

        action[3] = joint_pos[rob_i * 7 + 6]  # Gripper position
        actions = action[None]
    elif ctrl_mode == "single_rope":
        action = np.zeros(5)  # (x, y, z, theta, gripper)
        rob_i = 0
        fk_joint_pos = joint_pos[rob_i * 7 : (rob_i + 1) * 7]
        fk_joint_pos = np.concatenate([fk_joint_pos, fk_joint_pos[-1:]])
        rob_t_eef = kin_helper.compute_fk_from_link_idx(
            fk_joint_pos, [kin_helper.sapien_eef_idx]
        )[0]
        rob_t_eef[0, 3] = np.clip(rob_t_eef[0, 3], 0.25, 1.0)
        rob_t_eef[1, 3] = np.clip(rob_t_eef[1, 3], -0.25, 0.25)
        world_t_robot = base_pose_in_world[rob_i]
        world_t_eef = world_t_robot @ rob_t_eef
        action[:3] = world_t_eef[:3, 3]  # XYZ
        action[2] = np.clip(action[2], 0.18, 0.25)

        z_dir = rob_t_eef[:3, 2]
        theta = np.arctan2(z_dir[1], z_dir[0])
        theta = np.clip(theta, -np.pi / 2.0, np.pi / 2.0)
        action[3] = theta  # yaw
        action[4] = PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(
            MASTER_GRIPPER_JOINT_NORMALIZE_FN(joint_pos[rob_i * 7 + 6])
        )  # Gripper position
        actions = action[None]
    elif ctrl_mode == "bimanual_sweep":
        action = np.zeros(4)  # (right x, right y, left x, left y)
        for rob_i in range(num_robots):
            fk_joint_pos = joint_pos[rob_i * 7 : (rob_i + 1) * 7]
            fk_joint_pos = np.concatenate([fk_joint_pos, fk_joint_pos[-1:]])
            rob_t_eef = kin_helper.compute_fk_from_link_idx(
                fk_joint_pos, [kin_helper.sapien_eef_idx]
            )[0]
            if rob_i == 0:
                rob_t_eef[0, 3] = np.clip(rob_t_eef[0, 3], 0.25, 0.75)
                rob_t_eef[1, 3] = np.clip(rob_t_eef[1, 3], -0.25, 0.25)
            elif rob_i == 1:
                rob_t_eef[0, 3] = np.clip(rob_t_eef[0, 3], 0.25, 0.50)
                rob_t_eef[1, 3] = np.clip(rob_t_eef[1, 3], -0.25, 0.25)
            world_t_robot = base_pose_in_world[rob_i]
            world_t_eef = world_t_robot @ rob_t_eef
            action[rob_i * 2 : (rob_i + 1) * 2] = world_t_eef[:2, 3]
        actions = action[None]
    elif ctrl_mode == "bimanual_sweep_v2":
        action = np.zeros(4)  # (right x, right y, left x, left y)
        for rob_i in range(num_robots):
            fk_joint_pos = joint_pos[rob_i * 7 : (rob_i + 1) * 7]
            fk_joint_pos = np.concatenate([fk_joint_pos, fk_joint_pos[-1:]])
            rob_t_eef = kin_helper.compute_fk_from_link_idx(
                fk_joint_pos, [kin_helper.sapien_eef_idx]
            )[0]
            if rob_i == 0:
                rob_t_eef[0, 3] = np.clip(rob_t_eef[0, 3], 0.25, 0.6)
                rob_t_eef[1, 3] = np.clip(rob_t_eef[1, 3], -0.2, 0.2)
            elif rob_i == 1:
                rob_t_eef[0, 3] = np.clip(rob_t_eef[0, 3], 0.4, 0.5)
                rob_t_eef[1, 3] = np.clip(rob_t_eef[1, 3], -0.2, 0.2)
            world_t_robot = base_pose_in_world[rob_i]
            world_t_eef = world_t_robot @ rob_t_eef
            action[rob_i * 2 : (rob_i + 1) * 2] = world_t_eef[:2, 3]
        actions = action[None]
    elif ctrl_mode == "single_grasp":
        action = np.zeros(4)  # (x, y, z, gripper)
        rob_i = 0
        fk_joint_pos = joint_pos[rob_i * 7 : (rob_i + 1) * 7]
        fk_joint_pos = np.concatenate([fk_joint_pos, fk_joint_pos[-1:]])
        rob_t_eef = kin_helper.compute_fk_from_link_idx(
            fk_joint_pos, [kin_helper.sapien_eef_idx]
        )[0]
        rob_t_eef[0, 3] = np.clip(rob_t_eef[0, 3], 0.25, 0.75)
        rob_t_eef[1, 3] = np.clip(rob_t_eef[1, 3], -0.25, 0.25)
        world_t_robot = base_pose_in_world[rob_i]
        world_t_eef = world_t_robot @ rob_t_eef
        action[:3] = world_t_eef[:3, 3]
        action[2] = np.clip(action[2], 0.0, 0.2)
        action[3] = PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(
            MASTER_GRIPPER_JOINT_NORMALIZE_FN(joint_pos[rob_i * 7 + 6])
        )  # Gripper position
        actions = action[None]
    elif ctrl_mode == "bimanual_pack":
        action = np.zeros(6)  # (right x, right y, right z, left x, left y, left z)
        for rob_i in range(num_robots):
            fk_joint_pos = joint_pos[rob_i * 7 : (rob_i + 1) * 7]
            fk_joint_pos = np.concatenate([fk_joint_pos, fk_joint_pos[-1:]])
            rob_t_eef = kin_helper.compute_fk_from_link_idx(
                fk_joint_pos, [kin_helper.sapien_eef_idx]
            )[0]
            rob_t_eef[0, 3] = np.clip(rob_t_eef[0, 3], 0.25, 0.75)
            rob_t_eef[1, 3] = np.clip(rob_t_eef[1, 3], -0.25, 0.25)
            world_t_robot = base_pose_in_world[rob_i]
            world_t_eef = world_t_robot @ rob_t_eef
            action[rob_i * 3 : (rob_i + 1) * 3] = world_t_eef[:3, 3]
            action[rob_i * 3 + 2] = np.clip(action[rob_i * 3 + 2], 0.1, 0.3)
        actions = action[None]
    elif ctrl_mode == "bimanual_rope":
        action = np.zeros(8)
        # (right x, right y, right z, right_gripper,
        # left x, left y, left z, left_gripper)
        for rob_i in range(num_robots):
            fk_joint_pos = joint_pos[rob_i * 7 : (rob_i + 1) * 7]
            fk_joint_pos = np.concatenate([fk_joint_pos, fk_joint_pos[-1:]])
            rob_t_eef = kin_helper.compute_fk_from_link_idx(
                fk_joint_pos, [kin_helper.sapien_eef_idx]
            )[0]
            rob_t_eef[0, 3] = np.clip(rob_t_eef[0, 3], 0.2, 0.4)
            rob_t_eef[1, 3] = np.clip(rob_t_eef[1, 3], -0.15, 0.15)
            world_t_robot = base_pose_in_world[rob_i]
            world_t_eef = world_t_robot @ rob_t_eef

            action[rob_i * 4 : rob_i * 4 + 3] = world_t_eef[:3, 3]
            action[rob_i * 4 + 2] = np.clip(action[rob_i * 4 + 2], 0.08, 0.16)
            action[rob_i * 4 + 3] = PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(
                MASTER_GRIPPER_JOINT_NORMALIZE_FN(joint_pos[rob_i * 7 + 6])
            )  # Gripper position
        actions = action[None]
    elif ctrl_mode == "single_chain_in_box":
        action = np.zeros(4)
        for rob_i in range(1):
            fk_joint_pos = joint_pos[rob_i * 7 : (rob_i + 1) * 7]
            fk_joint_pos = np.concatenate([fk_joint_pos, fk_joint_pos[-1:]])
            rob_t_eef = kin_helper.compute_fk_from_link_idx(
                fk_joint_pos, [kin_helper.sapien_eef_idx]
            )[0]
            rob_t_eef[0, 3] = np.clip(rob_t_eef[0, 3], 0.2, 0.6)
            rob_t_eef[1, 3] = np.clip(rob_t_eef[1, 3], -0.2, 0.2)
            world_t_robot = base_pose_in_world[rob_i]
            world_t_eef = world_t_robot @ rob_t_eef

            action[rob_i * 4 : rob_i * 4 + 3] = world_t_eef[:3, 3]
            action[rob_i * 4 + 2] = np.clip(action[rob_i * 4 + 2], 0.08, 0.3)
            action[rob_i * 4 + 3] = PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(
                MASTER_GRIPPER_JOINT_NORMALIZE_FN(joint_pos[rob_i * 7 + 6])
            )  # Gripper position
        actions = action[None]
    elif ctrl_mode == "bimanual_box":
        action = joint_pos
        for rob_i in range(num_robots):
            action[rob_i * 7 + 6] = PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(
                MASTER_GRIPPER_JOINT_NORMALIZE_FN(joint_pos[rob_i * 7 + 6])
            )  # Gripper position
        actions = action[None]
    else:
        raise NotImplementedError(f"Unknown control mode: {ctrl_mode}")
    return actions
