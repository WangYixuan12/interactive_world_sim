import time
from typing import Callable, Optional  # noqa

import numpy as np
from interbotix_xs_modules.arm import InterbotixManipulatorXS
from interbotix_xs_msgs.msg import JointSingleCommand

from .aloha_conts import (
    DT,
    MASTER_GRIPPER_JOINT_MID,
    START_ARM_POSE,
)


def get_arm_joint_positions(bot: InterbotixManipulatorXS) -> np.ndarray:
    return bot.arm.core.joint_states.position[:6]


def get_arm_gripper_positions(bot: InterbotixManipulatorXS) -> float:
    joint_position = bot.gripper.core.joint_states.position[6]
    return joint_position


def move_arms(bot_list: list, target_pose_list: list, move_time: float = 1) -> None:
    num_steps = int(move_time / DT)
    curr_pose_list = [get_arm_joint_positions(bot) for bot in bot_list]
    traj_list = [
        np.linspace(curr_pose, target_pose, num_steps)
        for curr_pose, target_pose in zip(
            curr_pose_list, target_pose_list, strict=False
        )
    ]
    for t in range(num_steps):
        for bot_id, bot in enumerate(bot_list):
            bot.arm.set_joint_positions(traj_list[bot_id][t], blocking=False)
        time.sleep(DT)


def move_grippers(bot_list: list, target_pose_list: list, move_time: float) -> None:
    gripper_command = JointSingleCommand(name="gripper")
    num_steps = int(move_time / DT)
    curr_pose_list = [get_arm_gripper_positions(bot) for bot in bot_list]
    traj_list = [
        np.linspace(curr_pose, target_pose, num_steps)
        for curr_pose, target_pose in zip(
            curr_pose_list, target_pose_list, strict=False
        )
    ]
    for t in range(num_steps):
        for bot_id, bot in enumerate(bot_list):
            gripper_command.cmd = traj_list[bot_id][t]
            bot.gripper.core.pub_single.publish(gripper_command)
        time.sleep(DT)


def setup_puppet_bot(bot: InterbotixManipulatorXS) -> None:
    bot.dxl.robot_reboot_motors("single", "gripper", True)
    bot.dxl.robot_set_operating_modes("group", "arm", "position")
    bot.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
    torque_on(bot)


def setup_master_bot(bot: InterbotixManipulatorXS) -> None:
    bot.dxl.robot_set_operating_modes("group", "arm", "pwm")
    bot.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
    torque_off(bot)


def set_standard_pid_gains(bot: InterbotixManipulatorXS) -> None:
    bot.dxl.robot_set_motor_registers("group", "arm", "Position_P_Gain", 800)
    bot.dxl.robot_set_motor_registers("group", "arm", "Position_I_Gain", 0)


def set_low_pid_gains(bot: InterbotixManipulatorXS) -> None:
    bot.dxl.robot_set_motor_registers("group", "arm", "Position_P_Gain", 100)
    bot.dxl.robot_set_motor_registers("group", "arm", "Position_I_Gain", 0)


def torque_off(bot: InterbotixManipulatorXS) -> None:
    bot.dxl.robot_torque_enable("group", "arm", False)
    bot.dxl.robot_torque_enable("single", "gripper", False)


def torque_on(bot: InterbotixManipulatorXS) -> None:
    bot.dxl.robot_torque_enable("group", "arm", True)
    bot.dxl.robot_torque_enable("single", "gripper", True)


### Task parameters


def prep_puppet_robot(
    bot: InterbotixManipulatorXS, init_qpos: Optional[np.ndarray] = None
) -> None:
    # reboot gripper motors, and set operating modes for all motors
    bot.dxl.robot_set_operating_modes("group", "arm", "position")
    bot.dxl.robot_set_operating_modes("single", "gripper", "position")

    bot.dxl.robot_torque_enable("single", "gripper", False)
    bot.dxl.robot_set_motor_registers("single", "gripper", "Current_Limit", 100)
    bot.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
    bot.dxl.robot_torque_enable("single", "gripper", True)

    bot.dxl.robot_set_motor_registers("group", "arm", "Position_P_Gain", 800)
    bot.dxl.robot_set_motor_registers("group", "arm", "Position_I_Gain", 0)

    torque_on(bot)

    # move arms to starting position
    if init_qpos is None:
        start_arm_qpos = START_ARM_POSE[:6]
    else:
        start_arm_qpos = init_qpos[:6]
    move_arms([bot], [start_arm_qpos], move_time=1)
    # move grippers to starting position
    if init_qpos is None:
        move_grippers([bot], [MASTER_GRIPPER_JOINT_MID], move_time=0.5)
    else:
        move_grippers([bot], [init_qpos[6]], move_time=0.5)
