import argparse

import numpy as np
from interbotix_xs_modules.arm import InterbotixManipulatorXS

from interactive_world_sim.utils.aloha_conts import (
    ARM_OFFSET,
    START_ARM_POSE,
)
from interactive_world_sim.utils.aloha_utils import move_arms, torque_on

puppet_sleep_position = (0, -1.9, 1.6, -0.012, 0.8, 0)

init_pos = np.array(START_ARM_POSE)
init_pos[:6] += np.array(ARM_OFFSET[:6])
init_pos[8:14] += np.array(ARM_OFFSET[6:])


def robot_sleep(left: bool, right: bool) -> None:
    """Put the robot arms to sleep position and turn off torque."""
    init_node = True
    if left:
        puppet_bot_left = InterbotixManipulatorXS(
            robot_model="vx300s",
            group_name="arm",
            gripper_name="gripper",
            robot_name="puppet_left",
            init_node=init_node,
        )
        torque_on(puppet_bot_left)
        init_node = False
        move_arms([puppet_bot_left], [init_pos[8:14]], move_time=2)
    if right:
        puppet_bot_right = InterbotixManipulatorXS(
            robot_model="vx300s",
            group_name="arm",
            gripper_name="gripper",
            robot_name="puppet_right",
            init_node=init_node,
        )
        torque_on(puppet_bot_right)
        init_node = False
        move_arms([puppet_bot_right], [init_pos[:6]], move_time=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", action="store_true")
    parser.add_argument("--right", action="store_true")
    args = parser.parse_args()
    robot_sleep(args.left, args.right)
