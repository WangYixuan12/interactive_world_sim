import argparse
import time

from interbotix_xs_modules.arm import InterbotixManipulatorXS

from interactive_world_sim.utils.aloha_utils import move_arms, torque_off

puppet_sleep_position = (0, -1.9, 1.6, -0.012, 0.8, 0)


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
        init_node = False
        move_arms([puppet_bot_left], [puppet_sleep_position], move_time=2)
        time.sleep(2)
        torque_off(puppet_bot_left)
    if right:
        puppet_bot_right = InterbotixManipulatorXS(
            robot_model="vx300s",
            group_name="arm",
            gripper_name="gripper",
            robot_name="puppet_right",
            init_node=init_node,
        )
        init_node = False
        move_arms([puppet_bot_right], [puppet_sleep_position], move_time=2)
        time.sleep(2)
        torque_off(puppet_bot_right)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", action="store_true")
    parser.add_argument("--right", action="store_true")
    args = parser.parse_args()
    robot_sleep(args.left, args.right)
