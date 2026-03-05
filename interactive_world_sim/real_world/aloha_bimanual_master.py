import multiprocessing as mp
import time
from multiprocessing.managers import SharedMemoryManager
from typing import Any

import numpy as np
from interbotix_xs_modules.arm import InterbotixManipulatorXS

from interactive_world_sim.utils.aloha_conts import (
    DT,
    MASTER_GRIPPER_JOINT_CLOSE,
    MASTER_GRIPPER_JOINT_MID,
    START_ARM_POSE,
)
from interactive_world_sim.utils.aloha_utils import (
    get_arm_gripper_positions,
    move_arms,
    move_grippers,
    torque_off,
    torque_on,
)
from interactive_world_sim.utils.shared_memory_ring_buffer import SharedMemoryRingBuffer


def prep_robots(master_bot: InterbotixManipulatorXS) -> None:
    # reboot gripper motors, and set operating modes for all motors
    master_bot.dxl.robot_set_operating_modes("group", "arm", "position")
    master_bot.dxl.robot_set_operating_modes("single", "gripper", "position")
    torque_on(master_bot)

    # move arms to starting position
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
    torque_off(master_bot)
    print("Started!")


class AlohaBimanualMaster(mp.Process):
    """Aloha Master Robot"""

    def __init__(
        self,
        shm_manager: SharedMemoryManager,
        get_max_k: int = 30,
        frequency: int = 200,
        robot_sides: list = ["right"],  # noqa
        verbose: bool = False,
    ):
        """Read from master robot"""
        super().__init__()

        # copied variables
        self.frequency = frequency
        self.robot_sides = robot_sides
        self.num_bot = len(robot_sides)
        self.verbose = verbose

        example = {
            "joint_pos": np.array(START_ARM_POSE[:7] * self.num_bot),
            "receive_timestamp": time.time(),
        }
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency,
        )

        # shared variables
        self.ready_event = mp.Event()
        self.stop_event = mp.Event()
        self.ring_buffer = ring_buffer

    # ======= get state APIs ==========
    def get_motion_state(self) -> dict:
        """Get the current state of the robot"""
        state = self.ring_buffer.get()
        return state

    # ========== start stop API ===========

    def start(self, wait: bool = True) -> None:
        """Start the process"""
        super().start()
        if wait:
            self.ready_event.wait()

    def stop(self, wait: bool = True) -> None:
        """Stop the process"""
        self.stop_event.set()
        if wait:
            self.join()

    def __enter__(self) -> "AlohaBimanualMaster":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()

    # ========= main loop ==========
    def run(self) -> None:
        """Read from master robot"""
        try:
            # set up robot
            self.master_bots = []
            for rob_i, side in enumerate(self.robot_sides):
                if rob_i == 0:
                    init_node = True
                else:
                    init_node = False
                self.master_bots.append(
                    InterbotixManipulatorXS(
                        robot_model="wx250s",
                        group_name="arm",
                        gripper_name="gripper",
                        robot_name=f"master_{side}",
                        init_node=init_node,
                    )
                )
                prep_robots(self.master_bots[-1])
                press_to_start(self.master_bots[-1])

            # send one message immediately so client can start reading
            self.ring_buffer.put(
                {
                    "joint_pos": np.array(START_ARM_POSE[:7] * self.num_bot),
                    # 'ee_pose': np.array(START_EE_POSE).astype(np.float32),
                    "receive_timestamp": time.time(),
                }
            )
            self.ready_event.set()

            while not self.stop_event.is_set():
                t_start = time.perf_counter()
                joint_pos_ls = []
                for master_bot in self.master_bots:
                    joint_pos = master_bot.dxl.robot_get_joint_states().position[:7]
                    joint_pos_ls.append(joint_pos)

                receive_timestamp = time.time()
                joint_pos = np.concatenate(joint_pos_ls)
                last_joint_pos = joint_pos.copy()
                for i in range(len(self.master_bots)):
                    if np.allclose(joint_pos[7 * i : 7 * (i + 1)], np.zeros(7)):
                        joint_pos[7 * i : 7 * (i + 1)] = last_joint_pos[
                            7 * i : 7 * (i + 1)
                        ]
                self.ring_buffer.put(
                    {
                        "joint_pos": joint_pos,
                        # 'ee_pose': ee_pose, # (xyz, quat, gripper)
                        "receive_timestamp": receive_timestamp,
                    }
                )
                time.sleep(1 / self.frequency)
                if self.verbose:
                    print(f"AlohaMaster: {1/(time.perf_counter() - t_start)} Hz")
        finally:
            self.ready_event.set()


def test() -> None:
    with SharedMemoryManager() as shm_manager:
        with AlohaBimanualMaster(
            shm_manager=shm_manager,
            get_max_k=10,
            frequency=100,
            robot_sides=["right", "left"],
        ) as master:
            for _ in range(1000):
                state = master.get_motion_state()
                print("receive_timestamp: ", state["receive_timestamp"])
                print("joint_pos: ", state["joint_pos"])
                # print('ee_pose: ', state['ee_pose'])
                time.sleep(0.1)


if __name__ == "__main__":
    test()
