import enum
import multiprocessing as mp
import os
import time
import warnings
from multiprocessing.managers import SharedMemoryManager
from typing import List, Optional

import numpy as np
import transforms3d
from interbotix_xs_modules.arm import InterbotixManipulatorXS
from interbotix_xs_msgs.msg import JointSingleCommand
from yixuan_utilities.kinematics_helper import KinHelper

from interactive_world_sim.utils.action_utils import action_primitive_to_joint_pos

# from yixuan_utilities.kinematics_helper_v2 import KinHelper
from interactive_world_sim.utils.aloha_conts import (
    ARM_OFFSET,
    START_ARM_POSE,
    START_ARM_POSE_SINGLE,
    START_ARM_POSE_SINGLE_SWEEP,
)
from interactive_world_sim.utils.aloha_utils import prep_puppet_robot
from interactive_world_sim.utils.shared_memory_queue import SharedMemoryQueue
from interactive_world_sim.utils.shared_memory_ring_buffer import SharedMemoryRingBuffer


class Command(enum.Enum):
    """Command types for the puppet process."""

    STOP = 0
    ACTION = 1


class AlohaBimanualPuppet(mp.Process):
    """Process for controlling a bimanual puppet robot.

    This process is responsible for sending commands to the robot with
    predictable latency. It uses a separate process (bypassing the Python GIL)
    to ensure real-time control of the robot.
    """

    def __init__(
        self,
        shm_manager: SharedMemoryManager,
        frequency: int = 50,
        launch_timeout: float = 3.0,
        action_dim: int = 14,
        verbose: bool = False,
        get_max_k: int = 10,
        robot_sides: Optional[List[str]] = None,
        extrinsics_dir: str = os.path.join(
            os.path.dirname(__file__), "aloha_extrinsics"
        ),
        init_qpos: np.ndarray = np.array(START_ARM_POSE),  # noqa
        ctrl_mode: str = "joint",
    ) -> None:
        """Initialize the puppet process.

        Args:
            shm_manager: Shared memory manager.
            frequency: Control frequency (must be between 1 and 500 Hz).
            launch_timeout: Timeout for launching.
            action_dim: Dimension of the action space.
            verbose: Enable verbose logging.
            get_max_k: Maximum number of samples in the ring buffer.
            robot_sides: List of robot sides to control. Defaults to ["right"].
            extrinsics_dir: Directory containing extrinsics files.
            init_qpos: Optional initial joint positions.
            ctrl_mode: Control mode for the robot.
        """
        # Validate frequency.
        assert 0 < frequency <= 500

        if robot_sides is None:
            robot_sides = ["right"]
        side_str = "_".join(robot_sides)
        super().__init__(name=f"AlohaPuppet_{side_str}")
        self.frequency = frequency
        self.launch_timeout = launch_timeout
        self.verbose = verbose
        self.robot_sides = robot_sides
        self.num_bot = len(robot_sides)
        self.puppet_bots: List[InterbotixManipulatorXS] = []
        init_qpos = init_qpos[: len(robot_sides) * 8]
        if ctrl_mode == "single_push":
            init_qpos = np.array(START_ARM_POSE_SINGLE)
            assert (
                len(robot_sides) == 1
            ), "Single push mode requires exactly one robot side."
        elif ctrl_mode == "single_sweep" or ctrl_mode == "single_wipe":
            init_qpos = np.array(START_ARM_POSE_SINGLE_SWEEP)
            assert (
                len(robot_sides) == 1
            ), "Single end-effector mode requires exactly one robot side."

        for i in range(len(robot_sides)):
            init_qpos[i * 8 : i * 8 + 6] += ARM_OFFSET[i * 6 : i * 6 + 6]
        self.init_qpos = init_qpos
        assert len(self.init_qpos) == 8 * self.num_bot
        os.system(f"mkdir -p {extrinsics_dir}")
        extrinsics_paths = [
            os.path.join(extrinsics_dir, f"{robot_side}_base_pose_in_world.npy")
            for robot_side in robot_sides
        ]
        self.base_pose_in_world: np.ndarray = np.tile(
            np.eye(4)[None], (len(robot_sides), 1, 1)
        )
        for bot_i, extrinsics_path in enumerate(extrinsics_paths):
            if not os.path.exists(extrinsics_path):
                self.base_pose_in_world[bot_i] = np.eye(4)
                warnings.warn(
                    f"extrinsics_path {extrinsics_path} does not exist, using identity "
                    "matrix as base pose in world",
                    stacklevel=2,
                )
            else:
                self.base_pose_in_world[bot_i] = np.load(extrinsics_path)

        # Build input queue.
        if ctrl_mode == "single_wipe" or ctrl_mode == "single_sweep":
            action_example = np.empty(
                shape=(4,), dtype=np.float32
            )  # x, y, height, gripper
        else:
            action_example = np.empty(shape=(action_dim,), dtype=np.float32)

        example = {
            "cmd": Command.ACTION.value,
            "target_action": action_example,
            "target_time": time.time(),
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager, examples=example, buffer_size=256
        )

        # Build ring buffer.
        example = {
            "curr_joint_pos": np.empty(
                shape=(7 * self.num_bot,), dtype=np.float32
            ),  # 6dof + gripper
            "curr_full_joint_pos": np.empty(
                shape=(8 * self.num_bot,), dtype=np.float32
            ),  # 6dof + 2 gripper
            "curr_ee_pose": np.empty(
                shape=(7 * self.num_bot,), dtype=np.float32
            ),  # (xyz, euler_xyz, gripper)
            "robot_base_pose_in_world": np.empty(
                shape=(self.num_bot, 4, 4), dtype=np.float32
            ),
            "robot_receive_timestamp": time.time(),
        }
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency,
        )

        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer

        # Build teleop robot helper.
        self.kin_helper = KinHelper(robot_name="trossen_vx300s")
        self.ctrl_mode = ctrl_mode
        self.last_joint_pos = np.array(START_ARM_POSE)
        self.dt = 1 / self.frequency

        # # PID setting
        # if self.ctrl_mode == "joint":
        #     self.curr_vel = np.zeros(action_dim)
        #     self.k_p = 50
        #     self.k_v = 10
        #     self.vel_lim = 0.2
        #     self.acc_lim = 1.0
        # # NOTE
        # elif self.ctrl_mode == "bimanual_push":
        #     assert len(robot_sides) == 2
        #     self.curr_vel = np.zeros(4)
        #     self.k_p = 50
        #     self.k_v = 10
        #     self.vel_lim = 0.2
        #     self.acc_lim = 1.0
        # elif self.ctrl_mode == "single_ee":
        #     self.curr_vel = np.zeros(3)
        #     self.k_p = 50
        #     self.k_v = 10
        #     self.vel_lim = 0.2
        #     self.acc_lim = 1.0
        # elif self.ctrl_mode == "single_push":
        #     self.curr_vel = np.zeros(3)
        #     self.k_p = 50
        #     self.k_v = 10
        #     self.vel_lim = 0.2
        #     self.acc_lim = 1.0
        # elif self.ctrl_mode == "single_sweep":
        #     self.curr_vel = np.zeros(4)
        #     self.k_p = 50
        #     self.k_v = 10
        #     self.vel_lim = 0.2
        #     self.acc_lim = 1.0
        # elif self.ctrl_mode == "single_wipe":
        #     self.curr_vel = np.zeros(4)
        #     self.k_p = 50
        #     self.k_v = 10
        #     self.vel_lim = 0.2
        #     self.acc_lim = 1.0
        # else:
        #     raise ValueError(f"Unknown control mode: {self.ctrl_mode}")

    def start(self, wait: bool = True) -> None:
        """Start the puppet process."""
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"Puppet aloha process spawned at {self.pid}")

    def stop(self, wait: bool = True) -> None:
        """Stop the puppet process by sending a STOP command."""
        message = {"cmd": Command.STOP.value}
        self.input_queue.put(message)
        if wait:
            self.stop_wait()

    def start_wait(self) -> None:
        """Wait until the process is ready."""
        self.ready_event.wait()
        assert self.is_alive()

    def stop_wait(self) -> None:
        """Wait for the process to terminate."""
        self.join()

    @property
    def is_ready(self) -> bool:
        """Return True if the puppet process is ready."""
        return self.ready_event.is_set()

    def __enter__(self) -> "AlohaBimanualPuppet":
        """Enter the runtime context related to this object."""
        self.start()
        return self

    def __exit__(
        self,
        exc_type: Optional[Exception],
        exc_value: Optional[Exception],
        traceback: Optional[Exception],
    ) -> None:
        """Exit the runtime context and stop the process."""
        self.stop()

    def set_actions(self, target_action: np.ndarray, target_time: float) -> None:
        """Send a joint target command to the puppet process.

        Args:
            target_action: Target actions
            target_time: Time at which to apply the target.
        """
        message = {
            "cmd": Command.ACTION.value,
            "target_action": target_action,
            "target_time": target_time,
        }
        self.input_queue.put(message)

    def set_joint(self, joint_states: np.ndarray) -> None:
        """Set the joint positions of the puppet robot."""
        gripper_command = JointSingleCommand(name="gripper")
        for s_i in range(len(self.robot_sides)):
            self.puppet_bots[s_i].arm.set_joint_positions(
                joint_states[s_i * 7 : s_i * 7 + 6] + ARM_OFFSET[s_i * 6 : s_i * 6 + 6],
                blocking=False,
            )
            gripper_command.cmd = joint_states[s_i * 7 + 6]
            self.puppet_bots[s_i].gripper.core.pub_single.publish(gripper_command)

    def get_state(self, k: Optional[int] = None, out: Optional[dict] = None) -> dict:
        """Retrieve the current robot state from the ring buffer."""
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k, out=out)

    def get_all_state(self) -> dict:
        """Retrieve all robot state entries from the ring buffer."""
        return self.ring_buffer.get_all()

    def save_state(self) -> dict:
        """Update and save the robot state in the ring buffer."""
        state = {}

        curr_joint_pos_ls = []
        curr_ee_pose_ls = []
        full_joint_qpos_ls = []
        for bot_i in range(self.num_bot):
            # Get current joint positions.
            curr_joint_states = self.puppet_bots[bot_i].dxl.robot_get_joint_states()
            curr_joint_pos = np.array(curr_joint_states.position[:7])
            curr_joint_pos[:6] = (
                curr_joint_pos[:6] - ARM_OFFSET[bot_i * 6 : bot_i * 6 + 6]
            )
            curr_joint_pos_ls.append(curr_joint_pos)

            # Get current full joint positions for FK.
            full_joint_qpos = np.concatenate(
                [curr_joint_pos[:6], np.array(curr_joint_states.position[7:])]
            )
            full_joint_qpos_ls.append(full_joint_qpos)

            # Get current end-effector pose.
            curr_ee_pose_mat = self.kin_helper.compute_fk_from_link_idx(
                full_joint_qpos, [self.kin_helper.sapien_eef_idx]
            )[0]
            curr_ee_pose_mat = self.base_pose_in_world[bot_i] @ curr_ee_pose_mat
            curr_ee_pose_mat = (
                np.linalg.inv(self.base_pose_in_world[0]) @ curr_ee_pose_mat
            )
            curr_ee_pose = np.concatenate(
                [
                    curr_ee_pose_mat[:3, 3],
                    np.array(transforms3d.euler.mat2euler(curr_ee_pose_mat[:3, :3])),
                    [curr_joint_pos[-1]],
                ]
            )
            curr_ee_pose_ls.append(curr_ee_pose)

        curr_joint_pos = np.concatenate(curr_joint_pos_ls)
        full_joint_qpos = np.concatenate(full_joint_qpos_ls)
        curr_ee_pose = np.concatenate(curr_ee_pose_ls)

        state = {
            "curr_joint_pos": curr_joint_pos,
            "curr_full_joint_pos": full_joint_qpos,
            "curr_ee_pose": curr_ee_pose,
            "robot_base_pose_in_world": self.base_pose_in_world,
            "robot_receive_timestamp": time.time(),
        }
        self.ring_buffer.put(state)
        return state

    def convert_action_to_joint_cmd(self, action: np.ndarray) -> np.ndarray:
        """Convert action to joint command."""
        joint_pos = action_primitive_to_joint_pos(
            action_primitive=action,
            ctrl_mode=self.ctrl_mode,
            base_pose_in_world=self.base_pose_in_world,
            kin_helper=self.kin_helper,
            last_joint_pos=self.last_joint_pos,
        )
        joint_pos = 0.2 * joint_pos + 0.8 * self.last_joint_pos[: joint_pos.shape[0]]
        return joint_pos

    def run(self) -> None:
        """Main control loop executed in a separate process."""
        try:
            # Set up puppet robots.
            self.puppet_bots = []
            for bot_i, side in enumerate(self.robot_sides):
                init_node = bot_i == 0
                self.puppet_bots.append(
                    InterbotixManipulatorXS(
                        robot_model="vx300s",
                        group_name="arm",
                        gripper_name="gripper",
                        robot_name=f"puppet_{side}",
                        init_node=init_node,
                    )
                )
                prep_puppet_robot(self.puppet_bots[-1], self.init_qpos)
            self.last_joint_pos = None
            self.last_eef_pose = None
            self.max_joint_vels = np.array(
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10000.0] * self.num_bot
            )  # rad/s

            self.ready_event.set()

            gripper_command = JointSingleCommand(name="gripper")

            keep_running = True
            while keep_running:
                t_start = time.monotonic()

                # Fetch a single command from the input queue.
                if self.input_queue.empty():
                    self.save_state()
                    continue
                command = self.input_queue.get()
                cmd = command["cmd"]
                if cmd == Command.STOP.value:
                    keep_running = False
                    break
                elif cmd == Command.ACTION.value:
                    target_time = float(command["target_time"])
                    # Translate global time to monotonic time.
                    target_time = time.monotonic() - time.time() + target_time

                    # Wait until target time is reached.
                    while time.monotonic() < target_time:  # loose on time
                        step_time = time.monotonic()

                        # Save state.
                        state = self.save_state()
                        if self.last_joint_pos is None:
                            self.last_joint_pos = state["curr_joint_pos"].copy()

                        cmd = self.convert_action_to_joint_cmd(command["target_action"])
                        cmd = self.last_joint_pos + np.clip(
                            cmd - self.last_joint_pos,
                            -self.max_joint_vels * self.dt,
                            self.max_joint_vels * self.dt,
                        )

                        for bot_i in range(self.num_bot):
                            self.puppet_bots[bot_i].arm.set_joint_positions(
                                cmd[bot_i * 7 : bot_i * 7 + 6]
                                + ARM_OFFSET[bot_i * 6 : bot_i * 6 + 6],
                                blocking=False,
                            )
                            gripper_command.cmd = cmd[bot_i * 7 + 6]
                            self.puppet_bots[bot_i].gripper.core.pub_single.publish(
                                gripper_command
                            )

                        time.sleep(max(0, self.dt - (time.monotonic() - step_time)))
                        freq = 1 / (time.monotonic() - step_time)
                        if freq < self.frequency - 10:
                            warnings.warn(f"Puppet Actual frequency {freq} Hz")  # noqa
                        if hasattr(self, "last_execute_time"):
                            exec_freq = 1 / (time.monotonic() - self.last_execute_time)
                            if exec_freq < self.frequency - 10:
                                warnings.warn(
                                    f"Puppet Execute frequency {exec_freq} Hz",
                                    stacklevel=2,
                                )
                        self.last_execute_time: float = time.monotonic()
                        self.last_joint_pos = cmd

                    time.sleep(max(0, self.dt - (time.monotonic() - t_start)))
                else:
                    keep_running = False
                    break

                time.sleep(max(0, self.dt - (time.monotonic() - t_start)))
            # End of main loop.
        finally:
            self.ready_event.set()


# def test_joint_teleop(robot_sides: Optional[List[str]] = None) -> None:
#     """Run a test teleoperation scenario."""
#     if robot_sides is None:
#         robot_sides = ["right", "left"]
#     shm_manager = SharedMemoryManager()
#     shm_manager.start()
#     frequency = 50
#     puppet_robot = AlohaBimanualPuppet(
#         shm_manager=shm_manager,
#         robot_sides=robot_sides,
#         frequency=frequency,
#         verbose=True,
#     )
#     puppet_robot.start()
#     master_robot = AlohaBimanualMaster(
#         shm_manager=shm_manager, robot_sides=robot_sides, frequency=frequency
#     )
#     master_robot.start()
#     while True:
#         dt = 0.1
#         start_time = time.monotonic()
#         state = master_robot.get_motion_state()
#         target_state = state["joint_pos"].copy()
#         for rob_i in range(len(robot_sides)):
#             target_state[7 * rob_i + 6] = MASTER2PUPPET_JOINT_FN(
#                 state["joint_pos"][7 * rob_i + 6]
#             )
#         target_time = time.time() + 0.1
#         puppet_robot.set_target_joint_pos(target_state, target_time=target_time)
#         time.sleep(max(0, dt - (time.monotonic() - start_time)))


# if __name__ == "__main__":
#     test_joint_teleop()
