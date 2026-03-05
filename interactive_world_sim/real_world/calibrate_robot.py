import os
import time
from multiprocessing.managers import SharedMemoryManager

import numpy as np
from yixuan_utilities.draw_utils import (
    aggr_point_cloud_from_data,
    np2o3d,
    o3dVisualizer,
)
from yixuan_utilities.kinematics_helper import KinHelper

from interactive_world_sim.real_world.aloha_bimanual_master import AlohaBimanualMaster
from interactive_world_sim.real_world.aloha_bimanual_puppet import AlohaBimanualPuppet
from interactive_world_sim.real_world.multi_realsense import MultiRealsense
from interactive_world_sim.utils.aloha_conts import (
    MASTER2PUPPET_JOINT_FN,
    START_ARM_POSE,
)


def test_aloha_calibration(sides: list = ["right", "left"]) -> None:  # noqa
    o3d_visualizer = o3dVisualizer()
    o3d_visualizer.start()
    o3d_visualizer.add_triangle_mesh("origin", "origin", size=0.1)
    o3d_visualizer.render()

    # boundaries for visualizer
    boundaries = {
        "x_lower": -1.0,
        "x_upper": 1.0,
        "y_lower": -1.0,
        "y_upper": 1.0,
        "z_lower": -1.0,
        "z_upper": 1.0,
    }

    # pose manual guess
    robot_base_in_world = np.array(
        [
            [[0, 1, 0, 0.11], [-1, 0, 0, 0.42], [0, 0, 1, 0.02], [0, 0, 0, 1]],
            [[0, -1, 0, 0.11], [1, 0, 0, -0.64], [0, 0, 1, 0.02], [0, 0, 0, 1]],
        ]
    )

    # save current robot base pose in world
    curr_path = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(curr_path, "aloha_extrinsics")
    os.system(f"mkdir -p {save_dir}")
    for rob_i, side in enumerate(sides):
        np.save(
            os.path.join(save_dir, f"{side}_base_pose_in_world.npy"),
            robot_base_in_world[rob_i],
        )

    # start robot
    kin_helper = KinHelper(robot_name="trossen_vx300s")
    shm_manager = SharedMemoryManager()
    shm_manager.start()
    init_qpos = np.array(START_ARM_POSE)
    puppet_robot = AlohaBimanualPuppet(
        shm_manager=shm_manager,
        verbose=False,
        frequency=50,
        robot_sides=["right", "left"],
        init_qpos=init_qpos,
    )
    puppet_robot.start()
    master_robot = AlohaBimanualMaster(
        shm_manager=shm_manager,
        verbose=False,
        frequency=50,
        robot_sides=["right", "left"],
    )
    master_robot.start()

    # visualize realsense point cloud
    with MultiRealsense(
        resolution=(640, 480),
        put_downsample=False,
        enable_color=True,
        enable_depth=True,
        enable_infrared=False,
        verbose=False,
    ) as realsense:
        realsense.set_exposure(exposure=60, gain=64)
        realsense.set_white_balance(white_balance=3800)
        realsense.set_depth_preset("High Density")
        realsense.set_depth_exposure(3000, 16)

        time.sleep(3.0)  # wait for camera to warm up

        while True:
            out = realsense.get()

            # set robot joint pos
            master_state = master_robot.get_motion_state()
            target_state = master_state["joint_pos"].copy()
            for rob_i in range(2):
                target_state[7 * rob_i + 6] = MASTER2PUPPET_JOINT_FN(
                    master_state["joint_pos"][7 * rob_i + 6]
                )
            target_time = time.time() + 0.1
            puppet_robot.set_actions(target_state, target_time=target_time)

            # compute robot point cloud
            curr_robot_joint_pos = puppet_robot.get_state()["curr_full_joint_pos"]
            base_pcd_ls = []
            for rob_i, _ in enumerate(sides):
                base_pcd = kin_helper.compute_robot_pcd(
                    curr_robot_joint_pos[rob_i * 8 : (rob_i + 1) * 8],
                    pcd_name="rob_pcd",
                )
                base_pcd_i = base_pcd.copy()
                base_pcd_i = (
                    robot_base_in_world[rob_i]
                    @ np.concatenate(
                        [base_pcd_i, np.ones((base_pcd_i.shape[0], 1))], axis=-1
                    ).T
                )
                base_pcd_i = base_pcd_i[:3].T
                base_pcd_ls.append(base_pcd_i)
            base_pcd = np.concatenate(base_pcd_ls, axis=0)

            # visualize camera point cloud
            colors = np.stack([value["color"] for value in out.values()])[..., ::-1]
            depths = np.stack([value["depth"] for value in out.values()]) / 1000.0
            intrinsics = np.stack([value["intrinsics"] for value in out.values()])
            extrinsics = np.stack([value["extrinsics"] for value in out.values()])
            pcd = aggr_point_cloud_from_data(
                colors=colors,
                depths=depths,
                Ks=intrinsics,
                poses=extrinsics,
                downsample=False,
                boundaries=boundaries,
            )

            o3d_visualizer.update_pcd(pcd, "pcd")
            o3d_visualizer.update_pcd(np2o3d(base_pcd), "rob_pcd")

            # render
            o3d_visualizer.render()


if __name__ == "__main__":
    test_aloha_calibration()
