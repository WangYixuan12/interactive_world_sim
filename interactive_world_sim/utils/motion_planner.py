"""Motion planner for generating human-like bimanual T-pushing trajectories.
Generates 4 types of motions: linear, rotating, random contact, and random no-contact.
"""

import random
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .trajectory_primitives import (
    BimanualCoordination,
    CurvePrimitive,
    LinearPrimitive,
    TrajectoryConfig,
)


@dataclass
class TShapeInfo:
    """Information about T-shape pose and keypoints."""

    center: np.ndarray  # (x, y) center of T
    rotation: float  # rotation angle in radians
    keypoints: np.ndarray  # Nx2 array of keypoint positions
    width: float  # width of T top bar
    height: float  # height of T stem
    thickness: float  # thickness of T
    pose: np.ndarray  # world_t_obj


@dataclass
class WorkspaceConstraints:
    """Workspace constraints for arm movements."""

    x_min: float = 0.2
    x_max: float = 0.8
    y_min: float = -0.3
    y_max: float = 0.3
    min_arm_distance: float = 0.1  # Minimum distance between arms
    min_t_distance: float = 0.05  # Minimum distance from T when not contacting


class TGeometryAnalyzer:
    """Analyzes T-shape geometry and computes manipulation targets."""

    def __init__(self, t_info: TShapeInfo):
        self.t_info = t_info
        self._compute_geometric_features()

    def _compute_geometric_features(self):
        """Compute key geometric features of the T."""
        center = self.t_info.center
        rotation = self.t_info.rotation

        # Rotation matrix
        cos_r, sin_r = np.cos(rotation), np.sin(rotation)
        R = np.array([[cos_r, -sin_r], [sin_r, cos_r]])

        # Key points in local coordinates
        local_points = {
            "top_left": self.t_info.keypoints[0, :2],
            "top_right": self.t_info.keypoints[1, :2],
            "top_bar_bottom_side_left": self.t_info.keypoints[2, :2],
            "top_bar_bottom_side_right": self.t_info.keypoints[3, :2],
            "stem_top_left": self.t_info.keypoints[4, :2],
            "stem_top_right": self.t_info.keypoints[5, :2],
            "stem_bottom_left": self.t_info.keypoints[6, :2],
            "stem_bottom_right": self.t_info.keypoints[7, :2],
        }

        # Transform to world coordinates
        self.key_points = {}
        for name, local_pt in local_points.items():
            self.key_points[name] = center + R @ local_pt

        # Compute contact regions
        self._compute_contact_regions()

    def _compute_contact_regions(self):
        """Compute feasible contact points along T edges."""
        # Generate contact points along edges
        self.contact_points = []
        kypts_order = [
            "top_left",
            "top_right",
            "top_bar_bottom_side_right",
            "stem_top_right",
            "stem_bottom_right",
            "stem_bottom_left",
            "stem_top_left",
            "top_bar_bottom_side_left",
        ]

        for i in range(len(kypts_order)):
            last_kypt = self.key_points[kypts_order[i - 1]]
            curr_kypt = self.key_points[kypts_order[i]]
            side_len = np.linalg.norm(last_kypt - curr_kypt)
            weights = np.arange(0, side_len, 0.002) / side_len
            world_pts = last_kypt[None] + weights[:, None] * (
                curr_kypt[None] - last_kypt[None]
            )
            self.contact_points.append(world_pts)

        self.contact_points = np.concatenate(self.contact_points)

    def select_contact_point(self, side: str) -> np.ndarray:
        """Select contact points based on side preference.

        Args:
            side: "left", "right", "up", or "down" to select points on the specified side

        Returns:
            Selected contact points as (N, 2) array
        """
        if side not in ["left", "right", "up", "down"]:
            raise ValueError(
                f"Side must be 'left', 'right', 'up', or 'down', got '{side}'"
            )

        # Flatten all contact points from all edges into a single array
        all_points = self.contact_points.reshape(-1, 2)

        if len(all_points) == 0:
            return np.array([]).reshape(0, 2)

        # Use a tolerance for grouping similar coordinates
        tolerance = 0.002  # 2mm tolerance for coordinate grouping

        if side in ["left", "right"]:
            # Group by y-coordinate and select by x-coordinate
            return self._select_by_x_coordinate(all_points, side, tolerance)
        else:  # side in ["up", "down"]
            # Group by x-coordinate and select by y-coordinate
            return self._select_by_y_coordinate(all_points, side, tolerance)

    def _select_by_x_coordinate(
        self, all_points: np.ndarray, side: str, tolerance: float
    ) -> np.ndarray:
        """Group by y-coordinate and select by x-coordinate (left/right)."""
        # Sort points by y coordinate first
        sorted_indices = np.argsort(all_points[:, 1])
        sorted_points = all_points[sorted_indices]

        selected_points = []
        current_y_group = []
        current_y_ref = None

        for point in sorted_points:
            x, y = point

            # Check if this point belongs to the current y group
            if current_y_ref is None or abs(y - current_y_ref) <= tolerance:
                current_y_group.append(point)
                current_y_ref = y if current_y_ref is None else current_y_ref
            else:
                # Process the current group and start a new one
                if current_y_group:
                    selected_point = self._select_point_from_group_by_x(
                        current_y_group, side
                    )
                    selected_points.append(selected_point)

                current_y_group = [point]
                current_y_ref = y

        # Process the last group
        if current_y_group:
            selected_point = self._select_point_from_group_by_x(current_y_group, side)
            selected_points.append(selected_point)

        return (
            np.array(selected_points) if selected_points else np.array([]).reshape(0, 2)
        )

    def _select_by_y_coordinate(
        self, all_points: np.ndarray, side: str, tolerance: float
    ) -> np.ndarray:
        """Group by x-coordinate and select by y-coordinate (up/down)."""
        # Sort points by x coordinate first
        sorted_indices = np.argsort(all_points[:, 0])
        sorted_points = all_points[sorted_indices]

        selected_points = []
        current_x_group = []
        current_x_ref = None

        for point in sorted_points:
            x, y = point

            # Check if this point belongs to the current x group
            if current_x_ref is None or abs(x - current_x_ref) <= tolerance:
                current_x_group.append(point)
                current_x_ref = x if current_x_ref is None else current_x_ref
            else:
                # Process the current group and start a new one
                if current_x_group:
                    selected_point = self._select_point_from_group_by_y(
                        current_x_group, side
                    )
                    selected_points.append(selected_point)

                current_x_group = [point]
                current_x_ref = x

        # Process the last group
        if current_x_group:
            selected_point = self._select_point_from_group_by_y(current_x_group, side)
            selected_points.append(selected_point)

        return (
            np.array(selected_points) if selected_points else np.array([]).reshape(0, 2)
        )

    def _select_point_from_group_by_x(
        self, points: List[np.ndarray], side: str
    ) -> np.ndarray:
        """Select the leftmost or rightmost point from a group of points with similar y values."""
        if not points:
            return np.array([])

        points_array = np.array(points)
        x_coords = points_array[:, 0]

        if side == "left":
            # Select point with minimum x coordinate
            min_x_idx = np.argmin(x_coords)
        else:  # side == "right"
            # Select point with maximum x coordinate
            min_x_idx = np.argmax(x_coords)

        return points_array[min_x_idx]

    def _select_point_from_group_by_y(
        self, points: List[np.ndarray], side: str
    ) -> np.ndarray:
        """Select the topmost or bottommost point from a group of points with similar x values."""
        if not points:
            return np.array([])

        points_array = np.array(points)
        y_coords = points_array[:, 1]

        if side == "up":
            # Select point with maximum y coordinate
            max_y_idx = np.argmax(y_coords)
        else:  # side == "down"
            # Select point with minimum y coordinate
            max_y_idx = np.argmin(y_coords)

        return points_array[max_y_idx]

    def get_linear_push_waypoints(
        self, push_direction: str
    ) -> Dict[str, List[np.ndarray]]:
        """Generate waypoints for linear pushing motions."""
        push_distance = 0.15
        approach_distance = 0.08

        if push_direction == "horizontal_right":
            # Push T horizontally to the right
            left_contacts = self.select_contact_point("left")
            left_contact = left_contacts[np.random.choice(len(left_contacts))]

            # Approach points
            left_approach_theta = np.random.uniform(-np.pi / 6.0, np.pi / 6.0)
            left_approach_dir = np.array(
                [np.cos(left_approach_theta), np.sin(left_approach_theta)]
            )
            left_approach = left_contact - left_approach_dir * approach_distance

            # Target points
            left_target_theta = np.random.uniform(-np.pi / 6.0, np.pi / 6.0)
            left_target_dir = np.array(
                [np.cos(left_target_theta), np.sin(left_target_theta)]
            )
            left_target = left_contact + left_target_dir * push_distance

            return {
                "left": [left_approach, left_contact, left_target],
                "right": [],
            }

        elif push_direction == "horizontal_left":
            # Push T horizontally to the left
            right_contacts = self.select_contact_point("right")
            right_contact = right_contacts[np.random.choice(len(right_contacts))]

            right_approach_theta = np.random.uniform(-np.pi / 6.0, np.pi / 6.0)
            right_approach_dir = np.array(
                [np.cos(right_approach_theta), np.sin(right_approach_theta)]
            )
            right_approach = right_contact + right_approach_dir * approach_distance

            right_target_theta = np.random.uniform(-np.pi / 6.0, np.pi / 6.0)
            right_target_dir = np.array(
                [np.cos(right_target_theta), np.sin(right_target_theta)]
            )
            right_target = right_contact - right_target_dir * push_distance

            return {
                "left": [],
                "right": [right_approach, right_contact, right_target],
            }

        elif push_direction == "vertical_up":
            # Push T vertically upward
            up_contacts = self.select_contact_point("down")
            contacts = up_contacts[np.random.choice(len(up_contacts), 2)]
            left_contact = contacts[contacts[:, 0].argmin()]
            right_contact = contacts[contacts[:, 0].argmax()]

            left_approach_theta = np.random.uniform(-np.pi / 6.0, np.pi / 6.0)
            right_approach_theta = np.random.uniform(-np.pi / 6.0, np.pi / 6.0)
            left_approach_dir = np.array(
                [np.sin(left_approach_theta), np.cos(left_approach_theta)]
            )
            right_approach_dir = np.array(
                [np.sin(right_approach_theta), np.cos(right_approach_theta)]
            )
            left_approach = left_contact - left_approach_dir * approach_distance
            right_approach = right_contact - right_approach_dir * approach_distance

            left_target_theta = np.random.uniform(-np.pi / 6.0, np.pi / 6.0)
            right_target_theta = np.random.uniform(-np.pi / 6.0, np.pi / 6.0)
            left_target_dir = np.array(
                [np.sin(left_target_theta), np.cos(left_target_theta)]
            )
            right_target_dir = np.array(
                [np.sin(right_target_theta), np.cos(right_target_theta)]
            )
            left_target = left_contact + left_target_dir * push_distance
            right_target = right_contact + right_target_dir * push_distance

            return {
                "left": [left_approach, left_contact, left_target],
                "right": [right_approach, right_contact, right_target],
            }

        elif push_direction == "vertical_down":
            # Push T vertically downward
            down_contacts = self.select_contact_point("up")
            contacts = np.random.choice(len(down_contacts), 2)
            contacts = down_contacts[contacts]
            left_contact = contacts[contacts[:, 0].argmin()]
            right_contact = contacts[contacts[:, 0].argmax()]

            left_approach_theta = np.random.uniform(-np.pi / 6.0, np.pi / 6.0)
            right_approach_theta = np.random.uniform(-np.pi / 6.0, np.pi / 6.0)
            left_approach_dir = np.array(
                [np.sin(left_approach_theta), np.cos(left_approach_theta)]
            )
            right_approach_dir = np.array(
                [np.sin(right_approach_theta), np.cos(right_approach_theta)]
            )
            left_approach = left_contact + left_approach_dir * approach_distance
            right_approach = right_contact + right_approach_dir * approach_distance

            left_target_theta = np.random.uniform(-np.pi / 6.0, np.pi / 6.0)
            right_target_theta = np.random.uniform(-np.pi / 6.0, np.pi / 6.0)
            left_target_dir = np.array(
                [np.sin(left_target_theta), np.cos(left_target_theta)]
            )
            right_target_dir = np.array(
                [np.sin(right_target_theta), np.cos(right_target_theta)]
            )
            left_target = left_contact - left_target_dir * push_distance
            right_target = right_contact - right_target_dir * push_distance

            return {
                "left": [left_approach, left_contact, left_target],
                "right": [right_approach, right_contact, right_target],
            }

        else:
            raise ValueError(f"Unknown push direction: {push_direction}")

    def get_rotation_waypoints(
        self, rotation_direction: str
    ) -> Dict[str, List[np.ndarray]]:
        """Generate waypoints for rotating the T."""
        rotation_angle = np.pi / 6  # 30 degrees
        approach_distance = 0.06

        if rotation_direction == "clockwise":
            angle = self.t_info.rotation
            # T pointing down
            if angle > -np.pi / 4 and angle < np.pi / 4:
                left_grip = self.key_points["top_bar_bottom_side_left"]
                right_grip = self.key_points["top_right"]
            # T pointing right
            elif angle > np.pi / 4 and angle < 3 * np.pi / 4:
                left_grip = self.key_points["top_left"]
                right_grip = self.key_points["stem_bottom_right"]
            # T pointing up
            elif (angle > 3 * np.pi / 4 and angle < np.pi) or (
                angle > -np.pi and angle < -3 * np.pi / 4
            ):
                left_grip = self.key_points["top_right"]
                right_grip = self.key_points["top_bar_bottom_side_left"]
            # T pointing left
            elif angle > -3 * np.pi / 4 and angle < -np.pi / 4:
                left_grip = self.key_points["stem_bottom_right"]
                right_grip = self.key_points["top_left"]
            else:
                raise ValueError(f"Unknown rotation direction: {rotation_direction}")

            # Approach points
            left_approach_theta = np.random.uniform(-np.pi / 6.0, np.pi / 6.0)
            right_approach_theta = np.random.uniform(-np.pi / 6.0, np.pi / 6.0)
            left_approach_dir = np.array(
                [np.sin(left_approach_theta), np.cos(left_approach_theta)]
            )
            right_approach_dir = np.array(
                [np.sin(right_approach_theta), np.cos(right_approach_theta)]
            )
            left_approach = left_grip - left_approach_dir * approach_distance
            right_approach = right_grip + right_approach_dir * approach_distance

            # Rotation around center
            center = (left_grip + right_grip) / 2

            # Calculate rotated positions
            left_rel = left_grip - center
            right_rel = right_grip - center

            cos_r, sin_r = np.cos(rotation_angle), np.sin(rotation_angle)
            R_rot = np.array([[cos_r, sin_r], [-sin_r, cos_r]])  # Clockwise rotation

            left_rotated = center + R_rot @ left_rel
            right_rotated = center + R_rot @ right_rel

            return {
                "left": [left_approach, left_grip, left_rotated],
                "right": [right_approach, right_grip, right_rotated],
            }

        elif rotation_direction == "counterclockwise":
            # Rotate T counterclockwise - flip right and left
            angle = self.t_info.rotation
            if angle > -np.pi / 4 and angle < np.pi / 4:
                left_grip = self.key_points["top_left"]
                right_grip = self.key_points["top_bar_bottom_side_right"]
            elif angle > np.pi / 4 and angle < 3 * np.pi / 4:
                left_grip = self.key_points["top_right"]
                right_grip = self.key_points["stem_bottom_left"]
            # pointing up
            elif (angle > 3 * np.pi / 4 and angle < np.pi) or (
                angle > -np.pi and angle < -3 * np.pi / 4
            ):
                left_grip = self.key_points["top_bar_bottom_side_right"]
                right_grip = self.key_points["top_left"]
            elif angle > -3 * np.pi / 4 and angle < -np.pi / 4:
                left_grip = self.key_points["stem_bottom_left"]
                right_grip = self.key_points["top_right"]
            else:
                raise ValueError(f"Unknown rotation direction: {rotation_direction}")

            # Approach points
            left_approach_theta = np.random.uniform(-np.pi / 6.0, np.pi / 6.0)
            right_approach_theta = np.random.uniform(-np.pi / 6.0, np.pi / 6.0)
            left_approach_dir = np.array(
                [np.sin(left_approach_theta), np.cos(left_approach_theta)]
            )
            right_approach_dir = np.array(
                [np.sin(right_approach_theta), np.cos(right_approach_theta)]
            )
            left_approach = left_grip + left_approach_dir * approach_distance
            right_approach = right_grip - right_approach_dir * approach_distance

            # Rotation around center
            center = (left_grip + right_grip) / 2
            left_rel = left_grip - center
            right_rel = right_grip - center

            cos_r, sin_r = np.cos(rotation_angle), np.sin(rotation_angle)
            R_rot = np.array(
                [[cos_r, -sin_r], [sin_r, cos_r]]
            )  # Counterclockwise rotation

            left_rotated = center + R_rot @ left_rel
            right_rotated = center + R_rot @ right_rel

            return {
                "left": [left_approach, left_grip, left_rotated],
                "right": [right_approach, right_grip, right_rotated],
            }

        else:
            raise ValueError(f"Unknown rotation direction: {rotation_direction}")


class CollisionChecker:
    """Checks for collisions between arms and with T-shape."""

    def __init__(
        self, t_analyzer: TGeometryAnalyzer, constraints: WorkspaceConstraints
    ):
        self.t_analyzer = t_analyzer
        self.constraints = constraints

    def check_arm_collision(self, left_pos: np.ndarray, right_pos: np.ndarray) -> bool:
        """Check if arms are too close to each other."""
        distance = np.linalg.norm(left_pos - right_pos)
        return distance < self.constraints.min_arm_distance

    def check_workspace_bounds(self, pos: np.ndarray) -> bool:
        """Check if position is within workspace bounds."""
        x, y = pos
        return (
            self.constraints.x_min <= x <= self.constraints.x_max
            and self.constraints.y_min <= y <= self.constraints.y_max
        )

    def check_t_collision(
        self, start_pos: np.ndarray, end_pos: np.ndarray, allow_contact: bool = False
    ) -> bool:
        """Check collision with T-shape for a line segment from start_pos to end_pos."""
        min_dist = 0.0 if allow_contact else self.constraints.min_t_distance

        # Get all contact points (edges of T-shape)
        center = self.t_analyzer.t_info.center

        # Sample points along the line segment
        num_samples = 20  # Number of points to check along the line
        t_values = np.linspace(0, 1, num_samples)
        line_points = start_pos[None, :] + t_values[:, None] * (
            end_pos[None, :] - start_pos[None, :]
        )

        # Check distance from each point on the line to all T-shape points
        for line_point in line_points:
            distances = np.linalg.norm(center - line_point)
            min_distance_to_t = distances
            if min_distance_to_t < min_dist:
                return True

        return False

    def is_contact_point_feasible(
        self, contact_point: np.ndarray, arm_positions: List[np.ndarray]
    ) -> bool:
        """Check if a contact point is reachable without collisions."""
        # Check workspace bounds
        if not self.check_workspace_bounds(contact_point):
            return False

        # Check if other arm positions would cause collision
        for other_pos in arm_positions:
            if self.check_arm_collision(contact_point, other_pos):
                return False

        return True


def actions_in_range(actions: np.ndarray) -> bool:
    """Check if actions are within range."""
    N = actions.shape[0]
    l_act = actions[:, :2]
    r_act = actions[:, 2:]
    world_bases = np.array(
        [
            [
                [1.0, 0.0, 0.0, -0.469],
                [0.0, 1.0, 0.0, -0.019],
                [0.0, 0.0, 1.0, 0.02],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [
                [-1.0, 0.0, 0.0, 0.469],
                [0.0, -1.0, 0.0, -0.019],
                [0.0, 0.0, 1.0, 0.02],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ]
    )
    l_poses = np.tile(np.eye(4), (N, 1, 1))
    r_poses = np.tile(np.eye(4), (N, 1, 1))
    l_poses[:, :2, 3] = l_act
    r_poses[:, :2, 3] = r_act
    robot_t_l_act = np.tile(np.linalg.inv(world_bases[0]), (N, 1, 1)) @ l_poses
    robot_t_r_act = np.tile(np.linalg.inv(world_bases[1]), (N, 1, 1)) @ r_poses
    x_max = 0.6 + 0.02
    x_min = 0.25 - 0.02
    y_max = 0.2 + 0.02
    y_min = -0.2 - 0.02
    in_range = (
        (robot_t_l_act[:, 0, 3].max() <= x_max)
        and (robot_t_l_act[:, 0, 3].min() >= x_min)
        and (robot_t_r_act[:, 0, 3].max() <= x_max)
        and (robot_t_r_act[:, 0, 3].min() >= x_min)
        and (robot_t_l_act[:, 1, 3].max() <= y_max)
        and (robot_t_l_act[:, 1, 3].min() >= y_min)
        and (robot_t_r_act[:, 1, 3].max() <= y_max)
        and (robot_t_r_act[:, 1, 3].min() >= y_min)
    )
    return in_range


def evaluate_vertical_up_success(
    initial_pose: np.ndarray, final_pose: np.ndarray, actions: np.ndarray
) -> bool:
    """Check if vertical upward push was successful."""
    initial_rot = initial_pose[:3, :3]
    initial_pos = initial_pose[:3, 3]
    final_pos = final_pose[:3, 3]
    final_rot = final_pose[:3, :3]

    # Check if T is lying flat on the table
    table_normal = np.array([0, 0, 1])
    final_z_axis = final_rot[:, 2]
    final_alignment = np.dot(final_z_axis, table_normal)
    is_final_flat = final_alignment > 0.95

    # Check if T is initially lying flat on the table
    initial_z_axis = initial_rot[:, 2]
    initial_alignment = np.dot(initial_z_axis, table_normal)
    is_initial_flat = initial_alignment > 0.95

    is_flat = is_final_flat and is_initial_flat

    # Check if y position increased
    y_movement = final_pos[1] - initial_pos[1]
    return is_flat and y_movement > 0.01 and actions_in_range(actions)


def evaluate_vertical_down_success(
    initial_pose: np.ndarray, final_pose: np.ndarray, actions: np.ndarray
) -> bool:
    """Check if vertical downward push was successful."""
    initial_rot = initial_pose[:3, :3]
    initial_pos = initial_pose[:3, 3]
    final_pos = final_pose[:3, 3]
    final_rot = final_pose[:3, :3]

    # Check if T is lying flat on the table
    table_normal = np.array([0, 0, 1])
    final_z_axis = final_rot[:, 2]
    final_alignment = np.dot(final_z_axis, table_normal)
    is_final_flat = final_alignment > 0.95

    # Check if T is initially lying flat on the table
    initial_z_axis = initial_rot[:, 2]
    initial_alignment = np.dot(initial_z_axis, table_normal)
    is_initial_flat = initial_alignment > 0.95

    is_flat = is_final_flat and is_initial_flat

    # Check if y position decreased
    y_movement = initial_pos[1] - final_pos[1]
    return is_flat and y_movement > 0.01 and actions_in_range(actions)


def evaluate_horizontal_right_success(
    initial_pose: np.ndarray, final_pose: np.ndarray, actions: np.ndarray
) -> bool:
    """Check if horizontal rightward push was successful."""
    initial_rot = initial_pose[:3, :3]
    initial_pos = initial_pose[:3, 3]
    final_pos = final_pose[:3, 3]
    final_rot = final_pose[:3, :3]

    # Check if T is lying flat on the table
    table_normal = np.array([0, 0, 1])
    final_z_axis = final_rot[:, 2]
    final_alignment = np.dot(final_z_axis, table_normal)
    is_final_flat = final_alignment > 0.95

    # Check if T is initially lying flat on the table
    initial_z_axis = initial_rot[:, 2]
    initial_alignment = np.dot(initial_z_axis, table_normal)
    is_initial_flat = initial_alignment > 0.95

    is_flat = is_final_flat and is_initial_flat

    # Check if x position increased
    x_movement = final_pos[0] - initial_pos[0]
    return is_flat and x_movement > 0.01 and actions_in_range(actions)


def evaluate_horizontal_left_success(
    initial_pose: np.ndarray, final_pose: np.ndarray, actions: np.ndarray
) -> bool:
    """Check if horizontal leftward push was successful."""
    initial_rot = initial_pose[:3, :3]
    initial_pos = initial_pose[:3, 3]
    final_pos = final_pose[:3, 3]
    final_rot = final_pose[:3, :3]

    # Check if T is lying flat on the table
    table_normal = np.array([0, 0, 1])
    final_z_axis = final_rot[:, 2]
    final_alignment = np.dot(final_z_axis, table_normal)
    is_final_flat = final_alignment > 0.95

    # Check if T is initially lying flat on the table
    initial_z_axis = initial_rot[:, 2]
    initial_alignment = np.dot(initial_z_axis, table_normal)
    is_initial_flat = initial_alignment > 0.95

    is_flat = is_final_flat and is_initial_flat

    # Check if x position decreased
    x_movement = initial_pos[0] - final_pos[0]
    return is_flat and x_movement > 0.01 and actions_in_range(actions)


def evaluate_rotation_cw_success(
    initial_pose: np.ndarray, final_pose: np.ndarray, actions: np.ndarray
) -> bool:
    """Check if rotation was successful."""
    initial_rot = initial_pose[:3, :3]
    final_rot = final_pose[:3, :3]

    # Check if T is lying flat on the table
    table_normal = np.array([0, 0, 1])
    final_z_axis = final_rot[:, 2]
    final_alignment = np.dot(final_z_axis, table_normal)
    is_final_flat = final_alignment > 0.95

    # Check if T is initially lying flat on the table
    initial_z_axis = initial_rot[:, 2]
    initial_alignment = np.dot(initial_z_axis, table_normal)
    is_initial_flat = initial_alignment > 0.95

    is_flat = is_final_flat and is_initial_flat

    # Check if there was significant rotation
    init_t_final_pose = np.linalg.inv(initial_pose) @ final_pose
    delta_angle = np.arctan2(init_t_final_pose[1, 0], init_t_final_pose[0, 0])
    return is_flat and delta_angle < -0.1 and actions_in_range(actions)


def evaluate_rotation_ccw_success(
    initial_pose: np.ndarray, final_pose: np.ndarray, actions: np.ndarray
) -> bool:
    """Check if rotation was successful."""
    initial_rot = initial_pose[:3, :3]
    final_rot = final_pose[:3, :3]

    # Check if T is lying flat on the table
    table_normal = np.array([0, 0, 1])
    final_z_axis = final_rot[:, 2]
    final_alignment = np.dot(final_z_axis, table_normal)
    is_final_flat = final_alignment > 0.95

    # Check if T is initially lying flat on the table
    initial_z_axis = initial_rot[:, 2]
    initial_alignment = np.dot(initial_z_axis, table_normal)
    is_initial_flat = initial_alignment > 0.95

    is_flat = is_final_flat and is_initial_flat

    # Check if there was significant rotation
    init_t_final_pose = np.linalg.inv(initial_pose) @ final_pose
    delta_angle = np.arctan2(init_t_final_pose[1, 0], init_t_final_pose[0, 0])
    return is_flat and delta_angle > 0.1 and actions_in_range(actions)


def evaluate_random_motion_success(
    initial_pose: np.ndarray, final_pose: np.ndarray, actions: np.ndarray
) -> bool:
    """Check if random motion was successful (T is still flat and not moved)."""
    initial_rot = initial_pose[:3, :3]
    final_rot = final_pose[:3, :3]

    # Check if T is lying flat on the table
    table_normal = np.array([0, 0, 1])
    final_z_axis = final_rot[:, 2]
    final_alignment = np.dot(final_z_axis, table_normal)
    is_final_flat = final_alignment > 0.95

    # Check if T is not moved
    final_pos = final_pose[:3, 3]
    initial_pos = initial_pose[:3, 3]

    # Check if T is initially lying flat on the table
    initial_z_axis = initial_rot[:, 2]
    initial_alignment = np.dot(initial_z_axis, table_normal)
    is_initial_flat = initial_alignment > 0.95

    is_flat = is_final_flat and is_initial_flat
    rotation_matrix = final_rot @ initial_rot.T
    trace = np.trace(rotation_matrix)
    angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
    is_moved = (angle > 0.1) or (np.linalg.norm(final_pos - initial_pos) > 0.01)
    return is_flat and not is_moved and actions_in_range(actions)


def evaluate_random_contact_success(
    initial_pose: np.ndarray, final_pose: np.ndarray, actions: np.ndarray
) -> bool:
    """Check if random motion was successful (T is still flat and moved)."""
    initial_rot = initial_pose[:3, :3]
    final_rot = final_pose[:3, :3]

    # Check if T is lying flat on the table
    table_normal = np.array([0, 0, 1])
    final_z_axis = final_rot[:, 2]
    final_alignment = np.dot(final_z_axis, table_normal)
    is_final_flat = final_alignment > 0.95

    # Check if T is not moved
    final_pos = final_pose[:3, 3]
    initial_pos = initial_pose[:3, 3]
    rotation_matrix = final_rot @ initial_rot.T

    # Check if T is initially lying flat on the table
    initial_z_axis = initial_rot[:, 2]
    initial_alignment = np.dot(initial_z_axis, table_normal)
    is_initial_flat = initial_alignment > 0.95

    is_flat = is_final_flat and is_initial_flat
    trace = np.trace(rotation_matrix)
    angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
    is_moved = (angle > 0.1) or (np.linalg.norm(final_pos - initial_pos) > 0.01)
    return is_flat and is_moved and actions_in_range(actions)


class MotionPlanner:
    """Plans 4 types of motions with specified ratios: linear(30%), rotating(30%), random contact(30%), random no-contact(10%)."""

    def __init__(self, workspace_constraints: WorkspaceConstraints = None):
        self.constraints = workspace_constraints or WorkspaceConstraints()
        self.config = TrajectoryConfig(
            noise_std=0.001,
        )
        self.eef_fingertip_offset = 0.08
        self.coordinator = BimanualCoordination(self.config)

        # Motion type probabilities
        self.motion_probabilities = {
            "linear": 0.3,
            "rotating": 0.3,
            "random_contact": 0.3,
            "random_no_contact": 0.1,
        }

    def plan_episode(
        self,
        t_info: TShapeInfo,
        current_arm_positions: np.ndarray,
        motion_type: str,
    ) -> tuple[np.ndarray, bool, callable, int]:
        """Plan a complete episode trajectory.

        Args:
            t_info: Information about T-shape
            current_arm_positions: Current positions (left_x, left_y, right_x, right_y)
            duration: Episode duration in seconds
            num_steps: Number of trajectory steps

        Returns:
            trajectory: Nx4 array of bimanual trajectory
        """
        # Initialize analyzers
        t_analyzer = TGeometryAnalyzer(t_info)
        collision_checker = CollisionChecker(t_analyzer, self.constraints)

        # Generate trajectory based on motion type
        if motion_type == "linear":
            duration = 15.0
            num_steps = 150
            traj, success, success_fn = self._plan_linear_motion(
                t_analyzer,
                collision_checker,
                current_arm_positions,
                duration,
                num_steps,
            )
        elif motion_type == "rotating":
            duration = 30.0
            num_steps = 300
            traj, success, success_fn = self._plan_rotating_motion(
                t_analyzer,
                collision_checker,
                current_arm_positions,
                duration,
                num_steps,
            )
        elif motion_type == "random_contact":
            duration = 30.0
            num_steps = 300
            traj, success, success_fn = self._plan_random_contact_motion(
                t_analyzer,
                collision_checker,
                current_arm_positions,
                duration,
                num_steps,
            )
        elif motion_type == "random_no_contact":
            duration = 15.0
            num_steps = 150
            traj, success, success_fn = self._plan_random_no_contact_motion(
                t_analyzer,
                collision_checker,
                current_arm_positions,
                duration,
                num_steps,
            )
        else:
            raise ValueError(f"Unknown motion type: {motion_type}")
        return traj, success, success_fn, num_steps

    def _select_motion_type(self) -> str:
        """Select motion type based on probabilities."""
        rand_val = random.random()
        cumulative_prob = 0.0

        for motion_type, prob in self.motion_probabilities.items():
            cumulative_prob += prob
            if rand_val <= cumulative_prob:
                return motion_type

        return "linear"  # fallback

    def _plan_linear_motion(
        self,
        t_analyzer: TGeometryAnalyzer,
        collision_checker: CollisionChecker,
        current_pos: np.ndarray,
        duration: float,
        num_steps: int,
    ) -> tuple[np.ndarray, bool, callable]:
        """Plan linear pushing motion."""
        # Randomly select push direction
        directions = [
            "horizontal_right",
            "horizontal_left",
            "vertical_up",
            "vertical_down",
        ]
        direction = random.choice(directions)

        waypoints = t_analyzer.get_linear_push_waypoints(direction)
        if direction == "horizontal_right":
            rand_noise = np.random.rand(3, 2) * 0.02
            for noise in rand_noise:
                waypoints["right"].append(current_pos[2:] + noise)
        elif direction == "horizontal_left":
            rand_noise = np.random.rand(3, 2) * 0.02
            for noise in rand_noise:
                waypoints["left"].append(current_pos[:2] + noise)

        for i, p in enumerate(waypoints["left"]):
            waypoints["left"][i][0] = p[0] - self.eef_fingertip_offset
        for i, p in enumerate(waypoints["right"]):
            waypoints["right"][i][0] = p[0] + self.eef_fingertip_offset

        waypoints["left"] = [current_pos[:2]] + waypoints["left"]
        waypoints["right"] = [current_pos[2:]] + waypoints["right"]

        # Create curve primitives from waypoints
        left_primitive = CurvePrimitive(waypoints["left"], self.config)
        right_primitive = CurvePrimitive(waypoints["right"], self.config)

        if direction == "vertical_up":
            success_fn = evaluate_vertical_up_success
        elif direction == "vertical_down":
            success_fn = evaluate_vertical_down_success
        elif direction == "horizontal_right":
            success_fn = evaluate_horizontal_right_success
        elif direction == "horizontal_left":
            success_fn = evaluate_horizontal_left_success

        # Generate coordinated trajectory
        return (
            self.coordinator.coordinate(
                left_primitive, right_primitive, duration, num_steps, "simultaneous"
            ),
            True,
            success_fn,
        )

    def _plan_rotating_motion(
        self,
        t_analyzer: TGeometryAnalyzer,
        collision_checker: CollisionChecker,
        current_pos: np.ndarray,
        duration: float,
        num_steps: int,
    ) -> tuple[np.ndarray, bool, callable]:
        """Plan rotating motion."""
        # Randomly select rotation direction
        direction = random.choice(["clockwise", "counterclockwise"])

        waypoints = t_analyzer.get_rotation_waypoints(direction)
        for i, p in enumerate(waypoints["left"]):
            waypoints["left"][i][0] = p[0] - self.eef_fingertip_offset
        for i, p in enumerate(waypoints["right"]):
            waypoints["right"][i][0] = p[0] + self.eef_fingertip_offset

        waypoints["left"] = [current_pos[:2]] + waypoints["left"]
        waypoints["right"] = [current_pos[2:]] + waypoints["right"]

        # Create curve primitives from waypoints
        left_primitive = CurvePrimitive(waypoints["left"], self.config)
        right_primitive = CurvePrimitive(waypoints["right"], self.config)

        if direction == "clockwise":
            success_fn = evaluate_rotation_cw_success
        elif direction == "counterclockwise":
            success_fn = evaluate_rotation_ccw_success
        else:
            raise ValueError(f"Unknown rotation direction: {direction}")

        # Generate coordinated trajectory
        return (
            self.coordinator.coordinate(
                left_primitive,
                right_primitive,
                duration,
                num_steps,
                "simultaneous",
                speed_profile="constant",
            ),
            True,
            success_fn,
        )

    def _plan_random_contact_motion(
        self,
        t_analyzer: TGeometryAnalyzer,
        collision_checker: CollisionChecker,
        current_pos: np.ndarray,
        duration: float,
        num_steps: int,
    ) -> tuple[np.ndarray, bool, callable]:
        """Plan random contact motion."""
        # Randomly select contact points
        left_contact = random.choice(t_analyzer.contact_points)
        right_contact = random.choice(t_analyzer.contact_points)

        # Generate approach and contact waypoints
        approach_distance = 0.03
        left_approach = left_contact - approach_distance * np.random.rand(2)
        right_approach = right_contact - approach_distance * np.random.rand(2)

        # Create small push motions from contact points
        push_distance = 0.03
        left_push = left_contact + push_distance * np.random.rand(2)
        right_push = right_contact + push_distance * np.random.rand(2)

        left_waypoints = [
            left_approach,
            left_contact,
            left_push,
        ]
        right_waypoints = [
            right_approach,
            right_contact,
            right_push,
        ]

        for i, p in enumerate(left_waypoints):
            left_waypoints[i][0] = p[0] - self.eef_fingertip_offset
        for i, p in enumerate(right_waypoints):
            right_waypoints[i][0] = p[0] + self.eef_fingertip_offset

        left_waypoints = [current_pos[:2]] + left_waypoints
        right_waypoints = [current_pos[2:]] + right_waypoints

        left_primitive = CurvePrimitive(left_waypoints, self.config)
        right_primitive = CurvePrimitive(right_waypoints, self.config)

        success_fn = evaluate_random_contact_success

        return (
            self.coordinator.coordinate(
                left_primitive,
                right_primitive,
                duration,
                num_steps,
                "overlap",
                speed_profile="constant",
            ),
            True,
            success_fn,
        )

    def _plan_random_no_contact_motion(
        self,
        t_analyzer: TGeometryAnalyzer,
        collision_checker: CollisionChecker,
        current_pos: np.ndarray,
        duration: float,
        num_steps: int,
    ) -> tuple[np.ndarray, bool, callable]:
        """Plan random motion with no T contact."""
        # Generate random target positions away from T
        max_attempts = 20

        for attempt in range(max_attempts):
            # Random positions in workspace
            left_target = np.array(
                [
                    random.uniform(self.constraints.x_min, self.constraints.x_max),
                    random.uniform(self.constraints.y_min, self.constraints.y_max),
                ]
            )
            right_target = np.array(
                [
                    random.uniform(self.constraints.x_min, self.constraints.x_max),
                    random.uniform(self.constraints.y_min, self.constraints.y_max),
                ]
            )

            # Check constraints
            if (
                collision_checker.check_workspace_bounds(left_target)
                and collision_checker.check_workspace_bounds(right_target)
                and not collision_checker.check_arm_collision(left_target, right_target)
                and not collision_checker.check_t_collision(
                    current_pos[:2], left_target, allow_contact=False
                )
                and not collision_checker.check_t_collision(
                    current_pos[2:], right_target, allow_contact=False
                )
            ):

                # Create linear primitives
                left_primitive = LinearPrimitive(
                    current_pos[:2], left_target, self.config
                )
                right_primitive = LinearPrimitive(
                    current_pos[2:], right_target, self.config
                )

                # Use sequential coordination for exploration-like behavior
                sync_type = "overlap"
                success_fn = evaluate_random_motion_success

                return (
                    self.coordinator.coordinate(
                        left_primitive,
                        right_primitive,
                        duration,
                        num_steps,
                        sync_type,
                        speed_profile="variable",
                    ),
                    True,
                    success_fn,
                )
        return np.zeros((num_steps, 4)), False, None
