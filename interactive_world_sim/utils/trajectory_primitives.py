"""Trajectory primitives for human-like bimanual T-pushing data generation.
Each primitive generates position trajectories with realistic speed profiles and noise.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from scipy.interpolate import CubicSpline


@dataclass
class TrajectoryConfig:
    """Configuration for trajectory generation."""

    noise_std: float = 0.002  # 2mm standard deviation for position noise
    speed_variation: float = 0.3  # ±30% speed variation
    min_speed_factor: float = 0.3  # Minimum speed as fraction of max
    max_speed_factor: float = 1.5  # Maximum speed as fraction of max


class SingleArmPrimitive:
    """Base class for single arm trajectory primitives."""

    def __init__(self, config: TrajectoryConfig = None):
        self.config = config or TrajectoryConfig()

    def generate(
        self, duration: float, num_steps: int, speed_profile: str = "variable"
    ) -> np.ndarray:
        """Generate trajectory points for single arm (returns Nx2 array)."""
        raise NotImplementedError

    def _add_noise(self, trajectory: np.ndarray) -> np.ndarray:
        """Add realistic noise to trajectory."""
        noise = np.random.normal(0, self.config.noise_std, trajectory.shape)
        return trajectory + noise

    def _generate_speed_profile(
        self, num_steps: int, profile_type: str = "variable"
    ) -> np.ndarray:
        """Generate realistic human-like speed profiles."""
        t = np.linspace(0, 1, num_steps)

        if profile_type == "constant":
            speeds = np.ones(num_steps)
        elif profile_type == "bell":
            # Bell curve: slow start, fast middle, slow end
            speeds = np.exp(-(((t - 0.5) / 0.2) ** 2))
        elif profile_type == "variable":
            # Variable speed with random fluctuations
            min_speed = self.config.min_speed_factor
            max_speed = self.config.max_speed_factor
            base_speed = 0.5 + 0.3 * np.sin(2 * np.pi * t) * max_speed  # Sinusoidal
            fluctuations = 0.2 * np.random.randn(num_steps)  # Random fluctuations
            speeds = base_speed + fluctuations
            speeds = np.clip(speeds, min_speed, max_speed)
        else:
            raise ValueError(f"Unknown profile type: {profile_type}")

        return speeds

    def _resample_with_speed_profile(
        self, waypoints: np.ndarray, speed_profile: np.ndarray
    ) -> np.ndarray:
        """Resample waypoints according to speed profile."""
        # Calculate cumulative distances
        dists = np.sqrt(np.sum(np.diff(waypoints, axis=0) ** 2, axis=1))
        cum_dists = np.concatenate([[0], np.cumsum(dists)])

        # Create time points based on speed profile
        dt = 1.0 / len(speed_profile)
        time_points = np.cumsum(dt / speed_profile)
        time_points = time_points / time_points[-1]  # Normalize to [0,1]

        # Interpolate positions at new time points
        trajectory = np.zeros((len(speed_profile), 2))
        for dim in range(2):
            cs = CubicSpline(cum_dists / cum_dists[-1], waypoints[:, dim])
            trajectory[:, dim] = cs(time_points)

        return trajectory


class LinearPrimitive(SingleArmPrimitive):
    """Linear trajectory between start and end positions."""

    def __init__(
        self,
        start_pos: np.ndarray,
        end_pos: np.ndarray,
        config: TrajectoryConfig = None,
    ):
        super().__init__(config)
        self.start_pos = np.array(start_pos)
        self.end_pos = np.array(end_pos)

    def generate(
        self, duration: float, num_steps: int, speed_profile: str = "variable"
    ) -> np.ndarray:
        """Generate linear trajectory with realistic speed profile."""
        # Generate base linear trajectory
        waypoints = np.array([self.start_pos, self.end_pos])

        # Generate speed profile
        speeds = self._generate_speed_profile(num_steps, speed_profile)

        # Resample according to speed profile
        trajectory = self._resample_with_speed_profile(waypoints, speeds)

        # Add noise
        trajectory = self._add_noise(trajectory)

        return trajectory


class CurvePrimitive(SingleArmPrimitive):
    """Curved trajectory interpolated from multiple waypoints using cubic splines."""

    def __init__(self, waypoints: List[np.ndarray], config: TrajectoryConfig = None):
        super().__init__(config)
        self.waypoints = np.array(waypoints)
        assert len(self.waypoints) >= 2, "Need at least 2 waypoints for curve"

    def generate(
        self, duration: float, num_steps: int, speed_profile: str = "variable"
    ) -> np.ndarray:
        """Generate curved trajectory with realistic speed profile."""
        # Generate speed profile
        speeds = self._generate_speed_profile(num_steps, speed_profile)

        # Resample according to speed profile
        trajectory = self._resample_with_speed_profile(self.waypoints, speeds)

        # Add noise
        trajectory = self._add_noise(trajectory)

        return trajectory


class StabilizePrimitive(SingleArmPrimitive):
    """Keep arm in fixed position with small natural movements."""

    def __init__(self, position: np.ndarray, config: TrajectoryConfig = None):
        super().__init__(config)
        self.position = np.array(position)

    def generate(
        self, duration: float, num_steps: int, speed_profile: str = "variable"
    ) -> np.ndarray:
        """Generate stabilizing trajectory with small natural movements."""
        # Base trajectory is constant position
        trajectory = np.tile(self.position, (num_steps, 1))

        # Add small random movements (humans can't stay perfectly still)
        micro_movements = np.random.normal(
            0, self.config.noise_std * 2, trajectory.shape
        )

        # Apply low-pass filter to make movements smooth
        from scipy import signal

        b, a = signal.butter(2, 0.1)  # Low-pass filter
        for dim in range(2):
            micro_movements[:, dim] = signal.filtfilt(b, a, micro_movements[:, dim])

        trajectory += micro_movements

        return trajectory


class BimanualCoordination:
    """Coordinates two arm motions with realistic timing and synchronization."""

    def __init__(self, config: TrajectoryConfig = None):
        self.config = config or TrajectoryConfig()

    def coordinate(
        self,
        left_primitive: SingleArmPrimitive,
        right_primitive: SingleArmPrimitive,
        duration: float,
        num_steps: int,
        sync_type: str = "simultaneous",
        speed_profile: Optional[str] = None,
    ) -> np.ndarray:
        """Coordinate two arm primitives.

        Args:
            left_primitive: Trajectory primitive for left arm
            right_primitive: Trajectory primitive for right arm
            duration: Total duration in seconds
            num_steps: Number of trajectory steps
            sync_type: "simultaneous", "overlap"

        Returns:
            trajectory: Nx4 array (left_x, left_y, right_x, right_y)
        """
        if sync_type == "simultaneous":
            return self._simultaneous_coordination(
                left_primitive, right_primitive, duration, num_steps, speed_profile
            )
        elif sync_type == "overlap":
            return self._overlap_coordination(
                left_primitive,
                right_primitive,
                duration,
                num_steps,
                speed_profile=speed_profile,
            )
        else:
            raise ValueError(f"Unknown sync_type: {sync_type}")

    def _simultaneous_coordination(
        self,
        left_prim: SingleArmPrimitive,
        right_prim: SingleArmPrimitive,
        duration: float,
        num_steps: int,
        speed_profile: Optional[str] = None,
    ) -> np.ndarray:
        """Both arms move simultaneously."""
        if speed_profile is None:
            speed_profiles = ["constant", "bell", "variable"]
            speed_profile = np.random.choice(speed_profiles)
        left_traj = left_prim.generate(duration, num_steps, speed_profile)
        right_traj = right_prim.generate(duration, num_steps, speed_profile)

        # Add small synchronization imperfection
        sync_noise = np.random.normal(0, 0.001, (num_steps, 4))  # Small desync

        trajectory = np.column_stack([left_traj, right_traj]) + sync_noise
        return trajectory

    def _overlap_coordination(
        self,
        left_prim: SingleArmPrimitive,
        right_prim: SingleArmPrimitive,
        duration: float,
        num_steps: int,
        speed_profile: Optional[str] = None,
    ) -> np.ndarray:
        """Arms have overlapping but offset timing."""
        # Left arm starts earlier, right arm starts later with overlap
        left_start = 0
        right_start = int(num_steps * 0.3)
        if speed_profile is None:
            speed_profiles = ["constant", "bell", "variable"]
            speed_profile = np.random.choice(speed_profiles)

        left_traj = left_prim.generate(
            duration * 0.8, num_steps - left_start, speed_profile
        )
        right_traj = right_prim.generate(
            duration * 0.7, num_steps - right_start, speed_profile
        )
        left_end = left_start + len(left_traj)
        right_end = right_start + len(right_traj)

        trajectory = np.zeros((num_steps, 4))

        # Fill left arm trajectory
        trajectory[left_start:left_end, :2] = left_traj

        # Fill right arm trajectory
        trajectory[right_start:right_end, 2:] = right_traj

        # Fill missing parts with held positions
        trajectory[:left_start, :2] = left_traj[0]
        trajectory[:right_start, 2:] = right_traj[0]
        trajectory[left_end:, :2] = left_traj[-1]
        trajectory[right_end:, 2:] = right_traj[-1]

        return trajectory
