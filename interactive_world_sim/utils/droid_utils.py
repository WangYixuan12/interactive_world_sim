import json
import os
from typing import Any, Dict, Optional, Tuple


def load_json_file(file_path: str) -> Dict[str, Any]:
    """Load a JSON file into memory.

    Args:
        file_path: Path to the JSON file.

    Returns:
        Dictionary containing the JSON data.
    """
    with open(file_path, "r") as f:
        return json.load(f)


def load_all_camera_data(
    root_dir: str,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Load all camera-related JSON files into memory.

    Returns:
        Tuple of (extrinsics_data, intrinsics_data, camera_serials_data)
    """
    extrinsics_json = "cam2base_extrinsics.json"
    super_extrinsics_json = "cam2base_extrinsic_superset.json"
    intrinsics_json = "intrinsics.json"
    camera_serials_json = "camera_serials.json"
    extrinsics_data = load_json_file(os.path.join(root_dir, extrinsics_json))
    super_extrinsics_data = load_json_file(
        os.path.join(root_dir, super_extrinsics_json)
    )
    extrinsics_data.update(super_extrinsics_data)
    intrinsics_data = load_json_file(os.path.join(root_dir, intrinsics_json))
    camera_serials_data = load_json_file(os.path.join(root_dir, camera_serials_json))

    return extrinsics_data, intrinsics_data, camera_serials_data


def find_episode_by_relative_path(
    relative_path: str, camera_serials_data: Dict[str, Any]
) -> Optional[str]:
    """Find episode key by matching relative_path.

    Args:
        relative_path: The relative path to search for.
        camera_serials_data: Loaded camera serials data.

    Returns:
        Episode key if found, None otherwise.
    """
    for episode_key, episode_data in camera_serials_data.items():
        if episode_data.get("relative_path") == relative_path:
            return episode_key
    return None


def build_relative_path_to_camera_info_map(
    extrinsics_data: Dict[str, Any],
    intrinsics_data: Dict[str, Any],
    camera_serials_data: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Build a dictionary mapping from relative_path to camera_info.

    Args:
        extrinsics_data: Pre-loaded extrinsics data.
        intrinsics_data: Pre-loaded intrinsics data.
        camera_serials_data: Pre-loaded camera serials data.

    Returns:
        Dictionary mapping relative_path to camera_info dictionary.
    """
    relative_path_to_camera_info = {}

    for episode_key, episode_data in extrinsics_data.items():
        relative_path = episode_data.get("relative_path")
        if relative_path is None:
            relative_path = episode_key

        # Extract camera serials from camera_serials_data if available
        camera_serials = {}
        if episode_key in camera_serials_data:
            for key, value in camera_serials_data[episode_key].items():
                if key.endswith("_cam_serial"):
                    camera_serials[key] = value

        # Extract extrinsics for this episode
        episode_extrinsics = {}
        for camera_serial, extrinsics in episode_data.items():
            if isinstance(extrinsics, list) and len(extrinsics) == 6:
                episode_extrinsics[camera_serial] = extrinsics

        # Extract intrinsics for this episode
        episode_intrinsics = {}
        if episode_key in intrinsics_data:
            for camera_serial, intrinsics in intrinsics_data[episode_key].items():
                if isinstance(intrinsics, dict):
                    episode_intrinsics[camera_serial] = intrinsics

        camera_info = {
            "episode_key": episode_key,
            "relative_path": relative_path,
            "camera_serials": camera_serials,
            "extrinsics": episode_extrinsics,
            "intrinsics": episode_intrinsics,
        }

        relative_path_to_camera_info[relative_path] = camera_info

    return relative_path_to_camera_info


# Example usage
if __name__ == "__main__":
    # Load all data once
    root_dir = "/home/yixuan/Downloads"
    extrinsics_data, intrinsics_data, camera_serials_data = load_all_camera_data(
        root_dir
    )

    # Build the mapping dictionary
    relative_path_to_camera_info = build_relative_path_to_camera_info_map(
        extrinsics_data, intrinsics_data, camera_serials_data
    )

    # Example: Get camera info for a specific relative_path
    example_relative_path = "AUTOLab/failure/2023-07-07/Fri_Jul__7_10:29:59_2023"
    camera_info = relative_path_to_camera_info[example_relative_path]
    print(f"Camera info for {example_relative_path}:")
    print(json.dumps(camera_info, indent=2))

    # Print some statistics
    print(f"\nTotal relative paths available: {len(relative_path_to_camera_info)}")
    print("First 5 relative paths:")
    for i, relative_path in enumerate(list(relative_path_to_camera_info.keys())[:5]):
        print(f"  {i+1}. {relative_path}")
