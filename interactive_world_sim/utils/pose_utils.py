from enum import Enum

import numpy as np
import torch
import torch.nn.functional as F
import transforms3d


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))


class PoseType(Enum):
    """Enum class for different types of poses"""

    MAT: str = "mat"  # (4, 4) matrix
    POS_QUAT: str = "pos_quat"  # (7,) position and quaternion
    ROT_6D: str = "rot_6d"  # (9,) position and 6D rotation
    EULER: str = "euler"  # (6,) position and euler angles


def rot_6d_to_mat(rot_6d: np.ndarray) -> np.ndarray:
    """Convert 6D rotation representation to rotation matrix."""
    assert rot_6d.shape[1] == 9, f"Invalid rot_6d shape: {rot_6d.shape}"
    pos = rot_6d[:, :3]
    rot_6d = rot_6d[:, 3:]
    rot_mat = rotation_6d_to_matrix(torch.from_numpy(rot_6d))
    rot_mat = rot_mat.numpy()
    mat = np.zeros((rot_6d.shape[0], 4, 4))
    mat[:, :3, :3] = rot_mat
    mat[:, :3, 3] = pos
    mat[:, 3, 3] = 1
    return mat


def mat_to_rot_6d(mat: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to 6D rotation representation."""
    assert mat.shape[1:] == (4, 4), f"Invalid matrix shape: {mat.shape}"
    rot_mat = mat[:, :3, :3]
    rot_6d = matrix_to_rotation_6d(torch.from_numpy(rot_mat))
    pos = mat[:, :3, 3]
    return np.concatenate([pos, rot_6d.numpy()], axis=1)


def pos_quat_to_mat(pose_in_pos_quat: np.ndarray) -> np.ndarray:
    assert (
        pose_in_pos_quat.shape[1] == 7
    ), f"Invalid pose_in_pos_quat shape: {pose_in_pos_quat.shape}"
    pos = pose_in_pos_quat[:, :3]
    quat = pose_in_pos_quat[:, 3:]
    mat = np.zeros((pose_in_pos_quat.shape[0], 4, 4))
    for i in range(pose_in_pos_quat.shape[0]):
        mat[i, :3, :3] = transforms3d.quaternions.quat2mat(quat[i])
    mat[:, :3, 3] = pos
    mat[:, 3, 3] = 1
    return mat


def pose_convert(
    pose: np.ndarray, from_type: PoseType, to_type: PoseType, convention: str = "xyz"
) -> np.ndarray:
    """Convert pose from one type to another type

    Args:
        pose: input pose
        from_type: input pose type
        to_type: output pose type
        convention: euler angle convention
    Returns:
        output pose
    """
    if from_type == to_type:
        return pose

    if from_type == PoseType.ROT_6D and to_type == PoseType.MAT:
        return rot_6d_to_mat(pose)
    elif from_type == PoseType.MAT and to_type == PoseType.ROT_6D:
        return mat_to_rot_6d(pose)
    elif from_type == PoseType.POS_QUAT and to_type == PoseType.MAT:
        return pos_quat_to_mat(pose)
    else:
        raise NotImplementedError(
            f"Conversion from {from_type} to {to_type} is not implemented"
        )
