import functools
from typing import Union

import numpy as np
import pytorch3d.transforms as pt
import scipy.spatial.transform as st
import torch


# An ugly hack for euler convention
def torch_euler_to_matrix(x: torch.Tensor, convention) -> torch.Tensor:
    x_np = x.detach().cpu().numpy()
    mat = st.Rotation.from_euler(convention, x_np, degrees=False).as_matrix()
    return torch.from_numpy(mat).to(x.device, x.dtype)


def torch_matrix_to_euler(x: torch.Tensor, convention) -> torch.Tensor:
    x_np = x.detach().cpu().numpy()
    euler = st.Rotation.from_matrix(x_np).as_euler(convention, degrees=False)
    return torch.from_numpy(euler).to(x.device, x.dtype)


class RotationTransformer:
    """Transform between different rotation representations."""

    valid_reps = ["axis_angle", "euler_angles", "quaternion", "rotation_6d", "matrix"]

    def __init__(
        self,
        from_rep="axis_angle",
        to_rep="rotation_6d",
        from_convention=None,
        to_convention=None,
    ):
        """Valid representations

        Always use matrix as intermediate representation.
        """
        assert from_rep != to_rep
        assert from_rep in self.valid_reps
        assert to_rep in self.valid_reps
        if from_rep == "euler_angles":
            assert from_convention is not None
        if to_rep == "euler_angles":
            assert to_convention is not None

        forward_funcs = list()
        inverse_funcs = list()

        if from_rep != "matrix":
            # An ugly hack for euler convention
            if from_rep == "euler_angles":
                funcs = [
                    functools.partial(
                        torch_euler_to_matrix, convention=from_convention
                    ),
                    functools.partial(
                        torch_matrix_to_euler, convention=from_convention
                    ),
                ]
            else:
                funcs = [
                    getattr(pt, f"{from_rep}_to_matrix"),
                    getattr(pt, f"matrix_to_{from_rep}"),
                ]
                if from_convention is not None:
                    funcs = [
                        functools.partial(func, convention=from_convention)
                        for func in funcs
                    ]
            forward_funcs.append(funcs[0])
            inverse_funcs.append(funcs[1])

        if to_rep != "matrix":
            if to_rep == "euler_angles":
                funcs = [
                    functools.partial(torch_matrix_to_euler, convention=to_convention),
                    functools.partial(torch_euler_to_matrix, convention=to_convention),
                ]
            else:
                funcs = [
                    getattr(pt, f"matrix_to_{to_rep}"),
                    getattr(pt, f"{to_rep}_to_matrix"),
                ]
                if to_convention is not None:
                    funcs = [
                        functools.partial(func, convention=to_convention)
                        for func in funcs
                    ]
            forward_funcs.append(funcs[0])
            inverse_funcs.append(funcs[1])

        inverse_funcs = inverse_funcs[::-1]

        self.forward_funcs = forward_funcs
        self.inverse_funcs = inverse_funcs

    @staticmethod
    def _apply_funcs(
        x: Union[np.ndarray, torch.Tensor], funcs: list
    ) -> Union[np.ndarray, torch.Tensor]:
        x_ = x
        if isinstance(x, np.ndarray):
            x_ = torch.from_numpy(x)
        for func in funcs:
            x_ = func(x_)
        y = x_
        if isinstance(x, np.ndarray):
            y = x_.numpy()
        return y

    def forward(
        self, x: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """Transform x to the target representation."""
        return self._apply_funcs(x, self.forward_funcs)

    def inverse(
        self, x: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """Transform x from the target representation."""
        return self._apply_funcs(x, self.inverse_funcs)


def test():
    tf = RotationTransformer()

    rotvec = np.random.uniform(-2 * np.pi, 2 * np.pi, size=(1000, 3))
    rot6d = tf.forward(rotvec)
    new_rotvec = tf.inverse(rot6d)

    from scipy.spatial.transform import Rotation

    diff = Rotation.from_rotvec(rotvec) * Rotation.from_rotvec(new_rotvec).inv()
    dist = diff.magnitude()
    assert dist.max() < 1e-7

    tf = RotationTransformer("rotation_6d", "matrix")
    rot6d_wrong = rot6d + np.random.normal(scale=0.1, size=rot6d.shape)
    mat = tf.forward(rot6d_wrong)
    mat_det = np.linalg.det(mat)
    assert np.allclose(mat_det, 1)
    # rotaiton_6d will be normalized to rotation matrix


def test_euler():
    tf = RotationTransformer(
        "euler_angles", "matrix", from_convention="xyz", to_convention="xyz"
    )
    euler = np.random.uniform(-np.pi, np.pi, size=(1000, 3))
    mat = tf.forward(euler)

    import transforms3d

    new_mat = np.zeros_like(mat)
    for i in range(euler.shape[0]):
        new_mat[i] = transforms3d.euler.euler2mat(*euler[i], axes="sxyz")
    assert np.allclose(new_mat, mat, 1e-5)

    import pytorch3d

    tf = RotationTransformer("euler_angles", "rotation_6d", from_convention="xyz")
    euler = np.random.uniform(-np.pi, np.pi, size=(1000, 3))
    rot6d = tf.forward(euler)
    mat = pytorch3d.transforms.rotation_6d_to_matrix(torch.from_numpy(rot6d)).numpy()

    mat_from_euler = np.zeros_like(mat)
    for i in range(euler.shape[0]):
        mat_from_euler[i] = transforms3d.euler.euler2mat(*euler[i], axes="sxyz")
    assert np.allclose(mat_from_euler, mat, 1e-5)


if __name__ == "__main__":
    test_euler()
