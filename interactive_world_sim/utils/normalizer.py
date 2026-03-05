from typing import Any, Callable, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import zarr

from interactive_world_sim.utils.dict_of_tensor_mixin import DictOfTensorMixin

from .pytorch_util import dict_apply


class LinearNormalizer(DictOfTensorMixin):
    """Linear Normalizer."""

    avaliable_modes = ["limits", "gaussian"]

    @torch.no_grad()
    def fit(
        self,
        data: Union[Dict, torch.Tensor, np.ndarray, zarr.Array],
        last_n_dims: int = 1,
        dtype: torch.dtype = torch.float32,
        mode: str = "limits",
        output_max: float = 1.0,
        output_min: float = -1.0,
        range_eps: float = 1e-4,
        fit_offset: bool = True,
    ) -> None:
        """Fit the normalizer."""
        if isinstance(data, dict):
            for key, value in data.items():
                self.params_dict[key] = _fit(
                    value,
                    last_n_dims=last_n_dims,
                    dtype=dtype,
                    mode=mode,
                    output_max=output_max,
                    output_min=output_min,
                    range_eps=range_eps,
                    fit_offset=fit_offset,
                )
        else:
            self.params_dict["_default"] = _fit(
                data,
                last_n_dims=last_n_dims,
                dtype=dtype,
                mode=mode,
                output_max=output_max,
                output_min=output_min,
                range_eps=range_eps,
                fit_offset=fit_offset,
            )

    def __call__(self, x: Union[Dict, torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Normalize the input."""
        return self.normalize(x)

    def __getitem__(self, key: str) -> "SingleFieldLinearNormalizer":
        return SingleFieldLinearNormalizer(self.params_dict[key])

    def __setitem__(self, key: str, value: "SingleFieldLinearNormalizer") -> None:
        self.params_dict[key] = value.params_dict

    def _normalize_impl(
        self, x: Union[Dict, torch.Tensor, np.ndarray], forward: bool = True
    ) -> torch.Tensor:
        if isinstance(x, dict):
            result = dict()
            for key, value in x.items():
                params = self.params_dict[key]
                result[key] = _normalize(value, params, forward=forward)
            return result
        else:
            if "_default" not in self.params_dict:
                raise RuntimeError("Not initialized")
            params = self.params_dict["_default"]
            return _normalize(x, params, forward=forward)

    def normalize(self, x: Union[Dict, torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Normalize the input."""
        return self._normalize_impl(x, forward=True)

    def unnormalize(self, x: Union[Dict, torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Unnormalize the input."""
        return self._normalize_impl(x, forward=False)

    def get_input_stats(self) -> Dict:
        """Return the input stats."""
        if len(self.params_dict) == 0:
            raise RuntimeError("Not initialized")
        if len(self.params_dict) == 1 and "_default" in self.params_dict:
            return self.params_dict["_default"]["input_stats"]

        result = dict()
        for key, value in self.params_dict.items():
            if key != "_default":
                result[key] = value["input_stats"]
        return result

    def get_output_stats(self, key: str = "_default") -> Dict:
        """Return the output stats."""
        input_stats = self.get_input_stats()
        if "min" in input_stats:
            # no dict
            return dict_apply(input_stats, self.normalize)

        result = dict()
        for key, group in input_stats.items():
            this_dict = dict()
            for name, value in group.items():
                this_dict[name] = self.normalize({key: value})[key]
            result[key] = this_dict
        return result


class SingleFieldLinearNormalizer(DictOfTensorMixin):
    """Single Field Linear Normalizer."""

    avaliable_modes = ["limits", "gaussian"]

    @torch.no_grad()
    def fit(
        self,
        data: Union[torch.Tensor, np.ndarray, zarr.Array],
        last_n_dims: int = 1,
        dtype: torch.dtype = torch.float32,
        mode: str = "limits",
        output_max: float = 1.0,
        output_min: float = -1.0,
        range_eps: float = 1e-4,
        fit_offset: bool = True,
    ) -> None:
        """Fit the normalizer."""
        self.params_dict = _fit(
            data,
            last_n_dims=last_n_dims,
            dtype=dtype,
            mode=mode,
            output_max=output_max,
            output_min=output_min,
            range_eps=range_eps,
            fit_offset=fit_offset,
        )

    @classmethod
    def create_fit(
        cls: Callable,
        data: Union[torch.Tensor, np.ndarray, zarr.Array],
        **kwargs: Dict[str, Any],
    ) -> "SingleFieldLinearNormalizer":
        """Create a normalizer by fitting the data."""
        obj = cls()
        obj.fit(data, **kwargs)
        return obj

    @classmethod
    def create_manual(
        cls: Callable,
        scale: Union[torch.Tensor, np.ndarray],
        offset: Union[torch.Tensor, np.ndarray],
        input_stats_dict: Dict[str, Union[torch.Tensor, np.ndarray]],
    ) -> "SingleFieldLinearNormalizer":
        """Create a normalizer manually."""

        def to_tensor(x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
            if not isinstance(x, torch.Tensor):
                x = torch.from_numpy(x)
            x = x.flatten()
            return x

        # check
        for x in [offset, *input_stats_dict.values()]:
            assert x.shape == scale.shape
            assert x.dtype == scale.dtype

        params_dict = nn.ParameterDict(
            {
                "scale": to_tensor(scale),
                "offset": to_tensor(offset),
                "input_stats": nn.ParameterDict(
                    dict_apply(input_stats_dict, to_tensor)
                ),
            }
        )
        return cls(params_dict)

    @classmethod
    def create_identity(
        cls: Callable, dtype: torch.dtype = torch.float32
    ) -> "SingleFieldLinearNormalizer":
        """Create an identity normalizer."""
        scale = torch.tensor([1], dtype=dtype)
        offset = torch.tensor([0], dtype=dtype)
        input_stats_dict = {
            "min": torch.tensor([-1], dtype=dtype),
            "max": torch.tensor([1], dtype=dtype),
            "mean": torch.tensor([0], dtype=dtype),
            "std": torch.tensor([1], dtype=dtype),
        }
        return cls.create_manual(scale, offset, input_stats_dict)  # type: ignore

    def normalize(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Normalize the input."""
        return _normalize(x, self.params_dict, forward=True)

    def unnormalize(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Unnormalize the input."""
        return _normalize(x, self.params_dict, forward=False)

    def get_input_stats(self) -> Dict:
        """Return the input stats."""
        return self.params_dict["input_stats"]

    def get_output_stats(self) -> Dict:
        """Return the output stats."""
        return dict_apply(self.params_dict["input_stats"], self.normalize)

    def __call__(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Normalize the input."""
        return self.normalize(x)


def _fit(
    data: Union[torch.Tensor, np.ndarray, zarr.Array],
    last_n_dims: int = 1,
    dtype: torch = torch.float32,
    mode: str = "limits",
    output_max: float = 1.0,
    output_min: float = -1.0,
    range_eps: float = 1e-4,
    fit_offset: bool = True,
) -> nn.ParameterDict:
    assert mode in ["limits", "gaussian"]
    assert last_n_dims >= 0
    assert output_max > output_min

    # convert data to torch and type
    if isinstance(data, zarr.Array):
        data = data[:]
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    if dtype is not None:
        data = data.type(dtype)

    # convert shape
    dim = 1
    if last_n_dims > 0:
        dim = np.prod(data.shape[-last_n_dims:])
    data = data.reshape(-1, dim)

    # compute input stats min max mean std
    input_min, _ = data.min(axis=0)
    input_max, _ = data.max(axis=0)
    input_mean = data.mean(axis=0)
    input_std = data.std(axis=0)

    # compute scale and offset
    if mode == "limits":
        if fit_offset:
            # unit scale
            input_range = input_max - input_min
            ignore_dim = input_range < range_eps
            input_range[ignore_dim] = output_max - output_min
            scale = (output_max - output_min) / input_range
            offset = output_min - scale * input_min
            offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]
            # ignore dims scaled to mean of output max and min
        else:
            # use this when data is pre-zero-centered.
            assert output_max > 0
            assert output_min < 0
            # unit abs
            output_abs = min(abs(output_min), abs(output_max))
            input_abs = torch.maximum(torch.abs(input_min), torch.abs(input_max))
            ignore_dim = input_abs < range_eps
            input_abs[ignore_dim] = output_abs
            # don't scale constant channels
            scale = output_abs / input_abs
            offset = torch.zeros_like(input_mean)
    elif mode == "gaussian":
        ignore_dim = input_std < range_eps
        scale = input_std.clone()
        scale[ignore_dim] = 1
        scale = 1 / scale

        if fit_offset:
            offset = -input_mean * scale
        else:
            offset = torch.zeros_like(input_mean)

    # save
    this_params = nn.ParameterDict(
        {
            "scale": scale,
            "offset": offset,
            "input_stats": nn.ParameterDict(
                {
                    "min": input_min,
                    "max": input_max,
                    "mean": input_mean,
                    "std": input_std,
                }
            ),
        }
    )
    for p in this_params.parameters():
        p.requires_grad_(False)
    return this_params


def _normalize(
    x: Union[np.ndarray, torch.Tensor],
    params: dict[str, torch.Tensor],
    forward: bool = True,
) -> torch.Tensor:
    assert "scale" in params
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    scale = params["scale"]
    offset = params["offset"]
    x = x.to(device=scale.device, dtype=scale.dtype)
    src_shape = x.shape
    x = x.reshape(-1, scale.shape[0])
    if forward:
        x = x * scale + offset
    else:
        x = (x - offset) / scale
    x = x.reshape(src_shape)
    return x


def test() -> None:
    data = torch.zeros((100, 10, 9, 2)).uniform_()
    data[..., 0, 0] = 0

    normalizer = SingleFieldLinearNormalizer()
    normalizer.fit(data, mode="limits", last_n_dims=2)
    datan = normalizer.normalize(data)
    assert datan.shape == data.shape
    assert np.allclose(datan.max(), 1.0)
    assert np.allclose(datan.min(), -1.0)
    dataun = normalizer.unnormalize(datan)
    assert torch.allclose(data, dataun, atol=1e-7)

    normalizer = SingleFieldLinearNormalizer()
    normalizer.fit(data, mode="limits", last_n_dims=1, fit_offset=False)
    datan = normalizer.normalize(data)
    assert datan.shape == data.shape
    assert np.allclose(datan.max(), 1.0, atol=1e-3)
    assert np.allclose(datan.min(), 0.0, atol=1e-3)
    dataun = normalizer.unnormalize(datan)
    assert torch.allclose(data, dataun, atol=1e-7)

    data = torch.zeros((100, 10, 9, 2)).uniform_()
    normalizer = SingleFieldLinearNormalizer()
    normalizer.fit(data, mode="gaussian", last_n_dims=0)
    datan = normalizer.normalize(data)
    assert datan.shape == data.shape
    assert np.allclose(datan.mean(), 0.0, atol=1e-3)
    assert np.allclose(datan.std(), 1.0, atol=1e-3)
    dataun = normalizer.unnormalize(datan)
    assert torch.allclose(data, dataun, atol=1e-7)

    # dict
    data = torch.zeros((100, 10, 9, 2)).uniform_()
    data[..., 0, 0] = 0

    normalizer = LinearNormalizer()
    normalizer.fit(data, mode="limits", last_n_dims=2)
    datan = normalizer.normalize(data)
    assert datan.shape == data.shape
    assert np.allclose(datan.max(), 1.0)
    assert np.allclose(datan.min(), -1.0)
    dataun = normalizer.unnormalize(datan)
    assert torch.allclose(data, dataun, atol=1e-7)

    data = {
        "obs": torch.zeros((1000, 128, 9, 2)).uniform_() * 512,
        "action": torch.zeros((1000, 128, 2)).uniform_() * 512,
    }
    normalizer = LinearNormalizer()
    normalizer.fit(data)
    datan = normalizer.normalize(data)
    dataun = normalizer.unnormalize(datan)
    for key in data:
        assert torch.allclose(data[key], dataun[key], atol=1e-4)

    state_dict = normalizer.state_dict()
    n = LinearNormalizer()
    n.load_state_dict(state_dict)
    datan = n.normalize(data)
    dataun = n.unnormalize(datan)
    for key in data:
        assert torch.allclose(data[key], dataun[key], atol=1e-4)


def get_range_normalizer_from_stat(
    stat: Dict[str, np.ndarray],
    output_max: float = 1,
    output_min: float = -1,
    range_eps: float = 1e-7,
    ignore_dim: Optional[np.ndarray] = None,
) -> SingleFieldLinearNormalizer:
    # -1, 1 normalization
    input_max = stat["max"]
    input_min = stat["min"]
    input_range = input_max - input_min
    if ignore_dim is None:
        ignore_dim = input_range < range_eps
    input_range[ignore_dim] = output_max - output_min
    scale = (output_max - output_min) / input_range
    offset = output_min - scale * input_min
    offset[ignore_dim] = 0.0

    return SingleFieldLinearNormalizer.create_manual(
        scale=scale, offset=offset, input_stats_dict=stat
    )


def get_image_range_normalizer() -> SingleFieldLinearNormalizer:
    scale = np.array([2], dtype=np.float32)
    offset = np.array([-1], dtype=np.float32)
    stat = {
        "min": np.array([0], dtype=np.float32),
        "max": np.array([1], dtype=np.float32),
        "mean": np.array([0.5], dtype=np.float32),
        "std": np.array([np.sqrt(1 / 12)], dtype=np.float32),
    }
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale, offset=offset, input_stats_dict=stat
    )


def get_depth_range_normalizer() -> SingleFieldLinearNormalizer:
    scale = np.array([2], dtype=np.float32)
    offset = np.array([-1], dtype=np.float32)
    stat = {
        "min": np.array([0], dtype=np.float32),
        "max": np.array([1], dtype=np.float32),
        "mean": np.array([0.5], dtype=np.float32),
        "std": np.array([np.sqrt(1 / 12)], dtype=np.float32),
    }
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale, offset=offset, input_stats_dict=stat
    )


def get_identity_normalizer_from_stat(
    stat: Dict[str, np.ndarray]
) -> SingleFieldLinearNormalizer:
    scale = np.ones_like(stat["min"])
    offset = np.zeros_like(stat["min"])
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale, offset=offset, input_stats_dict=stat
    )


def get_hundred_times_normalizer_from_stat(
    stat: Dict[str, np.ndarray]
) -> SingleFieldLinearNormalizer:
    scale = np.ones_like(stat["min"]) * 100
    offset = np.zeros_like(stat["min"])
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale, offset=offset, input_stats_dict=stat
    )


def get_twenty_times_normalizer_from_stat(
    stat: Dict[str, np.ndarray]
) -> SingleFieldLinearNormalizer:
    scale = np.ones_like(stat["min"]) * 20
    offset = np.zeros_like(stat["min"])
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale, offset=offset, input_stats_dict=stat
    )


def array_to_stats(arr: np.ndarray) -> Dict[str, np.ndarray]:
    stat = {
        "min": np.min(arr, axis=0),
        "max": np.max(arr, axis=0),
        "mean": np.mean(arr, axis=0),
        "std": np.std(arr, axis=0),
    }
    return stat
