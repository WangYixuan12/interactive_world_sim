from typing import Optional, Union

import h5py
import numpy as np


def save_dict_to_hdf5(
    dic: dict, config_dict: dict, filename: str, attr_dict: Optional[dict] = None
) -> None:
    """...."""
    with h5py.File(filename, "w") as h5file:
        if attr_dict is not None:
            for key, item in attr_dict.items():
                h5file.attrs[key] = item
        recursively_save_dict_contents_to_group(h5file, "/", dic, config_dict)


def recursively_save_dict_contents_to_group(
    h5file: h5py.File, path: str, dic: dict, config_dict: dict
) -> None:
    """...."""
    for key, item in dic.items():
        if isinstance(item, np.ndarray):
            # h5file[path + key] = item
            if key not in config_dict:
                config_dict[key] = {}
            dset = h5file.create_dataset(
                path + key, shape=item.shape, **config_dict[key]
            )
            dset[...] = item
        elif isinstance(item, dict):
            if key not in config_dict:
                config_dict[key] = {}
            recursively_save_dict_contents_to_group(
                h5file, path + key + "/", item, config_dict[key]
            )
        else:
            raise ValueError("Cannot save %s type" % type(item))


def load_dict_from_hdf5(filename: str) -> Union[dict, h5py.File]:
    """...."""
    # with h5py.File(filename, 'r') as h5file:
    #     return recursively_load_dict_contents_from_group(h5file, '/')
    h5file = h5py.File(filename, "r")
    return recursively_load_dict_contents_from_group(h5file, "/"), h5file


def recursively_load_dict_contents_from_group(h5file: h5py.File, path: str) -> dict:
    """...."""
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py.Dataset):
            # ans[key] = np.array(item)
            ans[key] = item
        elif isinstance(item, h5py.Group):
            ans[key] = recursively_load_dict_contents_from_group(
                h5file, path + key + "/"
            )
    return ans
