import numbers
from multiprocessing.managers import SharedMemoryManager
from queue import Empty, Full
from typing import Dict, List, Optional, Union

import numpy as np

from interactive_world_sim.utils.shared_memory_util import (
    ArraySpec,
    SharedAtomicCounter,
)
from interactive_world_sim.utils.shared_ndarray import SharedNDArray


class SharedMemoryQueue:
    """A Lock-Free FIFO Shared Memory Data Structure.

    Stores a sequence of dicts of numpy arrays.
    """

    def __init__(
        self,
        shm_manager: SharedMemoryManager,
        array_specs: List[ArraySpec],
        buffer_size: int,
    ) -> None:
        # create atomic counters
        write_counter: SharedAtomicCounter = SharedAtomicCounter(shm_manager)
        read_counter: SharedAtomicCounter = SharedAtomicCounter(shm_manager)

        # allocate shared memory arrays based on the provided specs
        shared_arrays: Dict[str, SharedNDArray] = dict()
        for spec in array_specs:
            key: str = spec.name
            assert key not in shared_arrays
            array: SharedNDArray = SharedNDArray.create_from_shape(
                mem_mgr=shm_manager,
                shape=(buffer_size, *spec.shape),
                dtype=spec.dtype,
            )
            shared_arrays[key] = array

        self.buffer_size: int = buffer_size
        self.array_specs: List[ArraySpec] = array_specs
        self.write_counter: SharedAtomicCounter = write_counter
        self.read_counter: SharedAtomicCounter = read_counter
        self.shared_arrays: Dict[str, SharedNDArray] = shared_arrays

    @classmethod
    def create_from_examples(
        cls,
        shm_manager: SharedMemoryManager,
        examples: Dict[str, Union[np.ndarray, numbers.Number]],
        buffer_size: int,
    ) -> "SharedMemoryQueue":
        """Create a SharedMemoryQueue from example data."""
        specs: List[ArraySpec] = []
        for key, value in examples.items():
            shape = None
            dtype = None
            if isinstance(value, np.ndarray):
                shape = value.shape
                dtype = value.dtype
                # Ensure the dtype is not 'object'
                assert dtype != np.dtype("O")
            elif isinstance(value, numbers.Number):
                shape = tuple()
                dtype = np.dtype(type(value))
            else:
                raise TypeError(f"Unsupported type {type(value)} for key '{key}'")

            spec: ArraySpec = ArraySpec(name=key, shape=shape, dtype=dtype)
            specs.append(spec)

        obj: SharedMemoryQueue = cls(
            shm_manager=shm_manager, array_specs=specs, buffer_size=buffer_size
        )
        return obj

    def qsize(self) -> int:
        """Return the current number of items in the queue."""
        read_count: int = self.read_counter.load()
        write_count: int = self.write_counter.load()
        n_data: int = write_count - read_count
        return n_data

    def empty(self) -> bool:
        """Return True if the queue is empty, else False."""
        n_data: int = self.qsize()
        return n_data <= 0

    def clear(self) -> None:
        """Clear the queue by setting the read counter equal to the write counter."""
        self.read_counter.store(self.write_counter.load())

    def put(self, data: Dict) -> None:
        """Insert an item into the queue."""
        read_count: int = self.read_counter.load()
        write_count: int = self.write_counter.load()
        n_data: int = write_count - read_count
        if n_data >= self.buffer_size:
            raise Full()

        next_idx: int = write_count % self.buffer_size

        # write to shared memory arrays
        for key, value in data.items():
            arr: np.ndarray = self.shared_arrays[key].get()
            if isinstance(value, np.ndarray):
                arr[next_idx] = value
            else:
                arr[next_idx] = np.array(value, dtype=arr.dtype)

        # update counter
        self.write_counter.add(1)

    def get(self, out: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, np.ndarray]:
        """Retrieve one item from the queue."""
        write_count: int = self.write_counter.load()
        read_count: int = self.read_counter.load()
        n_data: int = write_count - read_count
        if n_data <= 0:
            raise Empty()

        if out is None:
            out = self._allocate_empty()

        next_idx: int = read_count % self.buffer_size
        for key, shared_obj in self.shared_arrays.items():
            arr: np.ndarray = shared_obj.get()
            np.copyto(out[key], arr[next_idx])
        # update counter
        self.read_counter.add(1)
        return out

    def get_k(
        self, k: int, out: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, np.ndarray]:
        """Retrieve k items from the queue."""
        write_count: int = self.write_counter.load()
        read_count: int = self.read_counter.load()
        n_data: int = write_count - read_count
        if n_data <= 0:
            raise Empty()
        assert k <= n_data

        out = self._get_k_impl(k, read_count, out=out)
        self.read_counter.add(k)
        return out

    def get_all(
        self, out: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, np.ndarray]:
        """Retrieve all available items from the queue."""
        write_count: int = self.write_counter.load()
        read_count: int = self.read_counter.load()
        n_data: int = write_count - read_count
        if n_data <= 0:
            raise Empty()

        out = self._get_k_impl(n_data, read_count, out=out)
        self.read_counter.add(n_data)
        return out

    def _get_k_impl(
        self, k: int, read_count: int, out: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, np.ndarray]:
        """Helper method: retrieve k consecutive items from shared memory."""
        if out is None:
            out = self._allocate_empty(k)

        curr_idx: int = read_count % self.buffer_size
        for key, shared_obj in self.shared_arrays.items():
            arr: np.ndarray = shared_obj.get()
            target: np.ndarray = out[key]
            start: int = curr_idx
            end: int = min(start + k, self.buffer_size)
            target_start: int = 0
            target_end: int = end - start
            target[target_start:target_end] = arr[start:end]

            remainder: int = k - (end - start)
            if remainder > 0:
                # Wrap-around: copy the beginning part of the buffer.
                start = 0
                end = start + remainder
                target_start = target_end
                target_end = k
                target[target_start:target_end] = arr[start:end]
        return out

    def _allocate_empty(self, k: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Allocate an output dictionary of empty NumPy arrays."""
        result: Dict[str, np.ndarray] = dict()
        for spec in self.array_specs:
            shape = spec.shape
            if k is not None:
                shape = (k, *shape)
            result[spec.name] = np.empty(shape=shape, dtype=spec.dtype)
        return result
