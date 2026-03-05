from dataclasses import dataclass
from multiprocessing.managers import SharedMemoryManager
from typing import Tuple

import numpy as np
from atomics import UINT, MemoryOrder, atomicview


@dataclass
class ArraySpec:
    """Specification of an array"""

    name: str
    shape: Tuple[int]
    dtype: np.dtype


class SharedAtomicCounter:
    """Shared atomic counter"""

    def __init__(self, shm_manager: SharedMemoryManager, size: int = 8):  # 64bit int
        shm = shm_manager.SharedMemory(size=size)
        self.shm = shm
        self.size = size
        self.store(0)  # initialize

    @property
    def buf(self) -> memoryview:
        """Get the memoryview of the shared memory"""
        return self.shm.buf[: self.size]

    def load(self) -> int:
        """Load the value of the counter"""
        with atomicview(buffer=self.buf, atype=UINT) as a:
            value = a.load(order=MemoryOrder.ACQUIRE)
        return value

    def store(self, value: int) -> None:
        """Store a value to the counter"""
        with atomicview(buffer=self.buf, atype=UINT) as a:
            a.store(value, order=MemoryOrder.RELEASE)

    def add(self, value: int) -> None:
        """Add a value to the counter"""
        with atomicview(buffer=self.buf, atype=UINT) as a:
            a.add(value, order=MemoryOrder.ACQ_REL)
