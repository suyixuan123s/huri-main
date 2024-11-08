# np_sharing.py
from multiprocessing import Process
from multiprocessing.managers import SharedMemoryManager
from multiprocessing.shared_memory import SharedMemory
from typing import Tuple

import numpy as np


def create_np_array_from_shared_mem(
    shared_mem: SharedMemory, shared_data_dtype: np.dtype, shared_data_shape: Tuple[int, ...]
) -> np.ndarray:
    arr = np.frombuffer(shared_mem.buf, dtype=shared_data_dtype)
    arr = arr.reshape(shared_data_shape)
    return arr


def child_process(
    shared_mem: SharedMemory, shared_data_dtype: np.dtype, shared_data_shape: Tuple[int, ...]
):
    """Logic to be executed by the child process"""
    arr = create_np_array_from_shared_mem(shared_mem, shared_data_dtype, shared_data_shape)
    arr[0, 0] = -arr[0, 0]  # modify the array backed by shared memory


def main():
    """Logic to be executed by the parent process"""

    # Data to be shared:
    data_to_share = np.random.rand(10, 10)

    SHARED_DATA_DTYPE = data_to_share.dtype
    SHARED_DATA_SHAPE = data_to_share.shape
    SHARED_DATA_NBYTES = data_to_share.nbytes

    with SharedMemoryManager() as smm:
        shared_mem = smm.SharedMemory(size=SHARED_DATA_NBYTES)

        arr = create_np_array_from_shared_mem(shared_mem, SHARED_DATA_DTYPE, SHARED_DATA_SHAPE)
        arr[:] = data_to_share  # load the data into shared memory

        print(f"The [0,0] element of arr is {arr[0,0]}")  # before

        # Run child process:
        p = Process(target=child_process, args=(shared_mem, SHARED_DATA_DTYPE, SHARED_DATA_SHAPE))
        p.start()
        p.join()

        print(f"The [0,0] element of arr is {arr[0,0]}")  # after

        del arr  # delete np array so the shared memory can be deallocated


if __name__ == "__main__":
    main()