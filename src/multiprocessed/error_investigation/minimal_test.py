from multiprocessing.shared_memory import SharedMemory
from multiprocessing.managers import SharedMemoryManager
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import current_process, cpu_count, Process, Pool
from datetime import datetime
import numpy as np
import pandas as pd
import time

import sys,traceback

# work_func:
# get shm from name arg
# make arr w/ shm buffer
# do work
def work_with_shared_memory(shm_name, shape, dtype):
    print(f'With SharedMemory: {current_process()=}')
    # Locate the shared memory by its name
    shm = SharedMemory(shm_name)
    # Create the np.ndarray from the buffer of the shared memory
    np_array = np.ndarray(shape=shape, dtype=dtype, buffer=shm.buf)
    
    print("reading data")
    result = np_array[50][50]
    print(result)
    return result

# main:
# init data
# with smm:
#  make smm.shm
#  make arr using shm buffer
#  copy init data into there
#  with ppe as exe:
#   exe.submit() the work_func w/ args 
#   for _ in as_completed: pass
if __name__ == "__main__":
    # initialize some data, an array to store terrain 
    np_array = np.random.normal(0, 1, (400,400))

    shape, dtype = np_array.shape, np_array.dtype
    print(f"np_array's size={np_array.nbytes/1e6}MB")

    # With shared memory
    # Start tracking memory usage
    start_time = time.time()
    with SharedMemoryManager() as smm:
        # Create a shared memory (using SharedMemoryManager) of size np_arry.nbytes
        shm = smm.SharedMemory(np_array.nbytes)
        # Create a np.ndarray using the buffer of shm
        shm_np_array = np.ndarray(shape=shape, dtype=dtype, buffer=shm.buf)
        # Copy the data into the shared memory
        np.copyto(shm_np_array, np_array)

        # do 1 work without any processing
        #work_with_shared_memory(shm.name, shape, dtype)
        
        #"""
        # Spawn some processes to do some work
        with ProcessPoolExecutor(cpu_count()) as exe:
            fs = [exe.submit(work_with_shared_memory, shm.name, shape, dtype) for _ in range(cpu_count())]
            for _ in as_completed(fs):
                print(fs)
                pass
        #"""

    print(f'Time elapsed: {time.time()-start_time:.2f}s')
