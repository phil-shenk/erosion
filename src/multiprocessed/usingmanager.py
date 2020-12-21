from multiprocessing.shared_memory import SharedMemory
from multiprocessing.managers import SharedMemoryManager
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import current_process, cpu_count, Process, Pool
from datetime import datetime
import numpy as np
import pandas as pd
import tracemalloc
import time



def work_with_shared_memory(shm_name, shape, dtype):
    print(f'With SharedMemory: {current_process()=}')
    # Locate the shared memory by its name
    shm = SharedMemory(shm_name)
    # create array with buffer of shared memory
    np_array = np.ndarray(shape=shape, dtype=dtype, buffer=shm.buf)
    print("reading data")
    result = np_array[50][50]
    print(result)
    return result



if __name__ == "__main__":
    # Make a large data frame with date, float and character columns
    a = [
        (datetime.today(), 1, 'string'),
        (datetime.today(), np.nan, 'abc'),
    ] * 5000000
    df = pd.DataFrame(a, columns=['date', 'val', 'character_col'])

    # Convert into numpy recarray to preserve the dtypes
    np_array = df.to_records(index=False)
    del df
    shape, dtype = np_array.shape, np_array.dtype
    print(f"np_array's size={np_array.nbytes/1e6}MB")

    # With shared memory
    # Start tracking memory usage
    tracemalloc.start()
    start_time = time.time()
    with SharedMemoryManager() as smm:
        # Create a shared memory (using SharedMemoryManager) of size np_arry.nbytes
        shm = smm.SharedMemory(np_array.nbytes)
        # Create a np.recarray using the buffer of shm
        shm_np_array = np.ndarray(shape=shape, dtype=dtype, buffer=shm.buf)
        # Copy the data into the shared memory
        np.copyto(shm_np_array, np_array)
        # Spawn some processes to do some work
        with ProcessPoolExecutor(cpu_count()) as exe:
            fs = [exe.submit(work_with_shared_memory, shm.name, shape, dtype) for _ in range(cpu_count())]
            for _ in as_completed(fs):
                print(fs)
                pass
            
    # Check memory usage
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage {current/1e6}MB; Peak: {peak/1e6}MB")
    print(f'Time elapsed: {time.time()-start_time:.2f}s')
    tracemalloc.stop()
