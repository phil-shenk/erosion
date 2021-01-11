import numpy as np
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.managers import SharedMemoryManager
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import current_process, cpu_count, Process, Pool
import random

import matplotlib.pyplot as plt

# work_func:
# get shm from name arg
# make arr w/ shm buffer
# do work
def work_function(x,y,shm_name, shm_shape, shm_dtype):
    print("doing work on",shm_name)
    return write_test(x,y,shm_name, shm_shape, shm_dtype)

def read_test(x,y,shm_name,shm_shape,shm_dtype):
    # Locate the shared memory by its name
    shm = SharedMemory(shm_name)
    # Create the np.ndarray from the buffer of the shared memory
    np_array = np.ndarray(shape=shm_shape, dtype=shm_dtype, buffer=shm.buf)
    result = np_array[x][y]
    print("RESULT:",result)
    return result

def write_test(x,y,shm_name,shm_shape,shm_dtype):
    """
    writes a random integer to the x,y cell of the shared numpy array
    haven't been able to find if writing from multiple processes to the same section of memory is handled,
    I have looked through the docs, I assume it's NOT safe, but i'm trying it nonetheless.
    """
    shm = SharedMemory(shm_name)
    np_array = np.ndarray(shape=shm_shape, dtype=shm_dtype, buffer=shm.buf)
    
    # can we write, does it do locks etc automatically, who knows:
    randn = np.random.randint(100)
    print('randn:',randn)
    np_array[x][y] = randn
    
    print("np_array[x][y]:",np_array[x][y])
    return randn


# main:
# init data
# with smm:
#  make smm.shm
#  make arr using shm buffer
#  copy init data into there
#  with ppe as exe:
#   exe.submit() the work_func w/ args 
#   for _ in as_completed: pass
def main():
    # initialize a shared 400x400 ndarray called "prev_grid"
    #name, shape, dtype = initialize_shared_ndarray_for_reading((400,400))
    #print("initialized ndarray memory named",name)

    # putting that whole func into main:
    # make an array to store terrain 
    shape = (400,400)
    #init_grid = np.random.normal(0, 1, shape)
    init_grid = np.zeros(shape)
    #print(init_grid)
    shape, dtype = init_grid.shape, init_grid.dtype
    print(shape)
    
    name = ''
    # create a section of shared memory of the same size as the grid array
    with SharedMemoryManager() as smm:
        #shm = shared_memory.SharedMemory(create=True, size=init_grid.nbytes)
        shm = smm.SharedMemory(size=init_grid.nbytes)
        name = shm.name
        
        # create another ndarray of the same shape & type as grid, backed by shared memory
        prev_grid = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

        # copy the initialized data into the shared area of memory
        np.copyto(prev_grid, init_grid)

    with ProcessPoolExecutor(cpu_count()) as exe:
        fs = [exe.submit(work_function, 1, 1, name, shape, dtype) for _ in range(cpu_count())]
        for future in as_completed(fs):
            print('as_completed yeeted,',fs)
            print('res:',future.result())
            pass
    
    print('\nprev_grid:')
    print(prev_grid)

if __name__== '__main__':
    main()