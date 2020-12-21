import numpy as np
import multiprocessing
#from multiprocessing import shared_memory
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.managers import SharedMemoryManager
import matplotlib.pyplot as plt

def initialize_shared_ndarray_for_reading(shape):
    # make an array to store terrain 
    init_grid = np.random.normal(0, 1, shape)
    #print(init_grid)

    # create a section of shared memory of the same size as the grid array
    with SharedMemoryManager() as smm:
        #shm = shared_memory.SharedMemory(create=True, size=init_grid.nbytes)
        shm = smm.SharedMemory(size=init_grid.nbytes)

        # create another ndarray of the same shape & type as grid, backed by shared memory
        prev_grid = np.ndarray(init_grid.shape, dtype=init_grid.dtype, buffer=shm.buf)

        np.copyto(prev_grid, init_grid)
    
    print("shared array",shm.name,"has been initialized")
    return shm.name, init_grid.shape, init_grid.dtype

def read_from_shared(x,y,shm_name, shm_shape, shm_dtype):
    print("SWONEEE")
    # Locate the shared memory by its name
    shm = SharedMemory(shm_name)
    # Create the np.recarray from the buffer of the shared memory
    np_array = np.recarray(shape=shm_shape, dtype=shm_dtype, buf=shm.buf)
    return np.nansum(np_array.val)

    """
    print("swine",x,y,shared_ndarray_name)
    print("attempting to get reference to shared memory",shared_ndarray_name)
    shared_arr = shared_memory.SharedMemory(name=shared_ndarray_name)
    print('swane AA!!!')
    for i in range(-5,6):
        for j in range(-5,6):
            print("guzzle",end="")
            print(shared_arr[(x+i)%400][(y+j)%400])
    print('read from',x,y)
    """

def pool_read(name, shm_shape, shm_dtype):
    # Create a multiprocessing Pool
    pool = multiprocessing.Pool(2) 

    # read multiple times with specified args
    args = [(19,53,name, shm_shape, shm_dtype),(35,52,name, shm_shape, shm_dtype),(24,63,name, shm_shape, shm_dtype),(7,86,name, shm_shape, shm_dtype)]
    pool.starmap(read_from_shared, args)
    
    # parallelized portion is finished, close the pool
    # not sure if this is entirely necessary here
    print("closing pool...")
    #pool.close()
    #print("pool closed")
    pool.join()
    print("pool joined")

def main():
    # initialize a shared 400x400 ndarray called "prev_grid"
    name, shm_shape, shm_dtype = initialize_shared_ndarray_for_reading((400,400))
    print("initialized ndarray memory named",name)
    
    # read without pool
    #read_from_shared(51,25,name, shm_shape, shm_dtype
    # everyone get in the pool to read
    pool_read(name, shm_shape, shm_dtype)

if __name__== '__main__':
    main()