import numpy as np
import matplotlib.pyplot as plt
import voronoi as v
import erosion_functions as er
import cv2
import time

from multiprocessing.shared_memory import SharedMemory
from multiprocessing.managers import SharedMemoryManager
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import current_process, cpu_count, Process, Pool, Queue

from mayavi import mlab

#TODO: make mavavi optional with cmd line flag
#from mayavi import mlab

def work_function(shm_name, shm_shape, shm_dtype, numDropsPerProcess):
    # locate shared memory block by name
    shm = SharedMemory(shm_name)
    # Create the np.ndarray from the buffer of the shared memory
    np_array = np.ndarray(shape=shm_shape, dtype=shm_dtype, buffer=shm.buf)
    
    for drop in range(numDropsPerProcess):
        er.simulateDropInplace(np_array)
        if(drop%500) == 0:
            print('drop',drop)
    return 'jazz'

# doubt this will work..
@mlab.animate(delay=10)
def anim(s, shared_array, msg_queue):
    print('test ing t e s t i n g')
    print('entering loop:')
    calculating = True
    while calculating:
        # still calculating if queue remains empty
        calculating = msg_queue.empty()
        s.mlab_source.scalars = shared_array
        yield
        
def plot_function(name, shape, dtype, msg_queue):
    print("\n\n\n\nPLOT FUNCTION PLOT FUNCTION PLOT FUNCTION\n\n\n\n")
    # locate shared memory block by name
    shm = SharedMemory(name)
    # Create the np.ndarray from the buffer of the shared memory
    shared_array = np.ndarray(shape=shape, dtype=dtype, buffer=shm.buf)

    s = mlab.surf(shared_array, color=(1,1,1), warp_scale=50.0)# warp_scale="auto" or 25.0 or 50.0 or whatever, colormap='copper',

    print('starting animation..')
    anim(s, shared_array, msg_queue)
    mlab.show()
    print('done w/ anim()')

def main():
    width  = 500
    length = 500

    showdrops = False

    xs = np.linspace(-np.pi/2,np.pi/2,width)
    ys = np.linspace(-np.pi/2,np.pi/2,length)
    xx, yy = np.meshgrid(xs,ys)
    #grid = xx**2 + yy**2
    mapsrc = "voronoi"
    mapsrc = "heightmap2.png"
    if mapsrc == "voronoi":
        grid = -1.0*v.calculateVoronoi(width,length,5,5)
        mapsrc = "voronoi.png"
    else:
        grid = cv2.imread("../../heightmaps/"+mapsrc, cv2.IMREAD_GRAYSCALE)
        grid = grid * (1.0/256.0)

    shape, dtype = grid.shape, grid.dtype
    print(shape)

    # set up messaging queue
    msg_queue = Queue()

    name = ''
    # create a section of shared memory of the same size as the grid array
    with SharedMemoryManager() as smm:
        #shm = shared_memory.SharedMemory(create=True, size=init_grid.nbytes)
        shm = smm.SharedMemory(size=grid.nbytes)
        name = shm.name
        
        # create another ndarray of the same shape & type as grid, backed by shared memory
        shared_array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

        # copy the initialized data into the shared area of memory
        np.copyto(shared_array, grid)

    numprocesses = 12-1
    numDropsPerProcess = 10000
    numdrops = numprocesses * numDropsPerProcess

    start = time.time()

    # really just need a single process but im gonna treat it as a pool for consistency with the drop-calculation one
    plot_process = Process(target=plot_function, args=(name, shape, dtype, msg_queue))
    plot_process.start()

    with ProcessPoolExecutor(numprocesses) as exe:
        fs = [exe.submit(work_function, name, shape, dtype, numDropsPerProcess) for _ in range(numprocesses)]
        for future in as_completed(fs):
            #print('as_completed yeeted,',fs)
            #print('res:',future.result())
            pass

    # export to image after processing has ended
    shm = SharedMemory(name)
    # Create the np.ndarray from the buffer of the shared memory
    grid = np.ndarray(shape=shape, dtype=dtype, buffer=shm.buf)

    end = time.time()
    print("ELAPSED:",end-start)
    output_filename = "../../heightmaps/"+mapsrc+"mp_eroded_"+str(numdrops)+".png
    # make sure output is scaled for image
    #grid *= (255.0/grid.max())

    cv2.imwrite(output_filename, grid*256.0)
    print('wrote to',output_filename)

    # tell plotting function that it can stop
    msg_queue.put('calculation complete')
    # end the plotting process
    plot_process.join()
    

    #end = time.time()
    #print("ELAPSED :::", end-start)
    #anim()
    #mlab.show()

if __name__ == '__main__':
    main()
    #freeze_support()